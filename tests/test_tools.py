"""
Tool Testing Module for ISP Customer Service AI Agent.

Tests the actual tool wrapper functions directly:
- retrieve_isp_knowledge (RAG)
- search_web
- get_next_best_question / diagnose_connection_issue / evaluate_escalation
- get_user_info / update_user_info (CRM)
- Tool validation policy enforcement

Also provides data-driven tests from data/tool_calls.json.
"""

import json
import os
import sys
import tempfile
from typing import Dict, List, Any
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
BACKEND_ROOT = os.path.join(PROJECT_ROOT, "backend")
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from mcp_server.tools.support_workflows import (
    next_best_question,
    diagnose_issue,
    decide_escalation,
)
from app.orchestration.langgraph_engine import TOOL_VALIDATION_POLICY


# ── Helpers ──────────────────────────────────────────────────────────────

def load_tool_test_cases(filepath: str = None) -> Dict[str, List[Dict[str, Any]]]:
    """Load tool test cases from ground truth file."""
    if filepath is None:
        filepath = os.path.join(PROJECT_ROOT, "data", "tool_calls.json")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Tool test cases file not found: {filepath}")
    with open(filepath, "r") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════
# Support workflow tools — direct invocation
# ═══════════════════════════════════════════════════════════════════════════


class TestNextBestQuestion:
    """Tests for get_next_best_question / next_best_question."""

    def test_empty_state_returns_connection_type(self):
        result = next_best_question({})
        assert result["target_field"] == "connection_type"
        assert "missing_fields" in result
        assert len(result["missing_fields"]) == 5

    def test_partial_state(self):
        result = next_best_question({"connection_type": "wifi"})
        assert result["target_field"] == "lights_status"

    def test_all_fields_filled(self):
        result = next_best_question({
            "router_model": "TP-Link",
            "lights_status": "green",
            "error_message": "none",
            "connection_type": "wifi",
            "has_restarted": "yes",
        })
        assert result["target_field"] == "resolution_check"

    def test_returns_question_string(self):
        result = next_best_question({})
        assert isinstance(result["question"], str)
        assert len(result["question"]) > 10


class TestDiagnoseIssue:
    """Tests for diagnose_connection_issue / diagnose_issue."""

    def test_red_lights(self):
        result = diagnose_issue({"lights_status": "blinking red"})
        assert result["likely_cause"] == "Router signal or hardware sync issue"
        assert len(result["next_steps"]) >= 2

    def test_wifi_only(self):
        result = diagnose_issue({
            "connection_type": "wifi",
            "has_restarted": "true",
        })
        assert "Wi-Fi" in result["likely_cause"]

    def test_dns_error(self):
        result = diagnose_issue({
            "error_message": "dns probe finished",
            "has_restarted": "true",
        })
        assert "DNS" in result["likely_cause"]

    def test_not_restarted(self):
        result = diagnose_issue({"has_restarted": False})
        assert "reboot" in result["likely_cause"].lower() or "restart" in str(result["next_steps"]).lower()

    def test_general_degradation(self):
        result = diagnose_issue({
            "lights_status": "green",
            "connection_type": "ethernet",
            "has_restarted": "true",
        })
        assert result["confidence"] == "low"


class TestDecideEscalation:
    """Tests for evaluate_escalation / decide_escalation."""

    def test_long_outage_high_priority(self):
        result = decide_escalation({}, failed_steps=[], minutes_without_service=120)
        assert result["escalate"] is True
        assert result["priority"] == "high"

    def test_no_light_escalates(self):
        result = decide_escalation(
            {"lights_status": "no light"}, failed_steps=[], minutes_without_service=None
        )
        assert result["escalate"] is True
        assert result["priority"] == "high"

    def test_three_failed_steps_escalates(self):
        result = decide_escalation(
            {}, failed_steps=["restart", "factory reset", "cable swap"], minutes_without_service=None
        )
        assert result["escalate"] is True
        assert result["priority"] == "medium"

    def test_no_escalation(self):
        result = decide_escalation(
            {"lights_status": "green"}, failed_steps=[], minutes_without_service=10
        )
        assert result["escalate"] is False
        assert result["priority"] == "none"


# ═══════════════════════════════════════════════════════════════════════════
# Tool validation policy
# ═══════════════════════════════════════════════════════════════════════════


class TestToolValidationPolicy:
    """Verify that tool validation policy is correctly structured."""

    EXPECTED_INTENTS = {"memory_read", "memory_write", "factual_lookup", "troubleshooting", "general_chat"}

    def test_all_intents_present(self):
        assert set(TOOL_VALIDATION_POLICY.keys()) == self.EXPECTED_INTENTS

    def test_memory_read_only_allows_get_user_info(self):
        policy = TOOL_VALIDATION_POLICY["memory_read"]
        assert policy["allowed_tools"] == {"get_user_info"}
        assert policy["requires_tool"] is True

    def test_memory_write_only_allows_update_user_info(self):
        policy = TOOL_VALIDATION_POLICY["memory_write"]
        assert policy["allowed_tools"] == {"update_user_info"}
        assert policy["requires_tool"] is True

    def test_factual_lookup_allows_rag_and_search(self):
        policy = TOOL_VALIDATION_POLICY["factual_lookup"]
        assert "retrieve_isp_knowledge" in policy["allowed_tools"]
        assert "search_web" in policy["allowed_tools"]
        assert policy["requires_tool"] is True

    def test_troubleshooting_allows_workflow_tools(self):
        policy = TOOL_VALIDATION_POLICY["troubleshooting"]
        assert "get_next_best_question" in policy["allowed_tools"]
        assert "diagnose_connection_issue" in policy["allowed_tools"]
        assert "evaluate_escalation" in policy["allowed_tools"]
        assert "get_user_info" not in policy["allowed_tools"]

    def test_general_chat_blocks_all_tools(self):
        policy = TOOL_VALIDATION_POLICY["general_chat"]
        assert len(policy["allowed_tools"]) == 0
        assert policy["requires_tool"] is False


# ═══════════════════════════════════════════════════════════════════════════
# Data-driven tests from tool_calls.json
# ═══════════════════════════════════════════════════════════════════════════


class TestSupportWorkflowsFromData:
    """Run support workflow tests from data/tool_calls.json."""

    @pytest.fixture(autouse=True)
    def load_cases(self):
        self.test_cases = load_tool_test_cases()

    def test_next_best_question_cases(self):
        for tc in self.test_cases.get("support_workflow_tests", []):
            if tc["tool_name"] != "get_next_best_question":
                continue
            result = next_best_question(tc["input"]["known_state"])
            assert result["target_field"] == tc["expected_target_field"], (
                f"Test {tc['id']}: expected target_field={tc['expected_target_field']}, "
                f"got {result['target_field']}"
            )

    def test_diagnose_cases(self):
        for tc in self.test_cases.get("support_workflow_tests", []):
            if tc["tool_name"] != "diagnose_connection_issue":
                continue
            result = diagnose_issue(tc["input"]["known_state"])
            assert tc["expected_likely_cause"] in result["likely_cause"], (
                f"Test {tc['id']}: expected '{tc['expected_likely_cause']}' in "
                f"'{result['likely_cause']}'"
            )

    def test_escalation_cases(self):
        for tc in self.test_cases.get("support_workflow_tests", []):
            if tc["tool_name"] != "evaluate_escalation":
                continue
            result = decide_escalation(
                tc["input"]["known_state"],
                failed_steps=tc["input"].get("failed_steps", []),
                minutes_without_service=tc["input"].get("minutes_without_service"),
            )
            assert result["escalate"] == tc["expected_escalate"], (
                f"Test {tc['id']}: expected escalate={tc['expected_escalate']}, "
                f"got {result['escalate']}"
            )
            assert result["priority"] == tc["expected_priority"], (
                f"Test {tc['id']}: expected priority={tc['expected_priority']}, "
                f"got {result['priority']}"
            )


# ═══════════════════════════════════════════════════════════════════════════
# CRM tool wrappers (async tests)
# ═══════════════════════════════════════════════════════════════════════════


class TestCRMToolWrappers:
    """Test CRM tool wrappers with a temp CRM file."""

    @pytest.fixture(autouse=True)
    def setup_temp_crm(self):
        import crm.crm_store as store_root
        import app.crm.crm_store as store_app

        self._original_root_file = store_root.CRM_FILE
        self._original_app_file = store_app.CRM_FILE
        self._tmpfile = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
        json.dump({}, self._tmpfile)
        self._tmpfile.close()

        store_root.CRM_FILE = self._tmpfile.name
        store_app.CRM_FILE = self._tmpfile.name
        yield

        store_root.CRM_FILE = self._original_root_file
        store_app.CRM_FILE = self._original_app_file
        if os.path.exists(self._tmpfile.name):
            os.unlink(self._tmpfile.name)

    @pytest.mark.asyncio
    async def test_get_user_info_new_user(self):
        from app.orchestration.tools.crm_tool import get_user_info
        result = await get_user_info("new_user_xyz")
        assert "new profile" in result.lower() or "No prior record" in result

    @pytest.mark.asyncio
    async def test_update_and_get_user_info(self):
        from app.orchestration.tools.crm_tool import get_user_info, update_user_info
        await update_user_info("crm_test_user", "name", "Ali Raza")
        result = await get_user_info("crm_test_user")
        parsed = json.loads(result)
        assert parsed["name"] == "Ali Raza"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

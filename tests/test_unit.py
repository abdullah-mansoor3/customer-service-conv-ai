"""
Unit tests for ISP Customer Service AI Agent.

Tests core functions without needing a running backend:
- STATE block parsing and extraction
- Known-state normalization and merging
- Intent detection
- Support workflow tools (next_best_question, diagnose_issue, decide_escalation)
- CRM store operations
"""

import json
import os
import sys
import tempfile
import pytest

# ── Ensure project root is on sys.path ──────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

BACKEND_ROOT = os.path.join(PROJECT_ROOT, "backend")
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

from app.orchestration.langgraph_engine import (
    parse_state_block,
    extract_visible_response_text,
    merge_non_null_state,
    normalize_known_state,
    ConversationOrchestrator,
)
from app.store import DEFAULT_STATE
from mcp_server.tools.support_workflows import (
    next_best_question,
    diagnose_issue,
    decide_escalation,
)


# ═══════════════════════════════════════════════════════════════════════════
# STATE block parsing
# ═══════════════════════════════════════════════════════════════════════════


class TestParseStateBlock:
    def test_basic_state_extraction(self):
        raw = 'Sure, let me help.\n<STATE>{"router_model": "TP-Link", "lights_status": null, "error_message": null, "connection_type": "wifi", "has_restarted": null}</STATE>'
        state, text = parse_state_block(raw)
        assert state["router_model"] == "TP-Link"
        assert state["connection_type"] == "wifi"
        assert state["lights_status"] is None
        assert "Sure, let me help." in text
        assert "<STATE>" not in text

    def test_no_state_block(self):
        raw = "I can help you with that."
        state, text = parse_state_block(raw)
        assert state == {}
        assert text == "I can help you with that."

    def test_empty_input(self):
        state, text = parse_state_block("")
        assert state == {}
        assert text == ""

    def test_malformed_json(self):
        raw = "Hello\n<STATE>{bad json}</STATE>"
        state, text = parse_state_block(raw)
        assert state == {}
        assert "Hello" in text

    def test_multiple_state_blocks_uses_last(self):
        raw = (
            'First.\n<STATE>{"router_model": "A"}</STATE>\n'
            'Second.\n<STATE>{"router_model": "B"}</STATE>'
        )
        state, text = parse_state_block(raw)
        assert state["router_model"] == "B"


class TestExtractVisibleResponseText:
    def test_hides_state_block(self):
        raw = "Here is your answer.\n<STATE>{}</STATE>"
        visible = extract_visible_response_text(raw)
        assert visible == "Here is your answer."
        assert "<STATE>" not in visible

    def test_no_state_block(self):
        raw = "Hello there!"
        visible = extract_visible_response_text(raw, final=True)
        assert visible == "Hello there!"

    def test_partial_tag_guard(self):
        raw = "Answering your question<ST"
        visible = extract_visible_response_text(raw, final=False)
        assert "<ST" not in visible


# ═══════════════════════════════════════════════════════════════════════════
# State normalization and merging
# ═══════════════════════════════════════════════════════════════════════════


class TestNormalizeKnownState:
    def test_none_input(self):
        state = normalize_known_state(None)
        assert set(state.keys()) == set(DEFAULT_STATE.keys())
        assert all(v is None for v in state.values())

    def test_partial_input(self):
        state = normalize_known_state({"router_model": "Huawei"})
        assert state["router_model"] == "Huawei"
        assert state["lights_status"] is None

    def test_extra_keys_ignored(self):
        state = normalize_known_state({"router_model": "X", "unknown_key": "value"})
        assert "unknown_key" not in state


class TestMergeNonNullState:
    def test_merge_non_null(self):
        existing = {"router_model": None, "lights_status": None, "error_message": None, "connection_type": None, "has_restarted": None}
        update = {"router_model": "TP-Link", "lights_status": None}
        merged = merge_non_null_state(existing, update)
        assert merged["router_model"] == "TP-Link"
        assert merged["lights_status"] is None

    def test_null_string_ignored(self):
        existing = {"router_model": "Huawei", "lights_status": None, "error_message": None, "connection_type": None, "has_restarted": None}
        update = {"router_model": "null"}
        merged = merge_non_null_state(existing, update)
        assert merged["router_model"] == "Huawei"

    def test_preserves_existing(self):
        existing = {"router_model": "Huawei", "lights_status": "green", "error_message": None, "connection_type": None, "has_restarted": None}
        update = {"error_message": "DNS error"}
        merged = merge_non_null_state(existing, update)
        assert merged["router_model"] == "Huawei"
        assert merged["lights_status"] == "green"
        assert merged["error_message"] == "DNS error"


# ═══════════════════════════════════════════════════════════════════════════
# Intent detection
# ═══════════════════════════════════════════════════════════════════════════


class TestDetectIntent:
    def test_memory_write(self):
        assert ConversationOrchestrator._detect_intent("My name is Ahmed", {}) == "memory_write"

    def test_memory_read(self):
        assert ConversationOrchestrator._detect_intent("What is my name?", {}) == "memory_read"

    def test_factual_lookup(self):
        assert ConversationOrchestrator._detect_intent("What are PTCL package prices?", {}) == "factual_lookup"

    def test_troubleshooting(self):
        assert ConversationOrchestrator._detect_intent("My internet is not working", {}) == "troubleshooting"

    def test_general_chat(self):
        assert ConversationOrchestrator._detect_intent("Hello there", {}) == "general_chat"

    def test_empty_message(self):
        assert ConversationOrchestrator._detect_intent("", {}) == "general_chat"

    def test_nayatel_triggers_factual(self):
        assert ConversationOrchestrator._detect_intent("What is the Nayatel helpline?", {}) == "factual_lookup"

    def test_router_triggers_troubleshooting(self):
        assert ConversationOrchestrator._detect_intent("My router keeps disconnecting", {}) == "troubleshooting"


# ═══════════════════════════════════════════════════════════════════════════
# Support workflow tools
# ═══════════════════════════════════════════════════════════════════════════


class TestNextBestQuestion:
    def test_empty_state_asks_connection_type(self):
        result = next_best_question({})
        assert result["target_field"] == "connection_type"

    def test_with_connection_type_asks_lights(self):
        result = next_best_question({"connection_type": "wifi"})
        assert result["target_field"] == "lights_status"

    def test_all_filled_asks_resolution(self):
        result = next_best_question({
            "router_model": "TP-Link",
            "lights_status": "green",
            "error_message": "none",
            "connection_type": "wifi",
            "has_restarted": "yes",
        })
        assert result["target_field"] == "resolution_check"
        assert result["missing_fields"] == []


class TestDiagnoseIssue:
    def test_red_lights_diagnosis(self):
        result = diagnose_issue({"lights_status": "blinking red"})
        assert "signal" in result["likely_cause"].lower() or "hardware" in result["likely_cause"].lower()

    def test_wifi_only_diagnosis(self):
        result = diagnose_issue({"connection_type": "wifi", "has_restarted": "true"})
        assert "wi-fi" in result["likely_cause"].lower()

    def test_dns_diagnosis(self):
        result = diagnose_issue({"error_message": "DNS probe finished no internet", "has_restarted": "true"})
        assert "dns" in result["likely_cause"].lower()

    def test_not_restarted(self):
        result = diagnose_issue({"has_restarted": False})
        assert "reboot" in result["likely_cause"].lower() or "restart" in result["next_steps"][0].lower()


class TestDecideEscalation:
    def test_long_outage_escalates(self):
        result = decide_escalation({}, failed_steps=[], minutes_without_service=180)
        assert result["escalate"] is True
        assert result["priority"] == "high"

    def test_red_lights_escalate(self):
        result = decide_escalation({"lights_status": "red"}, failed_steps=[], minutes_without_service=None)
        assert result["escalate"] is True

    def test_many_failed_steps_escalate(self):
        result = decide_escalation({}, failed_steps=["restarted", "factory reset", "cable swap"], minutes_without_service=None)
        assert result["escalate"] is True

    def test_no_escalation_needed(self):
        result = decide_escalation({"lights_status": "green"}, failed_steps=[], minutes_without_service=10)
        assert result["escalate"] is False


# ═══════════════════════════════════════════════════════════════════════════
# CRM store (uses temp file)
# ═══════════════════════════════════════════════════════════════════════════


class TestCRMStore:
    def test_create_and_get_user(self):
        import crm.crm_store as store
        original_file = store.CRM_FILE
        try:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
                json.dump({}, f)
                store.CRM_FILE = f.name

            user = store.create_or_get_user("unit_test_user")
            assert user["user_id"] == "unit_test_user"
            assert user["name"] is None

            fetched = store.get_user("unit_test_user")
            assert fetched is not None
            assert fetched["user_id"] == "unit_test_user"
        finally:
            store.CRM_FILE = original_file
            if os.path.exists(f.name):
                os.unlink(f.name)

    def test_update_user(self):
        import crm.crm_store as store
        original_file = store.CRM_FILE
        try:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
                json.dump({}, f)
                store.CRM_FILE = f.name

            store.create_or_get_user("update_test")
            updated = store.update_user("update_test", "name", "Ahmed Khan")
            assert updated["name"] == "Ahmed Khan"

            updated = store.update_user("update_test", "preferred_language", "Urdu")
            assert updated["preferences"]["preferred_language"] == "Urdu"
        finally:
            store.CRM_FILE = original_file
            if os.path.exists(f.name):
                os.unlink(f.name)

    def test_get_nonexistent_user(self):
        import crm.crm_store as store
        original_file = store.CRM_FILE
        try:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
                json.dump({}, f)
                store.CRM_FILE = f.name

            result = store.get_user("does_not_exist")
            assert result is None
        finally:
            store.CRM_FILE = original_file
            if os.path.exists(f.name):
                os.unlink(f.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
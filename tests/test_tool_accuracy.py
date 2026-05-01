"""
Tool Accuracy Testing Module for ISP Customer Service AI Agent.

Tests the LLM's ability to correctly select and invoke tools via WebSocket:
- Sends ISP-specific user utterances designed to trigger specific tools
- Captures tool invocations from status events
- Verifies correct tool is triggered

Requires a running backend server at ws://localhost:8000/ws/chat.
"""

import asyncio
import json
import logging
import os
import sys
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import websockets
except ImportError:
    websockets = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WS_URI = "ws://localhost:8000/ws/chat"


class ToolAccuracyTestCase:
    """Represents a single tool accuracy test case."""

    def __init__(
        self,
        test_id: str,
        user_utterance: str,
        expected_tools: List[str],
        description: str = "",
        tool_hints: List[str] = None,
    ):
        self.test_id = test_id
        self.user_utterance = user_utterance
        self.expected_tools = expected_tools
        self.description = description
        self.tool_hints = tool_hints or []


class WebSocketToolAccuracyTester:
    """WebSocket client for testing tool accuracy against the ISP agent."""

    def __init__(self, uri: str = WS_URI, timeout: int = 120):
        self.uri = uri
        self.timeout = timeout

    async def test_tool_accuracy(self, test_case: ToolAccuracyTestCase) -> Dict[str, Any]:
        """Test a single tool accuracy case via WebSocket."""
        result = {
            "test_id": test_case.test_id,
            "user_utterance": test_case.user_utterance,
            "expected_tools": test_case.expected_tools,
            "passed": False,
            "actual_tools": [],
            "errors": [],
        }

        session_id = f"tool_accuracy_{test_case.test_id}_{uuid.uuid4().hex[:8]}"

        try:
            async with websockets.connect(self.uri, close_timeout=30) as ws:
                payload = {
                    "message": test_case.user_utterance,
                    "session_id": session_id,
                }
                if test_case.tool_hints:
                    payload["tool_hints"] = test_case.tool_hints

                await ws.send(json.dumps(payload))

                tool_names_seen = set()
                done = False

                while not done:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=self.timeout)
                        data = json.loads(raw)

                        if data.get("type") == "status" and data.get("tool_name"):
                            tool_names_seen.add(data["tool_name"])

                        if data.get("type") == "token" and data.get("done"):
                            done = True

                    except asyncio.TimeoutError:
                        result["errors"].append("Timeout waiting for response")
                        done = True

                result["actual_tools"] = sorted(tool_names_seen)

                if not test_case.expected_tools:
                    # General chat should not trigger tools.
                    if not tool_names_seen:
                        result["passed"] = True
                    else:
                        result["errors"].append(
                            f"Expected no tool calls, got {sorted(tool_names_seen)}"
                        )
                elif any(tool in tool_names_seen for tool in test_case.expected_tools):
                    # Pass if at least one acceptable tool was invoked.
                    result["passed"] = True
                else:
                    result["errors"].append(
                        f"Expected one of {test_case.expected_tools}, "
                        f"got {sorted(tool_names_seen)}"
                    )

        except Exception as e:
            result["errors"].append(f"Connection error: {e}")

        return result

    async def run_test_suite(
        self, test_cases: List[ToolAccuracyTestCase]
    ) -> Dict[str, Any]:
        """Run a suite of tool accuracy tests."""
        results = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(test_cases),
            "passed": 0,
            "failed": 0,
            "test_results": [],
        }

        for test_case in test_cases:
            result = await self.test_tool_accuracy(test_case)
            results["test_results"].append(result)

            if result["passed"]:
                results["passed"] += 1
            else:
                results["failed"] += 1

            logger.info(
                f"Test {test_case.test_id}: "
                f"{'PASSED' if result['passed'] else 'FAILED'} "
                f"(tools: {result['actual_tools']})"
            )

            await asyncio.sleep(1)  # Avoid overwhelming the server

        results["pass_rate"] = (
            results["passed"] / results["total_tests"]
            if results["total_tests"] > 0
            else 0
        )
        return results


# ── ISP-specific test cases ──────────────────────────────────────────────

TOOL_ACCURACY_TEST_CASES = [
    ToolAccuracyTestCase(
        test_id="tool_rag_helpline",
        user_utterance="What is the PTCL helpline number?",
        expected_tools=["retrieve_isp_knowledge"],
        description="RAG should be triggered for PTCL factual lookup",
    ),
    ToolAccuracyTestCase(
        test_id="tool_rag_packages",
        user_utterance="What are Nayatel internet package prices?",
        expected_tools=["retrieve_isp_knowledge"],
        description="RAG should be triggered for pricing questions",
    ),
    ToolAccuracyTestCase(
        test_id="tool_crm_write",
        user_utterance="My name is Ahmed Khan, please remember it.",
        expected_tools=["update_user_info"],
        description="CRM update tool should be triggered for name storage",
    ),
    ToolAccuracyTestCase(
        test_id="tool_crm_read",
        user_utterance="What is my name? Do you remember me?",
        expected_tools=["get_user_info"],
        description="CRM get tool should be triggered for profile recall",
    ),
    ToolAccuracyTestCase(
        test_id="tool_web_search",
        user_utterance="Search the web for latest Nayatel router brand",
        expected_tools=["search_web"],
        description="Web search should be triggered for explicit search requests",
        tool_hints=["websearch"],
    ),
    ToolAccuracyTestCase(
        test_id="tool_troubleshoot",
        user_utterance="My internet is not working, router lights are red, I've restarted it.",
        expected_tools=["diagnose_connection_issue", "get_next_best_question"],
        description="Troubleshooting tool should be triggered for connectivity issues",
    ),
    ToolAccuracyTestCase(
        test_id="tool_no_tool_chat",
        user_utterance="Hello, how are you?",
        expected_tools=[],
        description="No tool should be triggered for general greetings",
    ),
]


# ── Pytest integration ───────────────────────────────────────────────────

@pytest.mark.skipif(
    not os.getenv("RUN_INTEGRATION_TESTS"),
    reason="Requires running backend (set RUN_INTEGRATION_TESTS=1)",
)
@pytest.mark.skipif(websockets is None, reason="websockets not installed")
@pytest.mark.asyncio
async def test_tool_accuracy_suite():
    """Run complete tool accuracy test suite."""
    tester = WebSocketToolAccuracyTester(timeout=120)
    results = await tester.run_test_suite(TOOL_ACCURACY_TEST_CASES)

    print("\n" + json.dumps(results, indent=2))
    # Allow the no-tool test to not count against pass rate
    tool_tests = [r for r in results["test_results"] if r["expected_tools"]]
    if tool_tests:
        pass_rate = sum(1 for r in tool_tests if r["passed"]) / len(tool_tests)
        assert pass_rate >= 0.5, f"Tool accuracy pass rate too low: {pass_rate:.2%}"


# ── CLI entry point ──────────────────────────────────────────────────────

async def main():
    """Run tool accuracy tests from command line."""
    tester = WebSocketToolAccuracyTester(timeout=120)
    results = await tester.run_test_suite(TOOL_ACCURACY_TEST_CASES)

    print("\n" + "=" * 60)
    print("TOOL ACCURACY TEST RESULTS")
    print("=" * 60)
    print(json.dumps(results, indent=2))

    os.makedirs("eval_reports", exist_ok=True)
    with open("eval_reports/tool_accuracy_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nPass Rate: {results['pass_rate']:.2%}")


if __name__ == "__main__":
    asyncio.run(main())

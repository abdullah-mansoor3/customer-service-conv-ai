"""
Tool Accuracy Testing Module

Tests the LLM's ability to correctly identify and invoke tools via WebSocket:
- Sends specific user utterances designed to trigger tools
- Captures tool invocations from the chatbot
- Verifies correct tool is triggered with expected arguments
"""

import asyncio
import json
import logging
import websockets
from typing import List, Dict, Any, Optional
from datetime import datetime
import pytest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToolAccuracyTestCase:
    """Represents a single tool accuracy test case."""
    
    def __init__(
        self,
        test_id: str,
        user_utterance: str,
        expected_tool: str,
        expected_args: Dict[str, Any],
        description: str = ""
    ):
        self.test_id = test_id
        self.user_utterance = user_utterance
        self.expected_tool = expected_tool
        self.expected_args = expected_args
        self.description = description


class WebSocketToolAccuracyTester:
    """WebSocket client for testing tool accuracy."""
    
    def __init__(self, uri: str = "ws://localhost:8000/ws", timeout: int = 10):
        self.uri = uri
        self.timeout = timeout
        self.websocket = None
        self.tool_calls_captured = []
        self.messages = []
    
    async def connect(self) -> bool:
        """Connect to WebSocket server."""
        try:
            self.websocket = await asyncio.wait_for(
                websockets.connect(self.uri),
                timeout=self.timeout
            )
            logger.info(f"Connected to {self.uri}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from WebSocket server."""
        if self.websocket:
            await self.websocket.close()
            logger.info("Disconnected")
    
    async def send_message(self, content: str, session_id: str = "test_session") -> bool:
        """Send a message to the chatbot."""
        try:
            message = {
                "type": "message",
                "content": content,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }
            await asyncio.wait_for(
                self.websocket.send(json.dumps(message)),
                timeout=self.timeout
            )
            logger.info(f"Sent: {content}")
            return True
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False
    
    async def receive_response(self) -> Optional[Dict[str, Any]]:
        """Receive a response from the chatbot."""
        try:
            response_str = await asyncio.wait_for(
                self.websocket.recv(),
                timeout=self.timeout
            )
            response = json.loads(response_str)
            self.messages.append(response)
            
            # Check if response contains tool calls
            if "tool_calls" in response:
                self.tool_calls_captured.extend(response["tool_calls"])
            
            logger.info(f"Received: {response}")
            return response
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for response")
            return None
        except Exception as e:
            logger.error(f"Failed to receive response: {e}")
            return None
    
    async def test_tool_accuracy(self, test_case: ToolAccuracyTestCase) -> Dict[str, Any]:
        """
        Test a single tool accuracy case.
        
        Args:
            test_case: ToolAccuracyTestCase instance
        
        Returns:
            Test result dictionary
        """
        result = {
            "test_id": test_case.test_id,
            "user_utterance": test_case.user_utterance,
            "expected_tool": test_case.expected_tool,
            "expected_args": test_case.expected_args,
            "passed": False,
            "actual_tool_calls": [],
            "errors": []
        }
        
        # Clear previous tool calls
        self.tool_calls_captured = []
        
        # Send user utterance
        if not await self.send_message(test_case.user_utterance):
            result["errors"].append("Failed to send message")
            return result
        
        # Receive response(s) - may need multiple receives
        responses_received = 0
        max_responses = 5
        
        while responses_received < max_responses:
            response = await self.receive_response()
            if response is None:
                break
            responses_received += 1
            
            # Check if we got tool calls
            if self.tool_calls_captured:
                break
            
            # Small delay between requests
            await asyncio.sleep(0.1)
        
        # Analyze captured tool calls
        result["actual_tool_calls"] = self.tool_calls_captured
        
        if not self.tool_calls_captured:
            result["errors"].append("No tool calls captured")
            return result
        
        # Verify tool accuracy
        for tool_call in self.tool_calls_captured:
            tool_name = tool_call.get("name") or tool_call.get("tool")
            tool_args = tool_call.get("arguments") or tool_call.get("args", {})
            
            if tool_name == test_case.expected_tool:
                # Check if arguments match (allow partial match)
                args_match = all(
                    tool_args.get(key) == value
                    for key, value in test_case.expected_args.items()
                )
                
                if args_match:
                    result["passed"] = True
                else:
                    result["errors"].append(
                        f"Arguments mismatch. Expected: {test_case.expected_args}, "
                        f"Got: {tool_args}"
                    )
            else:
                result["errors"].append(
                    f"Tool mismatch. Expected: {test_case.expected_tool}, "
                    f"Got: {tool_name}"
                )
        
        return result
    
    async def run_test_suite(
        self,
        test_cases: List[ToolAccuracyTestCase]
    ) -> Dict[str, Any]:
        """
        Run a suite of tool accuracy tests.
        
        Args:
            test_cases: List of ToolAccuracyTestCase instances
        
        Returns:
            Summary of all test results
        """
        if not await self.connect():
            return {"error": "Failed to connect to WebSocket"}
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": len(test_cases),
            "passed": 0,
            "failed": 0,
            "test_results": []
        }
        
        for test_case in test_cases:
            try:
                result = await self.test_tool_accuracy(test_case)
                results["test_results"].append(result)
                
                if result["passed"]:
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                
                logger.info(f"Test {test_case.test_id}: {'PASSED' if result['passed'] else 'FAILED'}")
            except Exception as e:
                logger.error(f"Error running test {test_case.test_id}: {e}")
                results["failed"] += 1
                results["test_results"].append({
                    "test_id": test_case.test_id,
                    "passed": False,
                    "errors": [str(e)]
                })
        
        await self.disconnect()
        
        results["pass_rate"] = (
            results["passed"] / results["total_tests"]
            if results["total_tests"] > 0
            else 0
        )
        
        return results


# Predefined test cases for common tool triggers

TOOL_ACCURACY_TEST_CASES = [
    ToolAccuracyTestCase(
        test_id="tool_1",
        user_utterance="What is the refund policy?",
        expected_tool="rag_tool",
        expected_args={"query": "refund policy"},
        description="RAG should be triggered for general questions"
    ),
    ToolAccuracyTestCase(
        test_id="tool_2",
        user_utterance="Get customer details for ID 123",
        expected_tool="crm_tool",
        expected_args={"action": "get_customer", "customer_id": "123"},
        description="CRM tool should be triggered for customer queries"
    ),
    ToolAccuracyTestCase(
        test_id="tool_3",
        user_utterance="Search for latest AI news",
        expected_tool="web_search_tool",
        expected_args={"query": "latest AI news"},
        description="Web search should be triggered for news queries"
    ),
    ToolAccuracyTestCase(
        test_id="tool_4",
        user_utterance="I have a login issue, can you help?",
        expected_tool="support_workflows_tool",
        expected_args={"issue": "login"},
        description="Support workflows should be triggered for support issues"
    ),
    ToolAccuracyTestCase(
        test_id="tool_5",
        user_utterance="Send me an email confirmation",
        expected_tool="integrations_tool",
        expected_args={"service": "email", "action": "send"},
        description="Integrations tool should be triggered for communication actions"
    ),
]


# Pytest fixtures

@pytest.fixture
async def websocket_tester():
    """Fixture providing WebSocket tester."""
    tester = WebSocketToolAccuracyTester()
    yield tester
    await tester.disconnect()


# Async test functions

@pytest.mark.asyncio
async def test_tool_accuracy_suite(websocket_tester):
    """Run complete tool accuracy test suite."""
    results = await websocket_tester.run_test_suite(TOOL_ACCURACY_TEST_CASES)
    
    # Print results
    print("\n" + json.dumps(results, indent=2))
    
    # Assert majority of tests pass
    assert results.get("pass_rate", 0) >= 0.7, f"Pass rate too low: {results['pass_rate']}"


@pytest.mark.asyncio
async def test_single_tool_accuracy(websocket_tester):
    """Test a single tool accuracy case."""
    test_case = TOOL_ACCURACY_TEST_CASES[0]
    result = await websocket_tester.test_tool_accuracy(test_case)
    
    print("\n" + json.dumps(result, indent=2))
    
    assert result["passed"], f"Test failed: {result['errors']}"


# Command-line interface

async def main():
    """Run tool accuracy tests from command line."""
    tester = WebSocketToolAccuracyTester()
    results = await tester.run_test_suite(TOOL_ACCURACY_TEST_CASES)
    
    print("\n" + "="*60)
    print("TOOL ACCURACY TEST RESULTS")
    print("="*60)
    print(json.dumps(results, indent=2))
    
    # Write results to file
    with open("eval_reports/tool_accuracy_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to eval_reports/tool_accuracy_results.json")
    print(f"Pass Rate: {results['pass_rate']:.2%}")


if __name__ == "__main__":
    asyncio.run(main())

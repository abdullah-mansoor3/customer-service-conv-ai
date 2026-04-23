"""
CRM and External Tools Testing Module

Direct invocation tests for tool functions:
- CRM tool (CRUD operations)
- Web search tool
- Support workflows tool
- Integrations tool
"""

import json
import os
from typing import List, Dict, Any
import pytest


def load_tool_test_cases(filepath: str = "data/tool_calls.json") -> Dict[str, List[Dict[str, Any]]]:
    """Load tool test cases from ground truth file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Tool test cases file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        return json.load(f)


class MockCRMTool:
    """Mock CRM Tool for testing."""
    
    def __init__(self):
        self.customers = {
            "123": {"id": "123", "name": "John Doe", "email": "john@example.com"},
            "124": {"id": "124", "name": "Jane Smith", "email": "jane@example.com"}
        }
        self.next_id = 125
    
    def get_customer(self, customer_id: str) -> Dict[str, Any]:
        """Get customer by ID."""
        if customer_id in self.customers:
            return self.customers[customer_id]
        raise ValueError(f"Customer {customer_id} not found")
    
    def create_customer(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Create a new customer."""
        customer_id = str(self.next_id)
        self.customers[customer_id] = {"id": customer_id, **data}
        self.next_id += 1
        return {"customer_id": customer_id}
    
    def update_customer(self, customer_id: str, data: Dict[str, Any]) -> str:
        """Update customer data."""
        if customer_id not in self.customers:
            raise ValueError(f"Customer {customer_id} not found")
        self.customers[customer_id].update(data)
        return "Updated successfully"
    
    def delete_customer(self, customer_id: str) -> str:
        """Delete a customer."""
        if customer_id not in self.customers:
            raise ValueError(f"Customer {customer_id} not found")
        del self.customers[customer_id]
        return "Deleted successfully"


class MockWebSearchTool:
    """Mock Web Search Tool for testing."""
    
    def search(self, query: str) -> str:
        """Perform web search."""
        return f"Search results for {query}"


class MockSupportWorkflowTool:
    """Mock Support Workflow Tool for testing."""
    
    def get_workflow(self, issue: str) -> str:
        """Get support workflow for issue."""
        workflows = {
            "login problem": "Workflow steps for login issue",
            "billing question": "Billing support workflow"
        }
        return workflows.get(issue, "Unknown workflow")


class MockIntegrationsTool:
    """Mock Integrations Tool for testing."""
    
    def execute_integration(self, service: str, action: str) -> str:
        """Execute an integration action."""
        if service == "email" and action == "send":
            return "Email sent"
        elif service == "slack" and action == "notify":
            return "Slack notification sent"
        raise ValueError(f"Unknown integration: {service} {action}")


# Test functions

def test_crm_get_customer():
    """Test CRM get_customer operation."""
    crm = MockCRMTool()
    result = crm.get_customer("123")
    assert result["name"] == "John Doe"
    assert result["email"] == "john@example.com"


def test_crm_create_customer():
    """Test CRM create_customer operation."""
    crm = MockCRMTool()
    result = crm.create_customer({"name": "Test User", "email": "test@example.com"})
    assert "customer_id" in result
    
    # Verify creation
    customer_id = result["customer_id"]
    customer = crm.get_customer(customer_id)
    assert customer["name"] == "Test User"


def test_crm_update_customer():
    """Test CRM update_customer operation."""
    crm = MockCRMTool()
    result = crm.update_customer("123", {"phone": "555-1234"})
    assert result == "Updated successfully"
    
    # Verify update
    customer = crm.get_customer("123")
    assert customer.get("phone") == "555-1234"


def test_crm_delete_customer():
    """Test CRM delete_customer operation."""
    crm = MockCRMTool()
    result = crm.delete_customer("124")
    assert result == "Deleted successfully"
    
    # Verify deletion
    with pytest.raises(ValueError):
        crm.get_customer("124")


def test_crm_operations_from_ground_truth():
    """Test CRM operations using ground truth test cases."""
    crm = MockCRMTool()
    test_cases = load_tool_test_cases()
    
    for test_case in test_cases.get("crm_tests", []):
        test_id = test_case["id"]
        input_data = test_case["input"]
        expected = test_case["expected"]
        
        if input_data["action"] == "get_customer":
            result = crm.get_customer(input_data["customer_id"])
            if isinstance(expected, dict):
                for key in expected:
                    assert result.get(key) == expected[key], f"Test {test_id} failed"
        
        elif input_data["action"] == "create_customer":
            result = crm.create_customer(input_data["data"])
            assert "customer_id" in result, f"Test {test_id} failed"
        
        elif input_data["action"] == "update_customer":
            result = crm.update_customer(input_data["customer_id"], input_data["data"])
            assert result == expected, f"Test {test_id} failed"


def test_web_search_tool():
    """Test web search tool."""
    ws = MockWebSearchTool()
    result = ws.search("latest news")
    assert "Search results" in result


def test_support_workflow_tool():
    """Test support workflow tool."""
    sw = MockSupportWorkflowTool()
    result = sw.get_workflow("login problem")
    assert "login issue" in result.lower()


def test_integrations_tool():
    """Test integrations tool."""
    integ = MockIntegrationsTool()
    result = integ.execute_integration("email", "send")
    assert result == "Email sent"


def test_integrations_from_ground_truth():
    """Test integrations using ground truth test cases."""
    integ = MockIntegrationsTool()
    test_cases = load_tool_test_cases()
    
    for test_case in test_cases.get("integrations_tests", []):
        test_id = test_case["id"]
        input_data = test_case["input"]
        expected = test_case["expected"]
        
        result = integ.execute_integration(input_data["service"], input_data["action"])
        assert result == expected, f"Integration test {test_id} failed: {result} != {expected}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

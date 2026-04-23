import pytest
from backend.app.crm.crm_tool import CRMTool  # Assuming this exists
from backend.app.orchestration.tools.rag_tool import RAGTool  # Assuming

def test_crm_tool_get_customer():
    crm = CRMTool()
    result = crm.get_customer("123")
    assert result is not None
    assert "name" in result

def test_rag_tool_query():
    rag = RAGTool()
    result = rag.query("What is RAG?")
    assert isinstance(result, str)
    assert len(result) > 0

# Add more unit tests as needed
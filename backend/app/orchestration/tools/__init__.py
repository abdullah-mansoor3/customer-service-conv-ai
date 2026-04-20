"""Tool definitions used by LangGraph orchestration."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.orchestration.tools.crm_tool import (
	GET_USER_INFO_TOOL_SPEC,
	UPDATE_USER_INFO_TOOL_SPEC,
	get_user_info,
	update_user_info,
)
from app.orchestration.tools.rag_tool import RETRIEVE_ISP_KNOWLEDGE_TOOL_SPEC, retrieve_isp_knowledge
from app.orchestration.tools.support_tools import (
	DIAGNOSE_CONNECTION_ISSUE_TOOL_SPEC,
	EVALUATE_ESCALATION_TOOL_SPEC,
	GET_NEXT_BEST_QUESTION_TOOL_SPEC,
	SEARCH_WEB_TOOL_SPEC,
	diagnose_connection_issue,
	evaluate_escalation,
	get_next_best_question,
	search_web,
)


class RetrieveIspKnowledgeArgs(BaseModel):
	query: str = Field(description="The user question or troubleshooting issue to retrieve knowledge for.")


class GetUserInfoArgs(BaseModel):
	user_id: str = Field(description="Session or user identifier.")


class UpdateUserInfoArgs(BaseModel):
	user_id: str = Field(description="Session or user identifier.")
	field: str = Field(description="Profile field to update (for example: name, contact, preferred_language).")
	value: str = Field(description="New value for the target field.")


class SearchWebArgs(BaseModel):
	query: str = Field(description="Search query text.")
	max_results: int = Field(default=5, description="Maximum number of results to include (1-10).")


class StateOnlyArgs(BaseModel):
	known_state: dict[str, Any] = Field(description="Known support state dictionary.")


class EvaluateEscalationArgs(BaseModel):
	known_state: dict[str, Any] = Field(description="Known support state dictionary.")
	failed_steps: list[str] = Field(default_factory=list, description="Already attempted troubleshooting steps that failed.")
	minutes_without_service: int | None = Field(default=None, description="Approximate outage duration in minutes.")

ALL_TOOL_SPECS = [
	RETRIEVE_ISP_KNOWLEDGE_TOOL_SPEC,
	GET_USER_INFO_TOOL_SPEC,
	UPDATE_USER_INFO_TOOL_SPEC,
	SEARCH_WEB_TOOL_SPEC,
	GET_NEXT_BEST_QUESTION_TOOL_SPEC,
	DIAGNOSE_CONNECTION_ISSUE_TOOL_SPEC,
	EVALUATE_ESCALATION_TOOL_SPEC,
]

ALL_TOOLS_BY_NAME = {
	"retrieve_isp_knowledge": retrieve_isp_knowledge,
	"get_user_info": get_user_info,
	"update_user_info": update_user_info,
	"search_web": search_web,
	"get_next_best_question": get_next_best_question,
	"diagnose_connection_issue": diagnose_connection_issue,
	"evaluate_escalation": evaluate_escalation,
}

ALL_LANGCHAIN_TOOLS = [
	StructuredTool.from_function(
		func=retrieve_isp_knowledge,
		name="retrieve_isp_knowledge",
		description=(
			"Retrieve ISP troubleshooting, router/device details, setup guides, and provider-specific "
			"technical context for factual customer support questions."
		),
		args_schema=RetrieveIspKnowledgeArgs,
	),
	StructuredTool.from_function(
		func=get_user_info,
		coroutine=get_user_info,
		name="get_user_info",
		description="Retrieve stored information about the user profile and recent interactions.",
		args_schema=GetUserInfoArgs,
	),
	StructuredTool.from_function(
		func=update_user_info,
		coroutine=update_user_info,
		name="update_user_info",
		description="Store or update one user profile field such as name, contact, or preferences.",
		args_schema=UpdateUserInfoArgs,
	),
	StructuredTool.from_function(
		func=search_web,
		name="search_web",
		description="Search the public web for current ISP/networking information.",
		args_schema=SearchWebArgs,
	),
	StructuredTool.from_function(
		func=get_next_best_question,
		name="get_next_best_question",
		description="Suggest one best follow-up question based on missing troubleshooting state.",
		args_schema=StateOnlyArgs,
	),
	StructuredTool.from_function(
		func=diagnose_connection_issue,
		name="diagnose_connection_issue",
		description="Estimate likely root cause and next troubleshooting steps.",
		args_schema=StateOnlyArgs,
	),
	StructuredTool.from_function(
		func=evaluate_escalation,
		name="evaluate_escalation",
		description="Decide whether to escalate the support issue and priority level.",
		args_schema=EvaluateEscalationArgs,
	),
]

__all__ = [
	"RETRIEVE_ISP_KNOWLEDGE_TOOL_SPEC",
	"retrieve_isp_knowledge",
	"GET_USER_INFO_TOOL_SPEC",
	"UPDATE_USER_INFO_TOOL_SPEC",
	"get_user_info",
	"update_user_info",
	"SEARCH_WEB_TOOL_SPEC",
	"GET_NEXT_BEST_QUESTION_TOOL_SPEC",
	"DIAGNOSE_CONNECTION_ISSUE_TOOL_SPEC",
	"EVALUATE_ESCALATION_TOOL_SPEC",
	"search_web",
	"get_next_best_question",
	"diagnose_connection_issue",
	"evaluate_escalation",
	"ALL_TOOL_SPECS",
	"ALL_TOOLS_BY_NAME",
	"ALL_LANGCHAIN_TOOLS",
]

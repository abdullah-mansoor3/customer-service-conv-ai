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
	query: str = Field(
		description=(
			"Best-first factual lookup query for ISP support. Include provider name (for example PTCL/Nayatel) "
			"and target fact (router model, package, helpline, setup, troubleshooting step)."
		)
	)


class GetUserInfoArgs(BaseModel):
	user_id: str = Field(
		description="Session/user identifier used to fetch CRM profile memory (name, contact, preferences)."
	)


class UpdateUserInfoArgs(BaseModel):
	user_id: str = Field(description="Session/user identifier for CRM persistence.")
	field: str = Field(
		description=(
			"Field to persist. Prefer 'name' or 'contact' for top-level fields; other keys are stored under "
			"preferences (for example preferred_language)."
		)
	)
	value: str = Field(
		description="Value to store. Only persist values explicitly stated by the user or requested to be remembered."
	)


class SearchWebArgs(BaseModel):
	query: str = Field(
		description=(
			"Web query for current/time-sensitive information or when KB evidence is missing. "
			"Keep it concise and provider-specific."
		)
	)
	max_results: int = Field(default=3, ge=1, le=5, description="Maximum results to return (1-5).")


class StateOnlyArgs(BaseModel):
	known_state: dict[str, Any] = Field(
		description=(
			"Troubleshooting state object with keys: router_model, lights_status, error_message, "
			"connection_type, has_restarted."
		)
	)


class EvaluateEscalationArgs(BaseModel):
	known_state: dict[str, Any] = Field(
		description=(
			"Troubleshooting state object with keys: router_model, lights_status, error_message, "
			"connection_type, has_restarted."
		)
	)
	failed_steps: list[str] = Field(
		default_factory=list,
		description="Already-attempted troubleshooting actions that did not resolve the issue.",
	)
	minutes_without_service: int | None = Field(
		default=None,
		ge=0,
		description="Approximate outage duration in minutes (non-negative).",
	)

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
			"Primary knowledge tool for ISP/provider facts from local KB. Use for PTCL/Nayatel router brands, "
			"packages, helplines, setup and troubleshooting guidance before answering factual support queries."
		),
		args_schema=RetrieveIspKnowledgeArgs,
	),
	StructuredTool.from_function(
		func=get_user_info,
		coroutine=get_user_info,
		name="get_user_info",
		description=(
			"Use for personalization or memory recall: fetch saved user profile values (name/contact/preferences). "
			"Do not use for ISP factual lookup."
		),
		args_schema=GetUserInfoArgs,
	),
	StructuredTool.from_function(
		func=update_user_info,
		coroutine=update_user_info,
		name="update_user_info",
		description=(
			"Use only when user explicitly shares or asks to remember profile info (name/contact/preferences). "
			"Do not use for troubleshooting facts."
		),
		args_schema=UpdateUserInfoArgs,
	),
	StructuredTool.from_function(
		func=search_web,
		name="search_web",
		description=(
			"Secondary knowledge tool for current/time-sensitive public info when KB is insufficient or user asks "
			"explicitly to search online."
		),
		args_schema=SearchWebArgs,
	),
	StructuredTool.from_function(
		func=get_next_best_question,
		name="get_next_best_question",
		description=(
			"Workflow tool: generate the single highest-value next troubleshooting question from missing state fields."
		),
		args_schema=StateOnlyArgs,
	),
	StructuredTool.from_function(
		func=diagnose_connection_issue,
		name="diagnose_connection_issue",
		description=(
			"Workflow tool: infer likely connection root cause and immediate next actions from known troubleshooting state."
		),
		args_schema=StateOnlyArgs,
	),
	StructuredTool.from_function(
		func=evaluate_escalation,
		name="evaluate_escalation",
		description=(
			"Workflow tool: decide escalation need/priority after troubleshooting attempts and outage duration."
		),
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

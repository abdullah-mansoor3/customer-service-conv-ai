"""LangChain + LangGraph orchestration engine.

This module centralizes dialogue orchestration so routers no longer manually
construct prompts or parse model streams. The graph is intentionally simple
for this migration step (route -> dialogue -> parse) and can be extended with
RAG/tool branches in future steps.
"""

from __future__ import annotations

import logging
import json
import inspect
import re
from dataclasses import dataclass, field
from typing import Annotated, Any, AsyncIterator, Literal, TypedDict, cast

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

try:
    from langgraph.config import get_stream_writer
except ImportError:  # pragma: no cover - defensive fallback for older langgraph
    def get_stream_writer():
        return None

from app.config import (
    DEFAULT_SYSTEM_PROMPT,
    FEW_SHOT_EXAMPLES,
    LLAMA_API_KEY,
    LLAMA_MODEL_NAME,
    LLAMA_SERVER_URL,
    LLAMA_TEMPERATURE,
    LLAMA_TOOL_PLANNER_MAX_STEPS,
    LLAMA_TOOL_PLANNER_MAX_TOKENS,
    LLAMA_TOOL_PLANNER_TIMEOUT_SEC,
    LLAMA_TIMEOUT_SEC,
    LLAMA_TOP_P,
    MAX_HISTORY,
    PLANNER_PROMPT,
)
from app.orchestration.tools import ALL_LANGCHAIN_TOOLS, ALL_TOOLS_BY_NAME
from app.store import DEFAULT_STATE

logger = logging.getLogger(__name__)

IntentType = Literal[
    "memory_read",
    "memory_write",
    "factual_lookup",
    "troubleshooting",
    "general_chat",
]

TOOL_VALIDATION_POLICY: dict[IntentType, dict[str, Any]] = {
    "memory_read": {
        "allowed_tools": {"get_user_info"},
        "blocked_tools": {
            "update_user_info",
            "search_web",
            "retrieve_isp_knowledge",
            "get_next_best_question",
            "diagnose_connection_issue",
            "evaluate_escalation",
        },
        "requires_tool": True,
        "reason": "Memory-read requests must only use the profile lookup tool.",
    },
    "memory_write": {
        "allowed_tools": {"update_user_info"},
        "blocked_tools": {
            "get_user_info",
            "search_web",
            "retrieve_isp_knowledge",
            "get_next_best_question",
            "diagnose_connection_issue",
            "evaluate_escalation",
        },
        "requires_tool": True,
        "reason": "Memory-write requests must only use the profile update tool.",
    },
    "factual_lookup": {
        "allowed_tools": {"retrieve_isp_knowledge", "search_web"},
        "blocked_tools": {
            "get_user_info",
            "update_user_info",
            "get_next_best_question",
            "diagnose_connection_issue",
            "evaluate_escalation",
        },
        "requires_tool": True,
        "reason": "ISP factual questions must use knowledge tools, not profile-memory tools.",
    },
    "troubleshooting": {
        "allowed_tools": {
            "retrieve_isp_knowledge",
            "search_web",
            "get_next_best_question",
            "diagnose_connection_issue",
            "evaluate_escalation",
        },
        "blocked_tools": {"get_user_info", "update_user_info"},
        "requires_tool": False,
        "reason": "Troubleshooting flow should use troubleshooting/knowledge tools only.",
    },
    "general_chat": {
        "allowed_tools": set(),
        "blocked_tools": {
            "get_user_info",
            "update_user_info",
            "search_web",
            "retrieve_isp_knowledge",
            "get_next_best_question",
            "diagnose_connection_issue",
            "evaluate_escalation",
        },
        "requires_tool": False,
        "reason": "General chat should not call tools unless another intent is clearly detected.",
    },
}


class GraphConversationState(TypedDict, total=False):
    """Runtime graph state shape used by LangGraph."""

    messages: Annotated[list[BaseMessage], add_messages]
    known_state: dict[str, Any]
    emit_state_block: bool
    route: str
    session_id: str
    user_id: str
    tool_hints: list[str]
    raw_response: str
    assistant_text: str
    state_update: dict[str, Any]


@dataclass
class AgentTurnResult:
    """Result payload returned to routers after one assistant turn."""

    route: str
    raw_response: str
    assistant_text: str
    state_update: dict[str, Any]
    messages: list[dict[str, str]] = field(default_factory=list)
    known_state: dict[str, Any] = field(default_factory=dict)


def normalize_known_state(raw_state: dict[str, Any] | None) -> dict[str, Any]:
    """Return a known-state dict with all expected keys present."""
    normalized = {**DEFAULT_STATE}
    if not raw_state:
        return normalized

    for key in normalized:
        if key in raw_state:
            normalized[key] = raw_state[key]
    return normalized


def build_state_prompt(known_state: dict[str, Any], *, emit_state_block: bool = True) -> str:
    """Build the system prompt with current extracted session state."""
    state_str = json.dumps(known_state)
    if not emit_state_block:
        return (
            f"{DEFAULT_SYSTEM_PROMPT}\n\n"
            "/no_think\n\n"
            "Respond with 1-2 short sentences only.\n"
            "Do not include JSON, XML-like tags, or metadata blocks in your reply.\n\n"
            f"CURRENT KNOWN STATE (use for context):\n{state_str}"
        )

    return (
        f"{DEFAULT_SYSTEM_PROMPT}\n\n"
        "/no_think\n\n"
        "Tool selection/execution is handled in a separate planning phase before this response.\n"
        "Do NOT say you will check, look up, fetch, or call a tool later.\n"
        "If evidence is unavailable, say that clearly and ask one focused follow-up question.\n\n"
        "Respond first with your short reply (1-2 sentences max).\n"
        "Then append a newline and a <STATE> JSON block at the END "
        "of EVERY response.\n\n"
        "The STATE JSON has exactly these 5 keys:\n"
        '  router_model, lights_status, error_message, '
        'connection_type, has_restarted\n'
        'Use null for anything the user has NOT mentioned yet. '
        'Only fill a field when the user explicitly states it.\n\n'
        "=== FORMAT EXAMPLE (not real data, ignore these values) ===\n"
        'If a user said their BrandX router has green lights:\n'
        "I see your BrandX has green lights. What error are you getting?\n"
        '<STATE>{"router_model": "BrandX", "lights_status": "green", '
        '"error_message": null, "connection_type": null, '
        '"has_restarted": null}</STATE>\n'
        "=== END FORMAT EXAMPLE ===\n\n"
        f"CURRENT KNOWN STATE (carry forward and update):\n{state_str}"
    )


def parse_state_block(raw_response: str) -> tuple[dict[str, Any], str]:
    """Extract <STATE>{...}</STATE> from model output and return clean text."""
    if not raw_response:
        return {}, ""

    matches = list(re.finditer(r"<STATE>\s*(\{.*?\})\s*</STATE>", raw_response, re.DOTALL))
    if not matches:
        return {}, raw_response.strip()

    match = matches[-1]
    clean_text = (raw_response[:match.start()] + raw_response[match.end():]).strip()

    if not match.group(1):
        return {}, clean_text

    try:
        parsed = json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}, clean_text

    if not isinstance(parsed, dict):
        return {}, clean_text

    return parsed, clean_text


def extract_visible_response_text(raw_response: str, *, final: bool = False) -> str:
    """Return user-visible text while hiding a trailing <STATE> block."""
    if not raw_response:
        return ""

    state_open_tag = "<STATE>"
    state_start = raw_response.find(state_open_tag)
    if state_start != -1:
        return raw_response[:state_start].rstrip()

    if final:
        return raw_response

    # Guard against split tags across chunks, e.g. content ends with "<ST".
    max_partial = min(len(raw_response), len(state_open_tag) - 1)
    for partial_len in range(max_partial, 0, -1):
        if raw_response.endswith(state_open_tag[:partial_len]):
            return raw_response[:-partial_len]

    return raw_response


def merge_non_null_state(existing_state: dict[str, Any], update_state: dict[str, Any]) -> dict[str, Any]:
    """Merge non-empty state values into an existing state dictionary."""
    merged = dict(existing_state)
    for key, value in update_state.items():
        if key not in merged:
            continue
        if value is None:
            continue
        normalized = str(value).strip().lower()
        if normalized in {"", "null", "none"}:
            continue
        merged[key] = value
    return merged


class ConversationOrchestrator:
    """LangGraph orchestrator with a dialogue route ready for RAG/tool routing."""

    def __init__(self) -> None:
        self._checkpointer = InMemorySaver()
        base_llm = ChatOpenAI(
            model=LLAMA_MODEL_NAME,
            api_key=LLAMA_API_KEY,
            base_url=f"{LLAMA_SERVER_URL.rstrip('/')}/v1",
            temperature=LLAMA_TEMPERATURE,
            top_p=LLAMA_TOP_P,
            timeout=LLAMA_TIMEOUT_SEC,
            max_retries=0,
        )

        planner_timeout_sec = max(
            1.0,
            min(float(LLAMA_TIMEOUT_SEC), float(LLAMA_TOOL_PLANNER_TIMEOUT_SEC)),
        )
        planner_max_tokens = max(96, int(LLAMA_TOOL_PLANNER_MAX_TOKENS))
        planner_llm = ChatOpenAI(
            model=LLAMA_MODEL_NAME,
            api_key=LLAMA_API_KEY,
            base_url=f"{LLAMA_SERVER_URL.rstrip('/')}/v1",
            temperature=0.0,
            top_p=1.0,
            timeout=planner_timeout_sec,
            max_retries=0,
            max_tokens=planner_max_tokens,
        )

        self._llm = base_llm
        self._tool_llm = planner_llm.bind_tools(ALL_LANGCHAIN_TOOLS)
        self._planner_max_steps = max(1, int(LLAMA_TOOL_PLANNER_MAX_STEPS))
        self._planner_prompt = str(PLANNER_PROMPT).strip()
        self._tools_by_name = dict(ALL_TOOLS_BY_NAME)
        self._all_tool_names = set(self._tools_by_name.keys())
        self._tool_validation_policy = self._build_effective_tool_policy()
        self._graph = self._build_graph()

    def _build_effective_tool_policy(self) -> dict[IntentType, dict[str, Any]]:
        """Build policy so every registered tool is explicitly governed for each intent."""
        effective: dict[IntentType, dict[str, Any]] = {}
        for intent, base_policy in TOOL_VALIDATION_POLICY.items():
            allowed = set(base_policy.get("allowed_tools", set()))
            unknown_allowed = allowed - self._all_tool_names
            if unknown_allowed:
                logger.warning(
                    "tool_policy_unknown_allowed intent=%s tools=%s",
                    intent,
                    sorted(unknown_allowed),
                )
            normalized_allowed = allowed & self._all_tool_names
            normalized_blocked = set(base_policy.get("blocked_tools", set())) | (
                self._all_tool_names - normalized_allowed
            )
            effective[intent] = {
                **base_policy,
                "allowed_tools": normalized_allowed,
                "blocked_tools": normalized_blocked,
            }

        logger.info(
            "tool_policy_initialized intents=%s all_tools=%s",
            sorted(effective.keys()),
            sorted(self._all_tool_names),
        )
        return effective

    @staticmethod
    def _log_preview(text: str, *, limit: int = 800) -> str:
        """Normalize and cap text length for readable logs."""
        normalized = " ".join(str(text).split())
        if len(normalized) <= limit:
            return normalized
        return f"{normalized[: limit - 3]}..."

    @staticmethod
    def _extract_thinking_blocks(text: str) -> list[str]:
        """Extract model thinking blocks when template/runtime emits them."""
        patterns = [
            r"<think>\s*(.*?)\s*</think>",
            r"<\|begin_of_thought\|>\s*(.*?)\s*<\|end_of_thought\|>",
        ]

        blocks: list[str] = []
        for pattern in patterns:
            matches = re.findall(pattern, text, flags=re.DOTALL)
            if matches:
                blocks.extend(matches)

        return [block for block in blocks if str(block).strip()]

    @staticmethod
    def _extract_name_candidate(user_message: str) -> str | None:
        """Extract a likely name from self-identification phrases."""
        patterns = [
            r"\bmy name is\s+([A-Za-z][A-Za-z '\-]{0,40})",
            r"\bcall me\s+([A-Za-z][A-Za-z '\-]{0,40})",
            r"\bi am\s+([A-Za-z][A-Za-z '\-]{0,40})",
            r"\bi'm\s+([A-Za-z][A-Za-z '\-]{0,40})",
        ]
        for pattern in patterns:
            match = re.search(pattern, user_message, re.IGNORECASE)
            if not match:
                continue
            candidate = match.group(1).strip(" .,!?:;\"'")
            words = [word for word in candidate.split() if word]
            if words:
                return " ".join(words[:3])
        return None

    @staticmethod
    def _has_any_keyword(text: str, keywords: list[str]) -> bool:
        return any(keyword in text for keyword in keywords)

    @staticmethod
    def _should_force_lookup(user_message: str) -> bool:
        """Detect factual lookup requests that should not skip tools."""
        text = user_message.strip().lower()
        if not text:
            return False

        factual_keywords = [
            "price",
            "pricing",
            "cost",
            "package",
            "monthly",
            "plan",
            "pkr",
            "rupees",
            "brand",
            "model",
            "which router",
            "what router",
            "router brand",
            "latest",
            "current",
            "search the web",
            "search web",
            "look up",
            "find",
        ]
        if any(keyword in text for keyword in factual_keywords):
            return True

        wh_openers = ("what", "which", "who", "when", "where", "how much")
        return text.startswith(wh_openers) and "?" in text

    @staticmethod
    def _detect_intent(user_message: str, known_state: dict[str, Any]) -> IntentType:
        """Classify user turn into one planner intent bucket for tool policy checks."""
        text = user_message.strip().lower()
        if not text:
            return "general_chat"

        memory_write_keywords = [
            "my name is",
            "call me",
            "from now on",
            "remember my name",
            "save my name",
            "store my name",
            "remember this",
            "save my contact",
            "my number is",
            "my phone is",
            "my email is",
        ]
        if any(keyword in text for keyword in memory_write_keywords):
            return "memory_write"

        memory_read_keywords = [
            "what is my name",
            "who am i",
            "my profile",
            "my info",
            "what do you know about me",
            "do you remember me",
            "what is my contact",
            "what's my contact",
            "show my details",
        ]
        if any(keyword in text for keyword in memory_read_keywords):
            return "memory_read"

        troubleshooting_keywords = [
            "no internet",
            "no connection",
            "internet not working",
            "wifi not working",
            "router",
            "ont",
            "onu",
            "los",
            "blinking",
            "restart",
            "restarted",
            "disconnect",
            "latency",
            "packet loss",
            "slow internet",
            "troubleshoot",
            "not working",
        ]
        has_known_state_signal = any(
            value is not None and str(value).strip().lower() not in {"", "null", "none"}
            for value in (known_state or {}).values()
        )
        if any(keyword in text for keyword in troubleshooting_keywords) or (
            has_known_state_signal and any(keyword in text for keyword in ["still", "again", "issue", "problem", "help"])
        ):
            return "troubleshooting"

        factual_keywords = [
            "price",
            "pricing",
            "cost",
            "package",
            "plan",
            "helpline",
            "hotline",
            "support number",
            "contact number",
            "router brand",
            "which router",
            "what router",
            "latest",
            "current",
            "ptcl",
            "nayatel",
            "stormfiber",
            "transworld",
        ]
        if any(keyword in text for keyword in factual_keywords) or ConversationOrchestrator._should_force_lookup(text):
            return "factual_lookup"

        return "general_chat"

    def _intent_requires_tool(self, intent: IntentType) -> bool:
        policy = self._tool_validation_policy.get(intent, {})
        return bool(policy.get("requires_tool", False))

    def _validate_tool_call(
        self,
        *,
        intent: IntentType,
        tool_name: str | None,
        tool_args: dict[str, Any] | None,
        user_id: str,
    ) -> tuple[bool, str]:
        """Validate planner-selected tool against intent policy and argument rules."""
        policy = self._tool_validation_policy[intent]
        allowed_tools: set[str] = set(policy["allowed_tools"])
        blocked_tools: set[str] = set(policy["blocked_tools"])

        if tool_name is None:
            if self._intent_requires_tool(intent):
                return False, f"No-tool is invalid for intent '{intent}'. {policy['reason']}"
            return True, "No tool is valid for this intent."

        if tool_name in blocked_tools or tool_name not in allowed_tools:
            return (
                False,
                (
                    f"Tool '{tool_name}' is invalid for intent '{intent}'. "
                    f"Allowed tools: {sorted(allowed_tools)}. {policy['reason']}"
                ),
            )

        normalized_args = tool_args or {}

        if tool_name == "update_user_info":
            if not normalized_args:
                return False, "update_user_info requires arguments."
            field = str(normalized_args.get("field", "")).strip().lower()
            value = str(normalized_args.get("value", "")).strip()
            if not field:
                return False, "update_user_info requires a non-empty 'field'."
            if not value:
                return False, "update_user_info requires a non-empty 'value'."

        if tool_name == "get_user_info":
            requested_user = str(normalized_args.get("user_id") or user_id).strip()
            if not requested_user:
                return False, "get_user_info requires 'user_id'."

        if tool_name in {"search_web", "retrieve_isp_knowledge"}:
            query = str(normalized_args.get("query", "")).strip()
            if len(query) < 3:
                return False, f"{tool_name} requires a non-empty 'query'."

        if tool_name == "evaluate_escalation":
            if "known_state" not in normalized_args:
                return False, "evaluate_escalation requires 'known_state'."

        return True, "Tool call is valid."

    @staticmethod
    def _requires_fresh_web_lookup(user_message: str) -> bool:
        text = user_message.strip().lower()
        freshness_keywords = [
            "latest",
            "current",
            "today",
            "recent",
            "right now",
            "updated",
            "new",
        ]
        return any(keyword in text for keyword in freshness_keywords)

    def _required_tools_for_intent(self, intent: IntentType, user_message: str) -> set[str]:
        """Return tools that must execute successfully before final answer is generated."""
        if intent == "memory_read":
            return {"get_user_info"}
        if intent == "memory_write":
            return {"update_user_info"}
        if intent == "factual_lookup":
            required = {"retrieve_isp_knowledge"}
            if self._requires_fresh_web_lookup(user_message):
                required.add("search_web")
            return required
        return set()

    @staticmethod
    def _is_tool_output_success(tool_name: str, tool_output: str) -> bool:
        lowered = str(tool_output).strip().lower()
        failure_markers = (
            "tool policy blocked execution:",
            "tool execution failed:",
            "tool not available:",
        )
        if any(lowered.startswith(marker) for marker in failure_markers):
            return False

        if tool_name == "search_web":
            try:
                parsed = json.loads(str(tool_output))
            except Exception:  # noqa: BLE001
                return False
            if not isinstance(parsed, dict):
                return False
            results = parsed.get("results")
            if isinstance(results, list) and results:
                return True
            return int(parsed.get("count") or 0) > 0

        return True

    @staticmethod
    def _extract_regex_hits(text: str, pattern: str, *, max_items: int = 5) -> list[str]:
        """Return unique regex matches preserving first-seen order."""
        seen: set[str] = set()
        values: list[str] = []
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            value = match if isinstance(match, str) else " ".join(match)
            normalized = " ".join(str(value).split()).strip(" ,.;")
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            values.append(normalized)
            if len(values) >= max_items:
                break
        return values

    def _summarize_retrieval_context(self, *, context: str, query_text: str) -> str:
        """Extract compact factual nuggets from retrieved KB context."""
        compact_context = " ".join(context.split())
        lower_query = query_text.strip().lower()
        query_terms = [
            term
            for term in re.findall(r"[a-zA-Z]{4,}", lower_query)
            if term not in {"what", "which", "about", "their", "with", "that", "have", "from", "into", "pakistani"}
        ]

        sentence_candidates = re.split(r"[\n\r\.]+", context)
        relevant_sentences: list[str] = []
        for sentence in sentence_candidates:
            cleaned = " ".join(sentence.split()).strip()
            if len(cleaned) < 20:
                continue
            lower_cleaned = cleaned.lower()
            if query_terms and not any(term in lower_cleaned for term in query_terms):
                continue
            relevant_sentences.append(self._truncate_text(cleaned, max_chars=180))
            if len(relevant_sentences) >= 3:
                break

        if not relevant_sentences:
            # Fallback: keep a short leading context sample.
            relevant_sentences = [self._truncate_text(compact_context, max_chars=220)]

        phones = self._extract_regex_hits(
            compact_context,
            r"(?:\+?92\s?\d{2,4}[\s\-]?\d{6,8}|0\d{2,4}[\s\-]?\d{6,8}|\b1\d{3}\b)",
            max_items=5,
        )
        emails = self._extract_regex_hits(
            compact_context,
            r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}",
            max_items=5,
        )
        prices = self._extract_regex_hits(
            compact_context,
            r"(?:PKR|Rs\.?|Rupees?)\s?\d[\d,]*",
            max_items=6,
        )

        parts: list[str] = [
            f"evidence={ ' || '.join(relevant_sentences) }",
        ]
        if phones:
            parts.append(f"contact_phones={','.join(phones)}")
        if emails:
            parts.append(f"contact_emails={','.join(emails)}")
        if prices:
            parts.append(f"prices={','.join(prices)}")

        return self._truncate_text("; ".join(parts), max_chars=700)

    def _compact_tool_output(self, tool_name: str, output_text: str, *, query_text: str = "") -> str:
        """Compress raw tool payloads before injecting them into the final LLM prompt."""
        cleaned = str(output_text).strip()

        try:
            parsed = json.loads(cleaned)
        except Exception:  # noqa: BLE001
            return self._truncate_text(cleaned, max_chars=420)

        if tool_name == "search_web" and isinstance(parsed, dict):
            results = parsed.get("results") or []
            query_variants = parsed.get("query_variants") or []
            diagnostics = parsed.get("diagnostics") or []
            lines: list[str] = [
                f"query={parsed.get('query', '')}",
                f"count={len(results)}",
            ]
            if query_variants:
                lines.append(
                    "query_variants=" + ",".join(self._truncate_text(str(item), max_chars=60) for item in query_variants)
                )
            if not results:
                lines.append("no_results=true")

            provider_coverage = {"ptcl": False, "nayatel": False, "stormfiber": False, "transworld": False}
            combined_result_text: list[str] = []
            for idx, row in enumerate(results[:3], start=1):
                if not isinstance(row, dict):
                    continue
                title = self._truncate_text(str(row.get("title", "")), max_chars=100)
                url = self._truncate_text(str(row.get("url", "")), max_chars=120)
                snippet = self._truncate_text(str(row.get("snippet", "")), max_chars=180)
                source_query = self._truncate_text(str(row.get("source_query", "")), max_chars=70)
                line_text = f"{idx}) {title} | {url} | {snippet}"
                if source_query:
                    line_text = f"{line_text} | source_query={source_query}"
                lines.append(line_text)

                lower_blob = f"{title} {url} {snippet}".lower()
                for provider in provider_coverage:
                    if provider in lower_blob:
                        provider_coverage[provider] = True
                combined_result_text.append(f"{title} {snippet}")

            prices = self._extract_regex_hits(
                " ".join(combined_result_text),
                r"(?:PKR|Rs\.?|Rupees?)\s?\d[\d,]*",
                max_items=8,
            )
            lines.append(
                "provider_coverage="
                + ",".join(f"{name}:{'yes' if covered else 'no'}" for name, covered in provider_coverage.items())
            )
            if prices:
                lines.append("snippet_prices=" + ",".join(prices))
            if query_text and len(results) == 0:
                lines.append(f"user_query={self._truncate_text(query_text, max_chars=120)}")
            if diagnostics and len(results) == 0:
                diagnostic_bits: list[str] = []
                for item in diagnostics[:3]:
                    if not isinstance(item, dict):
                        continue
                    diagnostic_bits.append(
                        "{} raw={} parsed={}".format(
                            self._truncate_text(str(item.get("query", "")), max_chars=45),
                            item.get("raw_items", 0),
                            item.get("parsed_items", 0),
                        )
                    )
                if diagnostic_bits:
                    lines.append("diagnostics=" + " | ".join(diagnostic_bits))
            return self._truncate_text(" ; ".join(lines), max_chars=900)

        if tool_name == "retrieve_isp_knowledge" and isinstance(parsed, dict):
            similarity = parsed.get("best_similarity")
            context = str(parsed.get("context", ""))
            summary = self._summarize_retrieval_context(context=context, query_text=query_text)
            return f"best_similarity={similarity}; {summary}"

        if tool_name == "get_user_info" and isinstance(parsed, dict):
            name = parsed.get("name")
            contact = parsed.get("contact")
            pref_obj = parsed.get("preferences")
            prefs: dict[str, Any] = pref_obj if isinstance(pref_obj, dict) else {}
            pref_keys = ",".join(list(prefs.keys())[:5])
            return self._truncate_text(
                f"name={name}; contact={contact}; preference_keys={pref_keys or 'none'}",
                max_chars=280,
            )

        return self._truncate_text(cleaned, max_chars=420)

    def _prefetch_tool_calls(
        self,
        *,
        user_message: str,
        session_id: str,
        known_state: dict[str, Any],
        tool_hints: list[str],
    ) -> list[tuple[str, dict[str, Any], str]]:
        """Build deterministic prefetch tool plan from explicit hints and keywords."""
        # Keyword/hint prefetch is intentionally disabled.
        # Direct LLM planner-driven tool calling is used for every turn.
        _ = (user_message, session_id, known_state, tool_hints)
        return []

        _ = known_state
        text = user_message.strip().lower()
        normalized_hints = {hint.strip().lower() for hint in tool_hints if isinstance(hint, str)}

        plan: list[tuple[str, dict[str, Any], str]] = []
        seen: set[tuple[str, str]] = set()

        def add_call(tool_name: str, args: dict[str, Any], reason: str) -> None:
            key = (tool_name, json.dumps(args, sort_keys=True, default=str))
            if key in seen:
                return
            seen.add(key)
            plan.append((tool_name, args, reason))

        # Explicit user-selected frontend hints.
        if "profile" in normalized_hints:
            add_call("get_user_info", {"user_id": session_id}, "frontend_hint:profile")

        # CRM update for name memory.
        extracted_name = self._extract_name_candidate(user_message)
        if extracted_name and (
            "my name" in text or "call me" in text or "remember" in text or "store" in text or "save" in text
        ):
            add_call(
                "update_user_info",
                {"user_id": session_id, "field": "name", "value": extracted_name},
                "keyword:store_name",
            )

        # CRM retrieval for profile-memory asks.
        crm_fetch_keywords = [
            "what is my name",
            "who am i",
            "fetch my name",
            "my profile",
            "my info",
            "remember me",
            "what do you know about me",
            "from file",
        ]
        if any(keyword in text for keyword in crm_fetch_keywords):
            add_call("get_user_info", {"user_id": session_id}, "keyword:profile_lookup")

        # Determine information intent so we avoid shotgun-calling tools.
        web_intent_keywords = [
            "search the web",
            "web search",
            "search web",
            "look it up",
            "online",
            "latest",
            "today",
            "recent",
            "currently",
            "news",
            "right now",
            "now",
            "outage",
            "downdetector",
        ]
        troubleshooting_keywords = [
            "router",
            "modem",
            "nayatel",
            "wifi",
            "ethernet",
            "internet",
            "no internet",
            "no connection",
            "fiber",
            "ont",
            "onu",
            "los",
            "blinking",
            "overheat",
            "overheating",
            "disconnect",
            "latency",
            "packet",
            "speed",
            "error",
            "setup",
            "configure",
            "troubleshoot",
            "troubleshooting",
            "guide",
        ]

        prefers_web = "websearch" in normalized_hints or self._has_any_keyword(text, web_intent_keywords)
        prefers_rag = "rag" in normalized_hints or self._has_any_keyword(text, troubleshooting_keywords)

        # If user explicitly asks for web/latest, prioritize web and skip RAG keyword prefetch.
        if prefers_web:
            add_call("search_web", {"query": user_message, "max_results": 3}, "intent:web_or_freshness")
        elif prefers_rag:
            add_call("retrieve_isp_knowledge", {"query": user_message}, "intent:isp_troubleshooting")

        # Explicit hint can force RAG even when web intent exists.
        if "rag" in normalized_hints and not any(name == "retrieve_isp_knowledge" for name, _, _ in plan):
            add_call("retrieve_isp_knowledge", {"query": user_message}, "frontend_hint:rag")

        if "websearch" in normalized_hints and not any(name == "search_web" for name, _, _ in plan):
            add_call("search_web", {"query": user_message, "max_results": 3}, "frontend_hint:websearch")

        max_prefetch_calls = 3
        if len(plan) > max_prefetch_calls:
            plan = plan[:max_prefetch_calls]

        if plan:
            logger.info(
                "prefetch_plan session_id=%s tools=%s",
                session_id,
                [f"{name}:{reason}" for name, _, reason in plan],
            )
        return plan

    @staticmethod
    def _latest_user_message_text(state: GraphConversationState) -> str:
        messages = state.get("messages", [])
        for message in reversed(messages):
            if not isinstance(message, HumanMessage):
                continue

            content = message.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict):
                        maybe_text = item.get("text") or item.get("content")
                        if isinstance(maybe_text, str):
                            parts.append(maybe_text)
                return "".join(parts)
            return str(content)
        return ""

    @staticmethod
    def _tool_args_from_call(tool_args: Any) -> dict[str, Any]:
        if isinstance(tool_args, dict):
            return dict(tool_args)
        if isinstance(tool_args, str):
            return {"query": tool_args}
        return {"query": str(tool_args)}

    @staticmethod
    def _extract_json_block(text: str) -> str | None:
        """Return first balanced top-level JSON object found in text."""
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start:index + 1]
        return None

    def _extract_textual_tool_calls(self, decision_text: str) -> list[dict[str, Any]]:
        """Parse text-form tool calls like <tool_call>{...}</tool_call>."""
        if not decision_text:
            return []

        parsed_calls: list[dict[str, Any]] = []
        snippets = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", decision_text, flags=re.DOTALL)

        if not snippets and "<tool_call>" in decision_text:
            after_tag = decision_text.split("<tool_call>", 1)[1].strip()
            maybe_json = self._extract_json_block(after_tag)
            if maybe_json:
                snippets = [maybe_json]

        # Some models emit just raw JSON without explicit <tool_call> tags.
        if not snippets and '"name"' in decision_text and '"arguments"' in decision_text:
            maybe_json = self._extract_json_block(decision_text)
            if maybe_json:
                snippets = [maybe_json]

        for snippet in snippets:
            try:
                payload = json.loads(snippet)
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(payload, dict):
                continue
            name = payload.get("name")
            args = payload.get("arguments", payload.get("args", {}))
            if not isinstance(name, str) or not name.strip():
                continue
            parsed_calls.append({"name": name.strip(), "args": args})

        return parsed_calls

    async def _invoke_tool(
        self,
        *,
        tool_name: str,
        tool_args: Any,
        session_id: str,
        user_id: str,
        intent: IntentType,
        known_state: dict[str, Any],
    ) -> str:
        selected_tool = self._tools_by_name.get(tool_name)
        if selected_tool is None:
            logger.warning("tool_call_missing session_id=%s tool=%s", session_id, tool_name)
            return f"Tool not available: {tool_name}"

        call_args = self._tool_args_from_call(tool_args)
        resolved_user_id = str(user_id or session_id)

        # Auto-fill common tool arguments so models can use shorter calls.
        if tool_name in {"get_user_info", "update_user_info"}:
            call_args.setdefault("user_id", resolved_user_id)
        if tool_name in {
            "get_next_best_question",
            "diagnose_connection_issue",
            "evaluate_escalation",
        }:
            call_args.setdefault("known_state", known_state)
        if tool_name == "search_web" and intent == "factual_lookup":
            # Keep richer evidence for factual/provider queries.
            current_limit = call_args.get("max_results", 3)
            try:
                call_args["max_results"] = max(int(current_limit), 5)
            except (TypeError, ValueError):
                call_args["max_results"] = 5

        ok, reason = self._validate_tool_call(
            intent=intent,
            tool_name=tool_name,
            tool_args=call_args,
            user_id=resolved_user_id,
        )
        if not ok:
            logger.warning(
                "tool_call_policy_blocked session_id=%s intent=%s tool=%s reason=%s",
                session_id,
                intent,
                tool_name,
                self._log_preview(reason, limit=260),
            )
            return f"Tool policy blocked execution: {reason}"

        logger.info(
            "tool_call_start session_id=%s tool=%s args=%s",
            session_id,
            tool_name,
            self._log_preview(json.dumps(call_args, ensure_ascii=False, default=str)),
        )

        try:
            if inspect.iscoroutinefunction(selected_tool):
                output = await selected_tool(**call_args)
            else:
                output = selected_tool(**call_args)
        except TypeError:
            # Fallback for tools that only accept a single query argument.
            query = str(call_args.get("query", ""))
            if inspect.iscoroutinefunction(selected_tool):
                output = await selected_tool(query)
            else:
                output = selected_tool(query)
        except Exception as exc:  # noqa: BLE001
            logger.exception("tool_call_error session_id=%s tool=%s", session_id, tool_name)
            return f"Tool execution failed: {exc}"

        output_text = str(output)
        logger.info(
            "tool_call_result session_id=%s tool=%s output=%s",
            session_id,
            tool_name,
            self._log_preview(output_text),
        )
        return output_text

    @staticmethod
    def _thread_config(session_id: str) -> dict[str, dict[str, str]]:
        return {"configurable": {"thread_id": session_id}}

    @staticmethod
    def _extract_chunk_text(chunk: AIMessageChunk) -> str:
        """Normalize streamed chunk payloads to plain text."""
        content = chunk.content
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if isinstance(text, str):
                        parts.append(text)
            return "".join(parts)

        return ""

    @staticmethod
    def _extract_ai_message_text(message: AIMessage) -> str:
        """Normalize a non-streaming AI message payload to plain text."""
        content = message.content
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if isinstance(text, str):
                        parts.append(text)
            return "".join(parts)

        return str(content)

    @staticmethod
    def _extract_human_message_text(message: HumanMessage) -> str:
        """Normalize a non-streaming human message payload to plain text."""
        content = message.content
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if isinstance(text, str):
                        parts.append(text)
            return "".join(parts)

        return str(content)

    @staticmethod
    def _truncate_text(text: str, *, max_chars: int) -> str:
        """Cap text length to keep prompt size bounded on small context windows."""
        cleaned = text.strip()
        if len(cleaned) <= max_chars:
            return cleaned
        return f"{cleaned[: max_chars - 3].rstrip()}..."

    @staticmethod
    def _is_timeout_exception(exc: Exception) -> bool:
        """Return True when an exception chain indicates request timeout."""
        current: BaseException | None = exc
        while current is not None:
            name = current.__class__.__name__.lower()
            text = str(current).lower()
            if "timeout" in name or "timed out" in text or "readtimeout" in name:
                return True
            current = current.__cause__
        return False

    def _build_tool_decision_messages(
        self,
        state: GraphConversationState,
        known_state: dict[str, Any],
    ) -> list[BaseMessage]:
        """Build a compact tool-planning context to reduce prompt-token pressure."""
        compact_state = {
            key: value
            for key, value in known_state.items()
            if value is not None and str(value).strip().lower() not in {"", "null", "none"}
        }

        tool_hints = [
            str(item).strip().lower()
            for item in state.get("tool_hints", [])
            if str(item).strip()
        ]
        planner_prompt = (
            f"{self._planner_prompt}\n\n"
            "Tool catalog:\n"
            "- retrieve_isp_knowledge(query): FIRST choice for PTCL/Nayatel facts (helpline, package, router brand, setup, troubleshooting).\n"
            "- search_web(query, max_results): SECOND choice for current/time-sensitive info or when KB evidence is weak/empty.\n"
            "- get_user_info(user_id): only for personalization/memory (name/contact/preferences).\n"
            "- update_user_info(user_id, field, value): only when user explicitly shares/asks to save profile data.\n"
            "- get_next_best_question(known_state): troubleshooting workflow helper for next diagnostic question.\n"
            "- diagnose_connection_issue(known_state): troubleshooting workflow helper for likely root cause/next steps.\n"
            "- evaluate_escalation(known_state, failed_steps, minutes_without_service): troubleshooting workflow helper for escalation decision.\n\n"
            "Planning rules:\n"
            "1) For factual provider queries (price, package, helpline, router brand), call retrieve_isp_knowledge before final answer.\n"
            "2) If retrieve_isp_knowledge is low-confidence/insufficient or user asks for latest/current info, call search_web too.\n"
            "3) Use CRM tools only for user profile memory tasks, never for provider facts.\n"
            "4) Use workflow tools only for troubleshooting flows driven by known_state fields.\n"
            "5) You may call multiple tools over multiple rounds; if truly unnecessary, continue with no tool call.\n"
            "6) Keep arguments short and directly grounded in the latest user request.\n\n"
            f"Known state: {json.dumps(compact_state, ensure_ascii=False)}\n"
            f"Frontend tool hints: {json.dumps(tool_hints, ensure_ascii=False)}"
        )

        messages: list[BaseMessage] = [SystemMessage(content=planner_prompt)]

        recent: list[BaseMessage] = []
        for message in state.get("messages", []):
            if isinstance(message, HumanMessage):
                text = self._truncate_text(self._extract_human_message_text(message), max_chars=280)
                if text:
                    recent.append(HumanMessage(content=text))
            elif isinstance(message, AIMessage):
                text = self._truncate_text(self._extract_ai_message_text(message), max_chars=280)
                if text:
                    recent.append(AIMessage(content=text))

        if not recent:
            latest_user_text = self._truncate_text(self._latest_user_message_text(state), max_chars=280)
            if latest_user_text:
                recent.append(HumanMessage(content=latest_user_text))

        messages.extend(recent[-4:])
        return messages

    @staticmethod
    def _to_langchain_stored_messages(store_messages: list[dict[str, str]]) -> list[BaseMessage]:
        """Convert stored {role, content} messages into LangChain message objects."""
        converted: list[BaseMessage] = []
        for message in store_messages:
            role = message.get("role")
            content = str(message.get("content", ""))
            if role == "user":
                converted.append(HumanMessage(content=content))
            elif role == "assistant":
                converted.append(AIMessage(content=content))
        return converted

    def _to_langchain_messages(self, state: GraphConversationState) -> list[BaseMessage]:
        """Build the final LLM message list from persistent graph state."""
        known_state = normalize_known_state(state.get("known_state", {}))
        emit_state_block = bool(state.get("emit_state_block", True))
        messages: list[BaseMessage] = [
            SystemMessage(content=build_state_prompt(known_state, emit_state_block=emit_state_block))
        ]

        for example in FEW_SHOT_EXAMPLES:
            role = example.get("role")
            content = str(example.get("content", ""))
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

        conversation: list[BaseMessage] = []
        for message in state.get("messages", []):
            if isinstance(message, (HumanMessage, AIMessage)):
                conversation.append(message)

        messages.extend(conversation[-MAX_HISTORY:])
        return messages

    @staticmethod
    def _to_store_messages(messages: list[BaseMessage]) -> list[dict[str, str]]:
        """Serialize LangChain messages to store-friendly {role, content} dicts."""
        serialized: list[dict[str, str]] = []
        for message in messages:
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            else:
                continue

            content = message.content
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict):
                        maybe_text = item.get("text") or item.get("content")
                        if isinstance(maybe_text, str):
                            parts.append(maybe_text)
                text = "".join(parts)
            else:
                text = str(content)

            serialized.append({"role": role, "content": text})
        return serialized

    def _state_to_result(self, final_state: GraphConversationState) -> AgentTurnResult:
        known_state = normalize_known_state(final_state.get("known_state", {}))
        message_history = self._to_store_messages(final_state.get("messages", []))
        return AgentTurnResult(
            route=str(final_state.get("route", "dialogue")),
            raw_response=str(final_state.get("raw_response", "")),
            assistant_text=str(final_state.get("assistant_text", "")).strip(),
            state_update=dict(final_state.get("state_update", {})),
            messages=message_history,
            known_state=known_state,
        )

    def reset_memory(self) -> None:
        """Reset in-memory LangGraph checkpoint state (used by tests)."""
        self._checkpointer = InMemorySaver()
        self._graph = self._build_graph()

    async def ensure_thread_state(
        self,
        session_id: str,
        known_state: dict[str, Any],
        messages: list[dict[str, str]],
    ) -> None:
        """Hydrate a LangGraph thread from the compatibility in-memory session."""
        config = self._thread_config(session_id)

        try:
            snapshot = await self._graph.aget_state(config)
        except Exception:  # noqa: BLE001
            snapshot = None

        normalized_known_state = normalize_known_state(known_state)

        if snapshot and isinstance(snapshot.values, dict):
            values = snapshot.values
            if values.get("messages") or values.get("known_state"):
                if normalize_known_state(values.get("known_state", {})) != normalized_known_state:
                    await self._graph.aupdate_state(
                        config,
                        {"known_state": normalized_known_state},
                        as_node="parse_state",
                    )
                return

        hydrate_values: GraphConversationState = {
            "known_state": normalized_known_state,
        }

        stored_messages = self._to_langchain_stored_messages(messages)
        if stored_messages:
            hydrate_values["messages"] = stored_messages

        await self._graph.aupdate_state(config, hydrate_values, as_node="parse_state")

    async def get_session_snapshot(self, session_id: str) -> tuple[list[dict[str, str]], dict[str, Any]]:
        """Return current message history and known state for one thread."""
        config = self._thread_config(session_id)
        try:
            snapshot = await self._graph.aget_state(config)
        except Exception:  # noqa: BLE001
            return [], normalize_known_state({})

        if not snapshot or not isinstance(snapshot.values, dict):
            return [], normalize_known_state({})

        values = snapshot.values
        messages = self._to_store_messages(values.get("messages", []))
        known_state = normalize_known_state(values.get("known_state", {}))
        return messages, known_state

    async def delete_session_thread(self, session_id: str) -> None:
        """Delete one thread from the LangGraph checkpointer."""
        await self._checkpointer.adelete_thread(session_id)

    async def update_known_state(self, session_id: str, known_state: dict[str, Any]) -> None:
        """Persist the provided known state into a thread snapshot."""
        await self._graph.aupdate_state(
            self._thread_config(session_id),
            {"known_state": normalize_known_state(known_state)},
            as_node="parse_state",
        )

    async def extract_state_update(
        self,
        known_state: dict[str, Any],
        user_message: str,
        assistant_text: str,
    ) -> dict[str, Any]:
        """Extract state update asynchronously from one turn pair.

        This is used when streaming UX prioritizes low tail-latency and state
        extraction is decoupled from the visible assistant response path.
        """
        normalized = normalize_known_state(known_state)
        extractor_prompt = (
            "You extract a troubleshooting state JSON from one user turn and one assistant turn. "
            "Return ONLY valid JSON with exactly these keys: "
            "router_model, lights_status, error_message, connection_type, has_restarted. "
            "Use null when unknown. Do not include markdown or extra text."
        )
        payload = (
            f"CURRENT KNOWN STATE: {json.dumps(normalized)}\n"
            f"USER MESSAGE: {user_message}\n"
            f"ASSISTANT MESSAGE: {assistant_text}\n"
            "Return the updated JSON object now."
        )
        response = await self._llm.ainvoke(
            [
                SystemMessage(content=extractor_prompt),
                HumanMessage(content=payload),
            ]
        )

        text = self._extract_ai_message_text(cast(AIMessage, response)).strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}

        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}

        if not isinstance(parsed, dict):
            return {}

        return {
            key: parsed.get(key)
            for key in DEFAULT_STATE
        }

    def _build_graph(self):
        """Compile route -> dialogue -> parse graph."""

        async def route_node(state: GraphConversationState) -> GraphConversationState:
            # Placeholder route for future branches (rag/tools/plain).
            _ = state
            return {"route": "dialogue"}

        async def dialogue_node(state: GraphConversationState) -> GraphConversationState:
            writer = get_stream_writer()
            emit_state_block = bool(state.get("emit_state_block", True))
            session_id = str(state.get("session_id", ""))
            user_id = str(state.get("user_id") or session_id)
            known_state = normalize_known_state(state.get("known_state", {}))
            hidden_markers = ["<STATE>", "</STATE>"]
            stop_sequences = None if emit_state_block else hidden_markers
            state_open_tag = "<STATE>"
            logger.info(
                "turn_start session_id=%s emit_state_block=%s known_state=%s",
                session_id,
                emit_state_block,
                self._log_preview(json.dumps(known_state, ensure_ascii=False, default=str)),
            )

            def _emit_status(message: str, *, phase: str, **extra: Any) -> None:
                if writer is None:
                    return
                payload: dict[str, Any] = {
                    "type": "status",
                    "phase": phase,
                    "message": message,
                }
                for key, value in extra.items():
                    if value is not None:
                        payload[key] = value
                writer(payload)

            async def _stream_final_response(messages: list[BaseMessage]) -> str:
                raw_response = ""
                last_visible_text = ""
                visible_done_sent = False

                async for chunk in self._llm.astream(messages, stop=stop_sequences):
                    text = self._extract_chunk_text(chunk)
                    if not text:
                        continue
                    raw_response += text

                    if not emit_state_block:
                        cut_indices = [raw_response.find(marker) for marker in hidden_markers if marker in raw_response]
                        first_cut = min(cut_indices) if cut_indices else -1
                        if first_cut != -1:
                            raw_response = raw_response[:first_cut].rstrip()
                            if writer is not None and len(raw_response) > len(last_visible_text):
                                delta = raw_response[len(last_visible_text):]
                                if delta:
                                    writer({"type": "token", "token": delta})
                                last_visible_text = raw_response
                            break

                    if writer is not None:
                        visible_text = extract_visible_response_text(raw_response, final=False)
                        if len(visible_text) > len(last_visible_text):
                            delta = visible_text[len(last_visible_text):]
                            if delta:
                                writer({"type": "token", "token": delta})
                            last_visible_text = visible_text

                        if not visible_done_sent and state_open_tag in raw_response:
                            writer({"type": "visible_done"})
                            visible_done_sent = True

                if writer is not None and not visible_done_sent:
                    writer({"type": "visible_done"})

                thinking_matches = self._extract_thinking_blocks(raw_response)
                if thinking_matches:
                    for idx, block in enumerate(thinking_matches, start=1):
                        logger.info(
                            "model_thinking session_id=%s block=%s content=%s",
                            session_id,
                            idx,
                            self._log_preview(block),
                        )

                logger.info(
                    "model_response_raw session_id=%s content=%s",
                    session_id,
                    self._log_preview(raw_response),
                )

                return raw_response

            llm_messages = self._to_langchain_messages(state)

            # NOTE: Keyword/hint-based prefetch logic is intentionally disabled.
            # The planner model now decides all tool usage directly.
            # prefetch_plan = self._prefetch_tool_calls(...)

            tool_context_messages = self._build_tool_decision_messages(state, known_state)
            tool_round_messages: list[BaseMessage] = []
            fallback_tool_messages: list[BaseMessage] = []
            final_tool_evidence_messages: list[BaseMessage] = []
            planner_completed = False
            planner_timed_out = False
            latest_user_query = self._latest_user_message_text(state)
            detected_intent = self._detect_intent(latest_user_query, known_state)
            policy = self._tool_validation_policy[detected_intent]
            allowed_for_intent = sorted(policy["allowed_tools"])
            required_tools = self._required_tools_for_intent(detected_intent, latest_user_query)
            executed_valid_tools: set[str] = set()

            tool_context_messages.append(
                SystemMessage(
                    content=(
                        f"Detected intent: {detected_intent}. "
                        f"Strict tool policy: only {allowed_for_intent or ['no_tool']}. "
                        "If your last tool choice is invalid, self-correct and emit a valid tool call or no_tool."
                    )
                )
            )

            _emit_status("Analyzing your request and selecting tools...", phase="planning")
            _emit_status(
                f"Detected intent: {detected_intent.replace('_', ' ')}",
                phase="planning",
            )

            for step_index in range(self._planner_max_steps):
                _emit_status(
                    f"Planning step {step_index + 1}/{self._planner_max_steps}...",
                    phase="planning",
                )
                try:
                    decision = await self._tool_llm.ainvoke(tool_context_messages)
                except Exception as exc:  # noqa: BLE001
                    # Fall back to plain response generation when tool planning overflows
                    # or the provider rejects tool schemas.
                    if self._is_timeout_exception(exc):
                        planner_timed_out = True
                        logger.warning(
                            "tool_planner_timeout session_id=%s timeout_sec=%s error=%s",
                            session_id,
                            LLAMA_TOOL_PLANNER_TIMEOUT_SEC,
                            self._log_preview(str(exc), limit=220),
                        )
                        _emit_status(
                            "Planning timed out; continuing with the best available context.",
                            phase="planning",
                        )
                    else:
                        logger.exception("tool_planner_error session_id=%s", session_id)
                        _emit_status(
                            "Planner hit an internal issue; continuing with direct response generation.",
                            phase="planning",
                        )
                    break

                assistant_decision = cast(AIMessage, decision)
                decision_text = self._extract_ai_message_text(assistant_decision)
                preview = self._truncate_text(decision_text, max_chars=160)
                if preview:
                    _emit_status(f"Planner decision: {preview}", phase="planning")

                tool_calls = assistant_decision.tool_calls or []
                textual_tool_calls = self._extract_textual_tool_calls(decision_text)
                logger.info(
                    "tool_planner_decision session_id=%s step=%s tool_calls=%s textual_tool_calls=%s content=%s",
                    session_id,
                    step_index + 1,
                    len(tool_calls),
                    len(textual_tool_calls),
                    self._log_preview(decision_text),
                )
                if not tool_calls and textual_tool_calls:
                    tool_calls = textual_tool_calls

                if not tool_calls:
                    missing_required_tools = sorted(required_tools - executed_valid_tools)
                    if missing_required_tools:
                        reason = (
                            f"No-tool is invalid because required tools are still missing: {missing_required_tools}. "
                            f"Intent '{detected_intent}' must satisfy required tool policy first."
                        )
                        logger.warning(
                            "tool_policy_retry session_id=%s step=%s reason=%s",
                            session_id,
                            step_index + 1,
                            self._log_preview(reason, limit=260),
                        )
                        _emit_status(
                            "Planner skipped a required tool; retrying with policy feedback.",
                            phase="planning",
                        )
                        tool_context_messages.append(assistant_decision)
                        tool_context_messages.append(
                            SystemMessage(
                                content=(
                                    "Invalid no-tool choice. "
                                    f"{reason} Choose only a valid tool for intent '{detected_intent}'. "
                                    f"Allowed tools: {allowed_for_intent}."
                                )
                            )
                        )
                        continue

                    planner_completed = True
                    _emit_status(
                        "Planner selected no additional tool calls.",
                        phase="planning",
                    )
                    break

                validated_tool_calls: list[dict[str, Any]] = []
                validation_errors: list[str] = []
                for call_index, tool_call in enumerate(tool_calls, start=1):
                    candidate_tool_name = str(tool_call.get("name", ""))
                    candidate_args = self._tool_args_from_call(tool_call.get("args", {}))
                    ok, reason = self._validate_tool_call(
                        intent=detected_intent,
                        tool_name=candidate_tool_name,
                        tool_args=candidate_args,
                        user_id=user_id,
                    )
                    if not ok:
                        validation_errors.append(f"#{call_index} {reason}")
                        continue

                    normalized_call = dict(tool_call)
                    normalized_call["args"] = candidate_args
                    validated_tool_calls.append(normalized_call)

                if validation_errors:
                    policy_feedback = " | ".join(validation_errors)
                    logger.warning(
                        "tool_policy_retry session_id=%s step=%s feedback=%s",
                        session_id,
                        step_index + 1,
                        self._log_preview(policy_feedback, limit=320),
                    )
                    _emit_status(
                        "Planner selected an invalid tool; retrying with policy feedback.",
                        phase="planning",
                    )
                    tool_context_messages.append(assistant_decision)
                    tool_context_messages.append(
                        SystemMessage(
                            content=(
                                "Invalid tool choice. "
                                f"{policy_feedback} Choose only a valid tool for intent '{detected_intent}'. "
                                f"Allowed tools: {allowed_for_intent}."
                            )
                        )
                    )
                    continue

                tool_context_messages.append(assistant_decision)
                tool_round_messages.append(assistant_decision)

                for call_index, tool_call in enumerate(validated_tool_calls, start=1):
                    tool_name = str(tool_call.get("name", ""))
                    tool_id = str(tool_call.get("id", f"planner-text-{step_index + 1}-{call_index}"))
                    tool_args = tool_call.get("args", {})
                    query_text = str(self._tool_args_from_call(tool_args).get("query", ""))
                    tool_args_preview = self._log_preview(
                        json.dumps(self._tool_args_from_call(tool_args), ensure_ascii=False, default=str),
                        limit=180,
                    )
                    _emit_status(
                        f"Calling tool: {tool_name}",
                        phase="tool",
                        tool_name=tool_name,
                        tool_args=tool_args_preview,
                    )

                    tool_output = await self._invoke_tool(
                        tool_name=tool_name,
                        tool_args=tool_args,
                        session_id=session_id,
                        user_id=user_id,
                        intent=detected_intent,
                        known_state=known_state,
                    )
                    if self._is_tool_output_success(tool_name, tool_output):
                        executed_valid_tools.add(tool_name)

                    compact_output = self._compact_tool_output(tool_name, tool_output, query_text=query_text)
                    final_tool_evidence_messages.append(
                        SystemMessage(
                            content=(
                                f"TOOL_RESULT name={tool_name} "
                                f"payload={self._truncate_text(compact_output, max_chars=520)}"
                            )
                        )
                    )
                    _emit_status(
                        f"Completed tool: {tool_name}",
                        phase="tool",
                        tool_name=tool_name,
                        detail=self._truncate_text(compact_output, max_chars=220),
                    )

                    tool_message = ToolMessage(
                        content=self._truncate_text(str(tool_output), max_chars=600),
                        tool_call_id=tool_id,
                        name=tool_name or None,
                    )
                    tool_context_messages.append(tool_message)
                    tool_round_messages.append(tool_message)

            if planner_timed_out and not tool_round_messages:
                fallback_calls: list[tuple[str, dict[str, Any]]] = []
                if detected_intent == "memory_read":
                    fallback_calls = [("get_user_info", {})]
                elif detected_intent == "memory_write":
                    extracted_name = self._extract_name_candidate(latest_user_query)
                    if extracted_name:
                        fallback_calls = [
                            (
                                "update_user_info",
                                {"field": "name", "value": extracted_name},
                            )
                        ]
                elif detected_intent in {"factual_lookup", "troubleshooting"}:
                    fallback_calls = [
                        ("retrieve_isp_knowledge", {"query": latest_user_query}),
                        ("search_web", {"query": latest_user_query, "max_results": 3}),
                    ]

                if fallback_calls:
                    _emit_status(
                        "Planner timed out. Running policy-aware fallback tool calls...",
                        phase="planning",
                    )
                else:
                    _emit_status(
                        "Planner timed out and no fallback tool is required for this intent.",
                        phase="planning",
                    )

                for tool_name, tool_args in fallback_calls:
                    query_text = str(tool_args.get("query", ""))
                    _emit_status(
                        f"Fallback tool call: {tool_name}",
                        phase="tool",
                        tool_name=tool_name,
                        tool_args=self._log_preview(
                            json.dumps(tool_args, ensure_ascii=False, default=str),
                            limit=180,
                        ),
                    )
                    tool_output = await self._invoke_tool(
                        tool_name=tool_name,
                        tool_args=tool_args,
                        session_id=session_id,
                        user_id=user_id,
                        intent=detected_intent,
                        known_state=known_state,
                    )
                    if self._is_tool_output_success(tool_name, tool_output):
                        executed_valid_tools.add(tool_name)
                    compact_output = self._compact_tool_output(tool_name, tool_output, query_text=query_text)
                    fallback_tool_messages.append(
                        SystemMessage(
                            content=(
                                f"FALLBACK_TOOL_RESULT name={tool_name} "
                                f"payload={self._truncate_text(compact_output, max_chars=520)}"
                            )
                        )
                    )
                    final_tool_evidence_messages.append(
                        SystemMessage(
                            content=(
                                f"TOOL_RESULT name={tool_name} "
                                f"payload={self._truncate_text(compact_output, max_chars=520)}"
                            )
                        )
                    )
                    _emit_status(
                        f"Fallback tool completed: {tool_name}",
                        phase="tool",
                        tool_name=tool_name,
                        detail=self._truncate_text(compact_output, max_chars=220),
                    )

            if (
                planner_completed
                and not tool_round_messages
                and not fallback_tool_messages
                and detected_intent == "factual_lookup"
                and self._should_force_lookup(latest_user_query)
            ):
                _emit_status(
                    "Planner skipped tools on a factual lookup request. Running a fallback search pass...",
                    phase="planning",
                )
                forced_calls = [
                    ("search_web", {"query": latest_user_query, "max_results": 3}),
                    ("retrieve_isp_knowledge", {"query": latest_user_query}),
                ]
                for tool_name, tool_args in forced_calls:
                    query_text = str(tool_args.get("query", ""))
                    _emit_status(
                        f"Fallback tool call: {tool_name}",
                        phase="tool",
                        tool_name=tool_name,
                        tool_args=self._log_preview(
                            json.dumps(tool_args, ensure_ascii=False, default=str),
                            limit=180,
                        ),
                    )
                    tool_output = await self._invoke_tool(
                        tool_name=tool_name,
                        tool_args=tool_args,
                        session_id=session_id,
                        user_id=user_id,
                        intent=detected_intent,
                        known_state=known_state,
                    )
                    if self._is_tool_output_success(tool_name, tool_output):
                        executed_valid_tools.add(tool_name)
                    compact_output = self._compact_tool_output(tool_name, tool_output, query_text=query_text)
                    fallback_tool_messages.append(
                        SystemMessage(
                            content=(
                                f"FALLBACK_TOOL_RESULT name={tool_name} "
                                f"payload={self._truncate_text(compact_output, max_chars=520)}"
                            )
                        )
                    )
                    final_tool_evidence_messages.append(
                        SystemMessage(
                            content=(
                                f"TOOL_RESULT name={tool_name} "
                                f"payload={self._truncate_text(compact_output, max_chars=520)}"
                            )
                        )
                    )
                    _emit_status(
                        f"Fallback tool completed: {tool_name}",
                        phase="tool",
                        tool_name=tool_name,
                        detail=self._truncate_text(compact_output, max_chars=220),
                    )

            missing_required_tools = sorted(required_tools - executed_valid_tools)
            if missing_required_tools:
                _emit_status(
                    "Policy requires additional tool calls before final response. Enforcing now...",
                    phase="planning",
                )
                for tool_name in missing_required_tools:
                    forced_args: dict[str, Any]
                    if tool_name == "retrieve_isp_knowledge":
                        forced_args = {"query": latest_user_query}
                    elif tool_name == "search_web":
                        forced_args = {"query": latest_user_query, "max_results": 3}
                    elif tool_name == "get_user_info":
                        forced_args = {}
                    elif tool_name == "update_user_info":
                        extracted_name = self._extract_name_candidate(latest_user_query)
                        forced_args = {
                            "field": "name" if extracted_name else "note",
                            "value": extracted_name or latest_user_query,
                        }
                    else:
                        forced_args = {}

                    _emit_status(
                        f"Policy-enforced tool call: {tool_name}",
                        phase="tool",
                        tool_name=tool_name,
                        tool_args=self._log_preview(
                            json.dumps(forced_args, ensure_ascii=False, default=str),
                            limit=180,
                        ),
                    )

                    tool_output = await self._invoke_tool(
                        tool_name=tool_name,
                        tool_args=forced_args,
                        session_id=session_id,
                        user_id=user_id,
                        intent=detected_intent,
                        known_state=known_state,
                    )
                    if self._is_tool_output_success(tool_name, tool_output):
                        executed_valid_tools.add(tool_name)

                    query_text = str(forced_args.get("query", ""))
                    compact_output = self._compact_tool_output(tool_name, tool_output, query_text=query_text)
                    fallback_tool_messages.append(
                        SystemMessage(
                            content=(
                                f"FALLBACK_TOOL_RESULT name={tool_name} "
                                f"payload={self._truncate_text(compact_output, max_chars=520)}"
                            )
                        )
                    )
                    final_tool_evidence_messages.append(
                        SystemMessage(
                            content=(
                                f"TOOL_RESULT name={tool_name} "
                                f"payload={self._truncate_text(compact_output, max_chars=520)}"
                            )
                        )
                    )
                    _emit_status(
                        f"Policy-enforced tool completed: {tool_name}",
                        phase="tool",
                        tool_name=tool_name,
                        detail=self._truncate_text(compact_output, max_chars=220),
                    )

            missing_required_tools = sorted(required_tools - executed_valid_tools)
            if missing_required_tools:
                _emit_status(
                    "Required tool policy could not be fulfilled; skipping final LLM response.",
                    phase="response",
                )
                logger.error(
                    "tool_policy_unfulfilled session_id=%s intent=%s missing=%s",
                    session_id,
                    detected_intent,
                    missing_required_tools,
                )
                raw_response = (
                    "I could not complete the required tool step for this request, so I will not provide a final answer. "
                    "Please retry this question."
                )
                if writer is not None:
                    writer({"type": "token", "token": raw_response})
                    writer({"type": "visible_done"})
                return {"raw_response": raw_response}

            if not planner_completed:
                _emit_status(
                    "Reached planner step limit; moving to final answer.",
                    phase="planning",
                )

            response_messages = list(llm_messages)
            response_messages.append(
                SystemMessage(
                    content=(
                        "All tool planning/execution for this turn is complete before drafting this answer. "
                        "Do not use future-tense promises like 'let me check' or 'I will look it up'. "
                        "Never claim you do not have web/tool access. "
                        "Answer directly using available tool evidence; if evidence is missing, state uncertainty briefly. "
                        "If TOOL_RESULT or FALLBACK_TOOL_RESULT includes explicit facts (phone, email, package price, router model), "
                        "quote at least one exact fact in the answer and attribute it as retrieved context."
                    )
                )
            )
            if detected_intent == "factual_lookup":
                response_messages.append(
                    SystemMessage(
                        content=(
                            "Factual lookup response rules: "
                            "1) Use TOOL_RESULT evidence first. "
                            "2) If the user asked about multiple providers (for example PTCL and Nayatel), include a separate line for each provider. "
                            "3) Do not default to 'check official website' if search results already include relevant URLs/snippets. "
                            "4) If exact PKR values are missing in evidence, state that clearly and provide the best retrieved links/snippets."
                        )
                    )
                )
            response_messages.extend(final_tool_evidence_messages)
            response_messages.extend(fallback_tool_messages)
            response_messages.extend(tool_round_messages)
            _emit_status("Composing final response...", phase="response")

            raw_response = await _stream_final_response(response_messages)
            return {"raw_response": raw_response}

        async def parse_state_node(state: GraphConversationState) -> GraphConversationState:
            session_id = str(state.get("session_id", ""))
            raw_response = str(state.get("raw_response", ""))
            parsed_state, assistant_text = parse_state_block(raw_response)
            logger.info(
                "state_parse_input session_id=%s raw_response=%s",
                session_id,
                self._log_preview(raw_response),
            )
            logger.info(
                "state_parse_result session_id=%s parsed_state=%s",
                session_id,
                self._log_preview(json.dumps(parsed_state, ensure_ascii=False, default=str)),
            )
            merged_state = merge_non_null_state(
                normalize_known_state(state.get("known_state", {})),
                parsed_state,
            )
            logger.info(
                "state_merged session_id=%s merged_state=%s",
                session_id,
                self._log_preview(json.dumps(merged_state, ensure_ascii=False, default=str)),
            )
            logger.info(
                "assistant_text session_id=%s text=%s",
                session_id,
                self._log_preview(assistant_text),
            )
            updates: GraphConversationState = {
                "assistant_text": assistant_text,
                "state_update": parsed_state,
                "known_state": merged_state,
            }
            if assistant_text:
                updates["messages"] = [AIMessage(content=assistant_text)]
            return updates

        graph = StateGraph(GraphConversationState)
        graph.add_node("route", route_node)
        graph.add_node("dialogue", dialogue_node)
        graph.add_node("parse_state", parse_state_node)

        graph.add_edge(START, "route")
        graph.add_conditional_edges(
            "route",
            lambda s: s.get("route", "dialogue"),
            {
                "dialogue": "dialogue",
            },
        )
        graph.add_edge("dialogue", "parse_state")
        graph.add_edge("parse_state", END)

        return graph.compile(checkpointer=self._checkpointer)

    async def stream_turn_events(
        self,
        session_id: str,
        user_message: str,
        user_id: str | None = None,
        emit_state_block: bool = True,
        tool_hints: list[str] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Run one graph turn and yield streaming token + final result events."""
        input_state: GraphConversationState = {
            "messages": [HumanMessage(content=user_message)],
            "emit_state_block": emit_state_block,
            "session_id": session_id,
            "user_id": str(user_id or session_id),
            "tool_hints": list(tool_hints or []),
        }
        config = self._thread_config(session_id)

        final_state: GraphConversationState | None = None

        async for mode, chunk in self._graph.astream(
            input_state,
            config=config,
            stream_mode=["custom", "values"],
        ):
            if mode == "custom" and isinstance(chunk, dict):
                event_type = chunk.get("type")
                if event_type == "token":
                    token = str(chunk.get("token", ""))
                    if token:
                        yield {"type": "token", "token": token}
                elif event_type == "visible_done":
                    yield {"type": "visible_done"}
                elif event_type == "status":
                    payload = {
                        "type": "status",
                        "phase": str(chunk.get("phase", "planning")),
                        "message": str(chunk.get("message", "")),
                    }
                    if "tool_name" in chunk:
                        payload["tool_name"] = str(chunk.get("tool_name", ""))
                    if "tool_args" in chunk:
                        payload["tool_args"] = str(chunk.get("tool_args", ""))
                    if "detail" in chunk:
                        payload["detail"] = str(chunk.get("detail", ""))
                    yield payload
            elif mode == "values" and isinstance(chunk, dict):
                final_state = cast(GraphConversationState, chunk)

        if not final_state:
            raise RuntimeError("Conversation graph ended without a final state.")

        result = self._state_to_result(final_state)
        yield {"type": "final", "result": result}

    async def run_turn(
        self,
        session_id: str,
        user_message: str,
        user_id: str | None = None,
        emit_state_block: bool = True,
        tool_hints: list[str] | None = None,
    ) -> AgentTurnResult:
        """Run one graph turn without streaming, useful for voice endpoint."""
        input_state: GraphConversationState = {
            "messages": [HumanMessage(content=user_message)],
            "emit_state_block": emit_state_block,
            "session_id": session_id,
            "user_id": str(user_id or session_id),
            "tool_hints": list(tool_hints or []),
        }
        result_state = await self._graph.ainvoke(input_state, config=self._thread_config(session_id))
        return self._state_to_result(cast(GraphConversationState, result_state))


conversation_orchestrator = ConversationOrchestrator()

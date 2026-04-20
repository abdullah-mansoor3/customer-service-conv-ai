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
from typing import Annotated, Any, AsyncIterator, TypedDict, cast

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
    LLAMA_TOOL_PLANNER_TIMEOUT_SEC,
    LLAMA_TIMEOUT_SEC,
    LLAMA_TOP_P,
    MAX_HISTORY,
)
from app.orchestration.tools import ALL_LANGCHAIN_TOOLS, ALL_TOOLS_BY_NAME
from app.store import DEFAULT_STATE

logger = logging.getLogger(__name__)


class GraphConversationState(TypedDict, total=False):
    """Runtime graph state shape used by LangGraph."""

    messages: Annotated[list[BaseMessage], add_messages]
    known_state: dict[str, Any]
    emit_state_block: bool
    route: str
    session_id: str
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
        "You may call tools when useful:\n"
        "- retrieve_isp_knowledge: ISP troubleshooting/device/setup knowledge from local KB.\n"
        "- search_web: up-to-date web info for time-sensitive facts.\n"
        "- get_user_info / update_user_info: read or update CRM profile fields.\n"
        "- get_next_best_question / diagnose_connection_issue / evaluate_escalation: "
        "support workflow helpers.\n"
        "Do not call tools for greetings, chit-chat, or purely personal opinions.\n\n"
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

        # Keep tool-planning responses short to avoid long thinking loops
        # and planner-side timeouts on CPU inference.
        planner_timeout_sec = max(
            1.0,
            min(float(LLAMA_TIMEOUT_SEC), float(LLAMA_TOOL_PLANNER_TIMEOUT_SEC)),
        )
        planner_llm = ChatOpenAI(
            model=LLAMA_MODEL_NAME,
            api_key=LLAMA_API_KEY,
            base_url=f"{LLAMA_SERVER_URL.rstrip('/')}/v1",
            temperature=0.0,
            top_p=1.0,
            timeout=planner_timeout_sec,
            max_retries=0,
            max_tokens=96,
        )

        self._llm = base_llm
        self._tool_llm = planner_llm.bind_tools(ALL_LANGCHAIN_TOOLS)
        self._tools_by_name = dict(ALL_TOOLS_BY_NAME)
        self._graph = self._build_graph()

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

    def _compact_tool_output(self, tool_name: str, output_text: str) -> str:
        """Compress raw tool payloads before injecting them into the final LLM prompt."""
        cleaned = str(output_text).strip()

        try:
            parsed = json.loads(cleaned)
        except Exception:  # noqa: BLE001
            return self._truncate_text(cleaned, max_chars=420)

        if tool_name == "search_web" and isinstance(parsed, dict):
            results = parsed.get("results") or []
            lines: list[str] = [
                f"query={parsed.get('query', '')}",
                f"count={len(results)}",
            ]
            if not results:
                lines.append("no_results=true")
            for idx, row in enumerate(results[:3], start=1):
                if not isinstance(row, dict):
                    continue
                title = self._truncate_text(str(row.get("title", "")), max_chars=100)
                url = self._truncate_text(str(row.get("url", "")), max_chars=120)
                snippet = self._truncate_text(str(row.get("snippet", "")), max_chars=140)
                lines.append(f"{idx}) {title} | {url} | {snippet}")
            return self._truncate_text(" ; ".join(lines), max_chars=520)

        if tool_name == "retrieve_isp_knowledge" and isinstance(parsed, dict):
            similarity = parsed.get("best_similarity")
            context = self._truncate_text(str(parsed.get("context", "")), max_chars=360)
            return f"best_similarity={similarity}; context={context}"

        if tool_name == "get_user_info" and isinstance(parsed, dict):
            name = parsed.get("name")
            contact = parsed.get("contact")
            prefs = parsed.get("preferences") if isinstance(parsed.get("preferences"), dict) else {}
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

    async def _invoke_tool(
        self,
        *,
        tool_name: str,
        tool_args: Any,
        session_id: str,
        known_state: dict[str, Any],
    ) -> str:
        selected_tool = self._tools_by_name.get(tool_name)
        if selected_tool is None:
            logger.warning("tool_call_missing session_id=%s tool=%s", session_id, tool_name)
            return f"Tool not available: {tool_name}"

        call_args = self._tool_args_from_call(tool_args)

        # Auto-fill common tool arguments so models can use shorter calls.
        if tool_name in {"get_user_info", "update_user_info"}:
            call_args.setdefault("user_id", session_id)
        if tool_name in {
            "get_next_best_question",
            "diagnose_connection_issue",
            "evaluate_escalation",
        }:
            call_args.setdefault("known_state", known_state)

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

        planner_prompt = (
            "You are a tool planner for ISP support. "
            "Call tools only when they add needed factual data or save user profile info. "
            "Use the minimum number of tool calls, with short arguments. "
            "If no tool is needed, reply normally without any tool call.\n"
            "Available tools and intended usage:\n"
            "- retrieve_isp_knowledge(query): use for ISP/device/factual troubleshooting knowledge from local KB.\n"
            "- search_web(query, max_results): use only for time-sensitive or current public info.\n"
            "- get_user_info(user_id): read saved CRM profile fields.\n"
            "- update_user_info(user_id, field, value): persist user profile data like name/contact.\n"
            "- get_next_best_question(known_state): ask the best next troubleshooting question.\n"
            "- diagnose_connection_issue(known_state): infer likely root cause and next steps.\n"
            "- evaluate_escalation(known_state, failed_steps, minutes_without_service): escalation decision.\n"
            f"Known state: {json.dumps(compact_state, ensure_ascii=False)}"
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
            known_state = normalize_known_state(state.get("known_state", {}))
            tool_hints = [
                str(item).strip().lower()
                for item in state.get("tool_hints", [])
                if str(item).strip()
            ]
            hidden_markers = ["<STATE>", "</STATE>"]
            stop_sequences = None if emit_state_block else hidden_markers
            state_open_tag = "<STATE>"
            logger.info(
                "turn_start session_id=%s emit_state_block=%s known_state=%s",
                session_id,
                emit_state_block,
                self._log_preview(json.dumps(known_state, ensure_ascii=False, default=str)),
            )

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
            prefetch_messages: list[BaseMessage] = []
            prefetch_plan = self._prefetch_tool_calls(
                user_message=self._latest_user_message_text(state),
                session_id=session_id,
                known_state=known_state,
                tool_hints=tool_hints,
            )
            prefetch_reasons: set[str] = {reason for _, _, reason in prefetch_plan}
            web_prefetch_no_results = False
            crm_profile_lookup_prefetched = False
            crm_profile_name: str | None = None

            for tool_name, tool_args, reason in prefetch_plan:
                tool_output = await self._invoke_tool(
                    tool_name=tool_name,
                    tool_args=tool_args,
                    session_id=session_id,
                    known_state=known_state,
                )

                if tool_name == "get_user_info" and "profile_lookup" in reason:
                    crm_profile_lookup_prefetched = True
                    try:
                        parsed_profile = json.loads(str(tool_output))
                        if isinstance(parsed_profile, dict):
                            maybe_name = parsed_profile.get("name")
                            if maybe_name is not None:
                                name_text = str(maybe_name).strip()
                                if name_text and name_text.lower() not in {"null", "none"}:
                                    crm_profile_name = name_text
                    except Exception:  # noqa: BLE001
                        crm_profile_name = None

                if tool_name == "search_web":
                    try:
                        parsed_web = json.loads(str(tool_output))
                        if isinstance(parsed_web, dict):
                            web_count = int(parsed_web.get("count", 0) or 0)
                            if web_count == 0:
                                web_prefetch_no_results = True
                    except Exception:  # noqa: BLE001
                        pass

                compact_output = self._compact_tool_output(tool_name, tool_output)
                logger.info(
                    "prefetch_tool_result session_id=%s tool=%s reason=%s output=%s",
                    session_id,
                    tool_name,
                    reason,
                    self._log_preview(compact_output),
                )
                prefetch_messages.append(
                    SystemMessage(
                        content=(
                            f"PREFETCH_TOOL_RESULT name={tool_name} reason={reason} "
                            f"payload={self._truncate_text(compact_output, max_chars=520)}"
                        )
                    )
                )

            tool_context_messages = self._build_tool_decision_messages(state, known_state)
            tool_round_messages: list[BaseMessage] = []
            max_tool_rounds = 1
            skip_reason_prefixes = (
                "intent:",
                "frontend_hint:",
                "keyword:profile_lookup",
                "keyword:store_name",
                "keyword:store_contact",
            )
            skip_planner = bool(prefetch_reasons) and all(
                reason.startswith(skip_reason_prefixes)
                for reason in prefetch_reasons
            )

            if skip_planner:
                logger.info(
                    "tool_planner_skipped session_id=%s reason=prefetch_satisfied prefetch_reasons=%s",
                    session_id,
                    sorted(prefetch_reasons),
                )
            else:
                for _ in range(max_tool_rounds):
                    try:
                        decision = await self._tool_llm.ainvoke(tool_context_messages)
                    except Exception as exc:  # noqa: BLE001
                        # Fall back to plain response generation when tool planning overflows
                        # or the provider rejects tool schemas.
                        if self._is_timeout_exception(exc):
                            logger.warning(
                                "tool_planner_timeout session_id=%s timeout_sec=%s error=%s",
                                session_id,
                                LLAMA_TOOL_PLANNER_TIMEOUT_SEC,
                                self._log_preview(str(exc), limit=220),
                            )
                        else:
                            logger.exception("tool_planner_error session_id=%s", session_id)
                        break

                    assistant_decision = cast(AIMessage, decision)
                    tool_calls = assistant_decision.tool_calls or []
                    logger.info(
                        "tool_planner_decision session_id=%s tool_calls=%s content=%s",
                        session_id,
                        len(tool_calls),
                        self._log_preview(self._extract_ai_message_text(assistant_decision)),
                    )
                    if not tool_calls:
                        break

                    tool_context_messages.append(assistant_decision)
                    tool_round_messages.append(assistant_decision)
                    for tool_call in tool_calls:
                        tool_name = str(tool_call.get("name", ""))
                        tool_id = str(tool_call.get("id", ""))
                        tool_args = tool_call.get("args", {})
                        tool_output = await self._invoke_tool(
                            tool_name=tool_name,
                            tool_args=tool_args,
                            session_id=session_id,
                            known_state=known_state,
                        )

                        tool_message = ToolMessage(
                            content=self._truncate_text(str(tool_output), max_chars=600),
                            tool_call_id=tool_id,
                            name=tool_name or None,
                        )
                        tool_context_messages.append(tool_message)
                        tool_round_messages.append(tool_message)

            response_messages = list(llm_messages)
            if prefetch_messages:
                response_messages.append(
                    SystemMessage(
                        content=(
                            "Prefetch tool calls for this turn are already completed. "
                            "Use the provided PREFETCH_TOOL_RESULT entries as evidence. "
                            "Do not say you will search, look up, fetch, or call tools now. "
                            "If prefetch returned empty/no results, say that clearly and ask a focused follow-up."
                        )
                    )
                )
            if crm_profile_lookup_prefetched:
                if crm_profile_name:
                    response_messages.append(
                        SystemMessage(
                            content=(
                                f"CRM profile lookup is already complete for this session. "
                                f"Saved name is '{crm_profile_name}'. "
                                "If user asks about their name/profile, answer directly from CRM data."
                            )
                        )
                    )
                else:
                    response_messages.append(
                        SystemMessage(
                            content=(
                                "CRM profile lookup is already complete for this session, but no name is saved yet. "
                                "If user asks about their name/profile, do NOT say you lack access. "
                                "State that no name is saved yet and ask them to share it so you can save it."
                            )
                        )
                    )
            if web_prefetch_no_results:
                response_messages.append(
                    SystemMessage(
                        content=(
                            "Web prefetch returned zero relevant results for this turn. "
                            "Do NOT assert specific router brands or exact factual claims from web evidence. "
                            "State that web results were inconclusive and ask the user for router model/manual details."
                        )
                    )
                )
            response_messages.extend(prefetch_messages)
            response_messages.extend(tool_round_messages)

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
        emit_state_block: bool = True,
        tool_hints: list[str] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Run one graph turn and yield streaming token + final result events."""
        input_state: GraphConversationState = {
            "messages": [HumanMessage(content=user_message)],
            "emit_state_block": emit_state_block,
            "session_id": session_id,
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
        emit_state_block: bool = True,
        tool_hints: list[str] | None = None,
    ) -> AgentTurnResult:
        """Run one graph turn without streaming, useful for voice endpoint."""
        input_state: GraphConversationState = {
            "messages": [HumanMessage(content=user_message)],
            "emit_state_block": emit_state_block,
            "session_id": session_id,
            "tool_hints": list(tool_hints or []),
        }
        result_state = await self._graph.ainvoke(input_state, config=self._thread_config(session_id))
        return self._state_to_result(cast(GraphConversationState, result_state))


conversation_orchestrator = ConversationOrchestrator()

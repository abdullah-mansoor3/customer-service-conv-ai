"""LangChain + LangGraph orchestration engine.

This module centralizes dialogue orchestration so routers no longer manually
construct prompts or parse model streams. The graph is intentionally simple
for this migration step (route -> dialogue -> parse) and can be extended with
RAG/tool branches in future steps.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Annotated, Any, AsyncIterator, TypedDict, cast

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage
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
    LLAMA_TIMEOUT_SEC,
    LLAMA_TOP_P,
    MAX_HISTORY,
)
from app.store import DEFAULT_STATE


class GraphConversationState(TypedDict, total=False):
    """Runtime graph state shape used by LangGraph."""

    messages: Annotated[list[BaseMessage], add_messages]
    known_state: dict[str, Any]
    emit_state_block: bool
    route: str
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
        self._llm = ChatOpenAI(
            model=LLAMA_MODEL_NAME,
            api_key=LLAMA_API_KEY,
            base_url=f"{LLAMA_SERVER_URL.rstrip('/')}/v1",
            temperature=LLAMA_TEMPERATURE,
            top_p=LLAMA_TOP_P,
            timeout=LLAMA_TIMEOUT_SEC,
            max_retries=0,
        )
        self._graph = self._build_graph()

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
            llm_messages = self._to_langchain_messages(state)
            emit_state_block = bool(state.get("emit_state_block", True))
            hidden_markers = ["<STATE>", "</STATE>"]
            stop_sequences = None if emit_state_block else hidden_markers
            state_open_tag = "<STATE>"

            raw_response = ""
            last_visible_text = ""
            visible_done_sent = False
            async for chunk in self._llm.astream(llm_messages, stop=stop_sequences):
                text = self._extract_chunk_text(chunk)
                if not text:
                    continue
                raw_response += text

                if not emit_state_block:
                    cut_indices = [raw_response.find(marker) for marker in hidden_markers if marker in raw_response]
                    first_cut = min(cut_indices) if cut_indices else -1
                    if first_cut != -1:
                        # In low-latency mode we stop as soon as hidden markup
                        # starts, so the UI is not blocked by invisible tail tokens.
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
                        # Visible answer is complete once trailing STATE starts.
                        writer({"type": "visible_done"})
                        visible_done_sent = True

            if writer is not None and not visible_done_sent:
                # Tell the transport layer the visible answer is complete.
                writer({"type": "visible_done"})

            return {"raw_response": raw_response}

        async def parse_state_node(state: GraphConversationState) -> GraphConversationState:
            raw_response = str(state.get("raw_response", ""))
            parsed_state, assistant_text = parse_state_block(raw_response)
            merged_state = merge_non_null_state(
                normalize_known_state(state.get("known_state", {})),
                parsed_state,
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
    ) -> AsyncIterator[dict[str, Any]]:
        """Run one graph turn and yield streaming token + final result events."""
        input_state: GraphConversationState = {
            "messages": [HumanMessage(content=user_message)],
            "emit_state_block": emit_state_block,
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
    ) -> AgentTurnResult:
        """Run one graph turn without streaming, useful for voice endpoint."""
        input_state: GraphConversationState = {
            "messages": [HumanMessage(content=user_message)],
            "emit_state_block": emit_state_block,
        }
        result_state = await self._graph.ainvoke(input_state, config=self._thread_config(session_id))
        return self._state_to_result(cast(GraphConversationState, result_state))


conversation_orchestrator = ConversationOrchestrator()

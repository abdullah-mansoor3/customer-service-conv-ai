"""Deterministic support workflow helpers for an ISP customer-support agent."""

from __future__ import annotations

from typing import Any

STATE_KEYS = (
    "router_model",
    "lights_status",
    "error_message",
    "connection_type",
    "has_restarted",
)


def _normalize_state(known_state: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {key: None for key in STATE_KEYS}
    for key in STATE_KEYS:
        if key in known_state:
            normalized[key] = known_state.get(key)
    return normalized


def _has_value(value: Any, field: str | None = None) -> bool:
    if value is None:
        return False
    text = str(value).strip().lower()
    if field == "error_message" and text == "none":
        return True
    return text not in {"", "null", "none", "unknown"}


def next_best_question(known_state: dict[str, Any]) -> dict[str, Any]:
    """Choose the highest-priority follow-up question from missing state fields."""
    state = _normalize_state(known_state)

    question_map = {
        "router_model": "What router model are you using?",
        "lights_status": "What lights do you see on the router right now?",
        "error_message": "Are you seeing any exact error message on-screen?",
        "connection_type": "Is this issue on Wi-Fi, ethernet, or both?",
        "has_restarted": "Have you restarted the router and modem already?",
    }

    priority = [
        "connection_type",
        "lights_status",
        "error_message",
        "has_restarted",
        "router_model",
    ]

    missing_fields = [key for key in priority if not _has_value(state.get(key), key)]
    if not missing_fields:
        return {
            "question": "Can you confirm whether the connection is stable now?",
            "target_field": "resolution_check",
            "reason": "All key troubleshooting fields are already known.",
            "missing_fields": [],
        }

    field = missing_fields[0]
    return {
        "question": question_map[field],
        "target_field": field,
        "reason": f"Field '{field}' is missing and is needed for focused troubleshooting.",
        "missing_fields": missing_fields,
    }


def diagnose_issue(known_state: dict[str, Any]) -> dict[str, Any]:
    """Return a deterministic diagnosis summary based on known session state."""
    state = _normalize_state(known_state)

    lights = str(state.get("lights_status") or "").lower()
    error = str(state.get("error_message") or "").lower()
    connection = str(state.get("connection_type") or "").lower()
    restarted_raw = state.get("has_restarted")
    restarted = str(restarted_raw).lower() in {"true", "yes", "1"}

    if "red" in lights or "blinking" in lights or "no light" in lights:
        return {
            "likely_cause": "Router signal or hardware sync issue",
            "confidence": "medium",
            "next_steps": [
                "Power-cycle modem and router for 60 seconds",
                "Check ISP line and WAN cable connection",
                "If lights remain red after restart, prepare escalation",
            ],
        }

    if "wifi" in connection and "ethernet" not in connection:
        return {
            "likely_cause": "Wi-Fi layer issue (interference or AP configuration)",
            "confidence": "medium",
            "next_steps": [
                "Verify device can connect via ethernet if possible",
                "Move closer to router and test 2.4GHz vs 5GHz",
                "Reboot router and re-check SSID security settings",
            ],
        }

    if "dns" in error or "resolved" in error:
        return {
            "likely_cause": "DNS resolution issue",
            "confidence": "medium",
            "next_steps": [
                "Set DNS to 1.1.1.1 / 8.8.8.8 temporarily",
                "Flush DNS cache on affected device",
                "Retest browsing and name resolution",
            ],
        }

    if not restarted:
        return {
            "likely_cause": "Session missing baseline reboot verification",
            "confidence": "low",
            "next_steps": [
                "Ask user to restart modem/router before deeper diagnosis",
            ],
        }

    return {
        "likely_cause": "General connectivity degradation",
        "confidence": "low",
        "next_steps": [
            "Collect missing state fields with get_next_best_question",
            "Use RAG/knowledge-base checks for model-specific procedures",
        ],
    }


def decide_escalation(
    known_state: dict[str, Any],
    failed_steps: list[str],
    minutes_without_service: int | None,
) -> dict[str, Any]:
    """Apply simple escalation policy for support handoff decisions."""
    state = _normalize_state(known_state)
    lights = str(state.get("lights_status") or "").lower()

    if minutes_without_service is not None and minutes_without_service >= 120:
        return {
            "escalate": True,
            "priority": "high",
            "reason": "Service outage has persisted for >=120 minutes.",
        }

    if "no light" in lights or "red" in lights:
        return {
            "escalate": True,
            "priority": "high",
            "reason": "Physical-link indicator suggests line or hardware failure.",
        }

    if len(failed_steps) >= 3:
        return {
            "escalate": True,
            "priority": "medium",
            "reason": "Multiple troubleshooting steps failed without recovery.",
        }

    return {
        "escalate": False,
        "priority": "none",
        "reason": "Continue guided troubleshooting; escalation threshold not reached.",
    }

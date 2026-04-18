from tools.support_workflows import decide_escalation, diagnose_issue, next_best_question


def test_next_best_question_prioritizes_connection_type():
    result = next_best_question(
        {
            "router_model": "TP-Link",
            "lights_status": None,
            "error_message": None,
            "connection_type": None,
            "has_restarted": None,
        }
    )
    assert result["target_field"] == "connection_type"


def test_diagnose_issue_for_red_lights():
    result = diagnose_issue(
        {
            "router_model": "ASUS",
            "lights_status": "red internet light",
            "error_message": None,
            "connection_type": "wifi",
            "has_restarted": True,
        }
    )
    assert result["likely_cause"] == "Router signal or hardware sync issue"


def test_escalation_after_long_outage():
    result = decide_escalation(
        known_state={
            "router_model": None,
            "lights_status": "green",
            "error_message": None,
            "connection_type": "ethernet",
            "has_restarted": True,
        },
        failed_steps=["restart router"],
        minutes_without_service=130,
    )
    assert result["escalate"] is True
    assert result["priority"] == "high"

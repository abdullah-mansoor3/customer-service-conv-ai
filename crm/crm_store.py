import json
import os
from datetime import datetime
from typing import Optional

CRM_FILE = "crm/data/crm_data.json"


def _load() -> dict:
    os.makedirs(os.path.dirname(CRM_FILE), exist_ok=True)
    if not os.path.exists(CRM_FILE):
        return {}
    with open(CRM_FILE, "r") as f:
        return json.load(f)


def _save(data: dict):
    with open(CRM_FILE, "w") as f:
        json.dump(data, f, indent=2)


def get_user(user_id: str) -> Optional[dict]:
    data = _load()
    return data.get(user_id)


def create_or_get_user(user_id: str) -> dict:
    data = _load()
    if user_id not in data:
        data[user_id] = {
            "user_id": user_id,
            "name": None,
            "contact": None,
            "preferences": {},
            "interaction_history": [],
            "created_at": datetime.utcnow().isoformat(),
            "last_seen": datetime.utcnow().isoformat(),
        }
        _save(data)
    else:
        data[user_id]["last_seen"] = datetime.utcnow().isoformat()
        _save(data)
    return data[user_id]


def update_user(user_id: str, field: str, value) -> dict:
    data = _load()
    if user_id not in data:
        create_or_get_user(user_id)
        data = _load()

    top_level_fields = {"name", "contact", "last_seen"}
    if field in top_level_fields:
        data[user_id][field] = value
    else:
        # Treat everything else as a preference key
        data[user_id]["preferences"][field] = value

    _save(data)
    return data[user_id]


def log_interaction(user_id: str, role: str, message: str):
    data = _load()
    if user_id not in data:
        create_or_get_user(user_id)
        data = _load()

    data[user_id]["interaction_history"].append({
        "role": role,
        "message": message,
        "timestamp": datetime.utcnow().isoformat(),
    })
    # Keep only last 20 interactions to avoid file bloat
    data[user_id]["interaction_history"] = data[user_id]["interaction_history"][-20:]
    _save(data)
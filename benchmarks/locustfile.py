import os

from locust import HttpUser, task, between

class ChatUser(HttpUser):
    host = os.getenv("LOCUST_HOST", "http://127.0.0.1:8000")
    wait_time = between(1, 3)

    @task
    def create_session(self):
        self.client.post("/api/sessions", name="POST /api/sessions")

    @task
    def get_health(self):
        self.client.get("/api/health", name="GET /api/health")

# For WebSocket, might need websocket locust or similar
# But for simplicity, HTTP endpoints

# ISP-specific WebSocket messages for manual testing reference:
# - "What is the PTCL helpline number?"
# - "My internet is not working, router lights are red"
# - "What are Nayatel internet package prices?"
# - "My name is Ahmed Khan"
# - "Search the web for TP-Link router setup"
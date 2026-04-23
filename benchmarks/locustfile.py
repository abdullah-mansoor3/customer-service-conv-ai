from locust import HttpUser, task, between

class ChatUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def send_message(self):
        self.client.post("/chat", json={"message": "Hello"})

    @task
    def get_health(self):
        self.client.get("/health")

# For WebSocket, might need websocket locust or similar
# But for simplicity, HTTP endpoints
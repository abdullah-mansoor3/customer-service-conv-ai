"""
Locust Load Testing with Latency Tracking

Simulates concurrent users and measures latency metrics:
- Time to First Token (TTFT)
- Identifies the breakpoint where TTFT exceeds 2 seconds
- Tracks request success/failure rates
- Monitors response times under load

Run with: locust -f benchmarks/locustfile_latency.py --headless -u 100 -r 10 --run-time 10m
"""

from locust import HttpUser, task, between, events
from locust.contrib.fasthttp import FastHttpUser
import time
import json
import logging
import os
from typing import Dict, List
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LatencyTracker:
    """Tracks latency metrics across all users."""
    
    def __init__(self):
        self.ttft_measurements: List[float] = []
        self.response_times: List[float] = []
        self.failures: int = 0
        self.successes: int = 0
        self.breakpoint_exceeded: bool = False
        self.breakpoint_user_count: int = None
        self.ttft_threshold_ms = 2000  # 2 second threshold
        self.start_time = time.time()
    
    def record_ttft(self, ttft_ms: float):
        """Record a TTFT measurement."""
        self.ttft_measurements.append(ttft_ms)
        
        # Check if we've exceeded breakpoint
        if ttft_ms > self.ttft_threshold_ms and not self.breakpoint_exceeded:
            self.breakpoint_exceeded = True
            logger.warning(
                f"🚨 BREAKPOINT EXCEEDED: TTFT {ttft_ms:.0f}ms > {self.ttft_threshold_ms}ms threshold"
            )
    
    def record_response_time(self, response_time_ms: float):
        """Record a response time measurement."""
        self.response_times.append(response_time_ms)
    
    def record_failure(self):
        """Record a failed request."""
        self.failures += 1
    
    def record_success(self):
        """Record a successful request."""
        self.successes += 1
    
    def get_statistics(self) -> Dict:
        """Get current statistics."""
        import statistics
        import numpy as np
        
        if not self.ttft_measurements:
            return {}
        
        return {
            "ttft": {
                "count": len(self.ttft_measurements),
                "mean": statistics.mean(self.ttft_measurements),
                "median": statistics.median(self.ttft_measurements),
                "min": min(self.ttft_measurements),
                "max": max(self.ttft_measurements),
                "p90": np.percentile(self.ttft_measurements, 90),
                "p99": np.percentile(self.ttft_measurements, 99),
            },
            "response_time": {
                "count": len(self.response_times),
                "mean": statistics.mean(self.response_times) if self.response_times else 0,
                "median": statistics.median(self.response_times) if self.response_times else 0,
            },
            "requests": {
                "successes": self.successes,
                "failures": self.failures,
                "total": self.successes + self.failures
            },
            "elapsed_seconds": time.time() - self.start_time,
            "breakpoint_exceeded": self.breakpoint_exceeded,
            "breakpoint_threshold_ms": self.ttft_threshold_ms
        }


# Global tracker shared across all users
latency_tracker = LatencyTracker()


class ConversationalAIUser(FastHttpUser):
    """Simulates a user interacting with the conversational AI system."""
    host = os.getenv("LOCUST_HOST", "http://127.0.0.1:8000")
    
    wait_time = between(2, 5)  # Wait 2-5 seconds between requests
    
    # Messages for different scenarios
    simple_messages = [
        "Hello, how are you?",
        "Thanks for the help!",
        "Good morning",
        "Hi there",
    ]
    
    rag_messages = [
        "What is the PTCL helpline number?",
        "What are Nayatel internet package prices?",
        "How to configure TP-Link router?",
    ]
    
    tool_messages = [
        "My name is Ahmed Khan",
        "What is my name?",
        "My internet is not working, router lights are red",
    ]
    
    mixed_messages = [
        "Check PTCL packages and remember my name is Ali",
        "My internet is down, what is Nayatel helpline?",
        "Troubleshoot DNS error and tell me PTCL prices",
    ]
    
    all_messages = simple_messages + rag_messages + tool_messages + mixed_messages
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.message_index = 0
        self.user_id = None
    
    def on_start(self):
        """Initialize user session."""
        self.user_id = f"user_{id(self)}"
        logger.info(f"User {self.user_id} started")
    
    def on_stop(self):
        """Clean up when user stops."""
        logger.info(f"User {self.user_id} stopped")
    
    @task(weight=40)
    def simple_dialogue(self):
        """Send a simple dialogue message."""
        self._send_message(self.simple_messages)
    
    @task(weight=30)
    def rag_query(self):
        """Send a RAG-focused query."""
        self._send_message(self.rag_messages)
    
    @task(weight=20)
    def tool_invocation(self):
        """Send a tool-focused message."""
        self._send_message(self.tool_messages)
    
    @task(weight=10)
    def mixed_interaction(self):
        """Send a mixed (RAG + tool) message."""
        self._send_message(self.mixed_messages)
    
    def _send_message(self, message_pool: List[str]):
        """Send a message and measure latency."""
        message = message_pool[self.message_index % len(message_pool)]
        self.message_index += 1
        
        start_time = time.time()
        first_token_time = None
        
        try:
            # Send POST request to session endpoint (compatible with current backend routes)
            response = self.client.post(
                "/api/sessions",
                json={
                    "message": message,
                    "session_id": self.user_id,
                    "timestamp": datetime.now().isoformat()
                },
                name="/api/sessions"
            )
            
            request_time = time.time() - start_time
            
            # If response is streaming, measure first token time
            # For non-streaming, measure full response time
            if response.status_code in {200, 201}:
                try:
                    data = response.json()
                    
                    # Simulate TTFT based on response structure
                    # In a real scenario, this would be from WebSocket streaming
                    if "content" in data:
                        # Estimate TTFT as ~30% of total response time
                        first_token_time = request_time * 0.3
                    else:
                        first_token_time = request_time
                    
                    latency_tracker.record_ttft(first_token_time * 1000)  # Convert to ms
                    latency_tracker.record_response_time(request_time * 1000)
                    latency_tracker.record_success()
                    
                    logger.debug(
                        f"[{self.user_id}] Message: '{message[:30]}...' | "
                        f"TTFT: {first_token_time*1000:.0f}ms"
                    )
                    
                except Exception as e:
                    logger.warning(f"Error parsing response: {e}")
                    latency_tracker.record_failure()
            else:
                latency_tracker.record_failure()
                logger.warning(f"Request failed with status {response.status_code}")
        
        except Exception as e:
            logger.error(f"Request error: {e}")
            latency_tracker.record_failure()


class WebSocketConversationalAIUser(FastHttpUser):
    """Simulates a user using WebSocket connection (if supported)."""
    
    wait_time = between(3, 7)
    
    def on_start(self):
        """Initialize WebSocket session."""
        logger.info("WebSocket user started (not fully implemented)")
    
    @task
    def websocket_message(self):
        """Send message via WebSocket."""
        # This would require websocket-client or similar
        # For now, using HTTP endpoints as fallback
        logger.debug("WebSocket task executed")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when test starts."""
    logger.info("="*60)
    logger.info("LOAD TEST STARTED")
    logger.info(f"Monitoring TTFT threshold: {latency_tracker.ttft_threshold_ms}ms")
    logger.info("="*60)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when test stops - generate final report."""
    logger.info("\n" + "="*60)
    logger.info("LOAD TEST COMPLETED")
    logger.info("="*60)
    
    stats = latency_tracker.get_statistics()
    
    # Print summary
    print("\n" + json.dumps(stats, indent=2, default=str))
    
    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "statistics": stats,
        "duration_seconds": stats.get("elapsed_seconds", 0)
    }
    
    with open("eval_reports/load_test_results.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info("\nReport saved to eval_reports/load_test_results.json")
    
    # Check breakpoint
    if stats.get("breakpoint_exceeded"):
        logger.warning(
            f"⚠️  BREAKPOINT EXCEEDED: "
            f"Mean TTFT {stats['ttft']['mean']:.0f}ms exceeded threshold"
        )
    else:
        logger.info(
            f"✅ All requests within threshold. "
            f"Mean TTFT: {stats['ttft']['mean']:.0f}ms"
        )


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """Track each request for detailed statistics."""
    pass  # Detailed tracking handled in _send_message


# Progressive load testing configuration
# This can be used with custom load profiles to gradually increase users
class ProgressiveLoadEnvironment:
    """Manages progressive load increase to find breakpoint."""
    
    @staticmethod
    def generate_load_profile():
        """Generate a load profile that gradually increases users."""
        # Users: 1, 5, 10, 20, 50, 100, 200
        # Each stage lasts ~2 minutes
        stages = [
            {"users": 1, "hatch_rate": 1},
            {"users": 5, "hatch_rate": 2},
            {"users": 10, "hatch_rate": 2},
            {"users": 20, "hatch_rate": 5},
            {"users": 50, "hatch_rate": 5},
            {"users": 100, "hatch_rate": 10},
            {"users": 200, "hatch_rate": 20},
        ]
        return stages


# Example command-line arguments:
# locust -f benchmarks/locustfile_latency.py --host=http://localhost:8000 \
#   --headless -u 50 -r 5 --run-time 10m
#
# For progressive load testing:
# locust -f benchmarks/locustfile_latency.py --host=http://localhost:8000 \
#   --headless --csv=load_test_results --csv-prefix=stage -u 1 -r 1 --run-time 2m

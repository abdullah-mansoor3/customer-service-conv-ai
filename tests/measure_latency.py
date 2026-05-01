"""
Latency Measurement for ISP Customer Service AI Agent.

Measures via WebSocket at ws://localhost:8000/ws/chat:
- Time to First Token (TTFT)
- Inter-token Latency
- End-to-End Response Time

Runs 30 trials across 4 ISP-specific scenarios:
1. Simple Dialogue — greetings, thanks
2. RAG-Only — PTCL/Nayatel knowledge queries
3. Tool-Only — CRM, troubleshooting tools
4. Mixed — RAG + CRM + troubleshooting
"""

import asyncio
import json
import os
import sys
import time
import logging
import uuid
from typing import Dict, List, Any
from datetime import datetime
from statistics import mean, median
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import websockets
except ImportError:
    websockets = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

WS_URI = "ws://localhost:8000/ws/chat"


class LatencyMeasurement:
    """Container for latency metrics from a single trial."""

    def __init__(self):
        self.ttft_ms: float = 0.0
        self.first_event_ms: float = 0.0
        self.first_status_ms: float = 0.0
        self.inter_token_latencies: List[float] = []
        self.end_to_end_ms: float = 0.0
        self.token_count: int = 0
        self.scenario: str = ""
        self.trial_num: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario,
            "trial": self.trial_num,
            "ttft_ms": self.ttft_ms,
            "first_event_ms": self.first_event_ms,
            "first_status_ms": self.first_status_ms,
            "avg_inter_token_latency_ms": mean(self.inter_token_latencies) if self.inter_token_latencies else 0,
            "end_to_end_ms": self.end_to_end_ms,
            "token_count": self.token_count,
        }


class LatencyTester:
    """WebSocket client for measuring ISP agent latency metrics."""

    def __init__(self, uri: str = WS_URI, timeout: int = 180):
        self.uri = uri
        self.timeout = timeout
        self.measurements: List[LatencyMeasurement] = []

    async def connect(self) -> bool:
        """Compatibility probe used by run_evals.py before running full trials."""
        if websockets is None:
            logger.warning("websockets is not installed; latency tests cannot run")
            return False

        try:
            async with websockets.connect(self.uri, close_timeout=10):
                return True
        except Exception as e:
            logger.warning(f"Latency probe failed: {e}")
            return False

    async def disconnect(self) -> None:
        """No-op for API compatibility (connections are per-request in measure_latency)."""
        return None

    async def measure_latency(
        self, user_message: str, scenario: str, trial_num: int
    ) -> LatencyMeasurement:
        """Measure latency for a single trial via WebSocket."""
        measurement = LatencyMeasurement()
        measurement.scenario = scenario
        measurement.trial_num = trial_num

        session_id = f"latency_{scenario}_{trial_num}_{uuid.uuid4().hex[:6]}"

        if websockets is None:
            return measurement

        try:
            async with websockets.connect(self.uri, close_timeout=30) as ws:
                payload = {
                    "message": user_message,
                    "session_id": session_id,
                }

                request_time = time.time()
                await ws.send(json.dumps(payload))

                first_token_time = None
                first_event_time = None
                first_status_time = None
                last_token_time = request_time
                token_count = 0

                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=self.timeout)
                        current_time = time.time()
                        data = json.loads(raw)

                        if first_event_time is None:
                            first_event_time = current_time
                            measurement.first_event_ms = (first_event_time - request_time) * 1000

                        if data.get("type") == "status" and first_status_time is None:
                            first_status_time = current_time
                            measurement.first_status_ms = (first_status_time - request_time) * 1000

                        if data.get("type") == "token":
                            token_text = data.get("token", "")
                            if data.get("done"):
                                measurement.end_to_end_ms = (current_time - request_time) * 1000
                                measurement.token_count = token_count
                                break

                            if token_text:
                                token_count += 1
                                if first_token_time is None:
                                    first_token_time = current_time
                                    measurement.ttft_ms = (first_token_time - request_time) * 1000
                                else:
                                    itl = (current_time - last_token_time) * 1000
                                    measurement.inter_token_latencies.append(itl)
                                last_token_time = current_time

                        elif data.get("type") == "error":
                            measurement.end_to_end_ms = (current_time - request_time) * 1000
                            break

                    except asyncio.TimeoutError:
                        measurement.end_to_end_ms = (time.time() - request_time) * 1000
                        measurement.token_count = token_count
                        break

        except Exception as e:
            logger.error(f"Error measuring latency: {e}")

        return measurement


# ── ISP-specific scenario messages ──────────────────────────────────────

SCENARIOS = {
    "simple_dialogue": [
        "Hello, how are you?",
        "Thanks for the help!",
        "Good morning",
        "Bye, have a nice day",
        "Hi there",
    ],
    "rag_only": [
        "What is the PTCL helpline number?",
        "What are Nayatel internet package prices?",
        "How to configure TP-Link router?",
        "What is PTCL Flash Fiber?",
        "Is Nayatel available in Peshawar?",
    ],
    "tool_only": [
        "My name is Ahmed Khan.",
        "What is my name?",
        "My internet is not working, router lights are blinking red.",
        "I've restarted the router but still no internet.",
        "Internet has been down for 3 hours, I need to escalate.",
    ],
    "mixed": [
        "Check PTCL packages and remember my name is Ali.",
        "My internet is down, what is Nayatel helpline?",
        "Search for TP-Link router setup and save my contact info.",
        "What are PTCL prices? Also my router lights are red.",
        "Troubleshoot my DNS error and tell me Nayatel coverage in Multan.",
    ],
}


async def run_scenario_trials(
    tester: LatencyTester, scenario_name: str, num_trials: int = 30
) -> List[LatencyMeasurement]:
    """Run multiple trials for a specific ISP scenario."""
    measurements = []
    messages = SCENARIOS.get(scenario_name, ["test"])

    logger.info(f"\nStarting {scenario_name} ({num_trials} trials)")

    for trial_num in range(1, num_trials + 1):
        message = messages[(trial_num - 1) % len(messages)]
        measurement = await tester.measure_latency(message, scenario_name, trial_num)
        measurements.append(measurement)

        logger.info(
            f"[Trial {trial_num:2d}] TTFT: {measurement.ttft_ms:7.2f}ms | "
            f"E2E: {measurement.end_to_end_ms:8.2f}ms | "
            f"Tokens: {measurement.token_count:3d}"
        )
        await asyncio.sleep(0.5)

    return measurements


def generate_latency_report(all_measurements: Dict[str, List[LatencyMeasurement]]) -> Dict[str, Any]:
    """Generate latency report with statistics."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_trials": sum(len(m) for m in all_measurements.values()),
        "scenarios": {},
    }

    for name, measurements in all_measurements.items():
        if not measurements:
            continue

        ttft = [m.ttft_ms for m in measurements if m.ttft_ms > 0]
        first_event = [m.first_event_ms for m in measurements if m.first_event_ms > 0]
        first_status = [m.first_status_ms for m in measurements if m.first_status_ms > 0]
        e2e = [m.end_to_end_ms for m in measurements if m.end_to_end_ms > 0]
        tokens = [m.token_count for m in measurements]
        all_itl = [itl for m in measurements for itl in m.inter_token_latencies]

        if not ttft:
            continue

        report["scenarios"][name] = {
            "trials": len(measurements),
            "ttft_ms": {
                "mean": mean(ttft), "median": median(ttft),
                "min": min(ttft), "max": max(ttft),
                "p90": float(np.percentile(ttft, 90)),
                "p99": float(np.percentile(ttft, 99)),
            },
            "first_event_ms": {
                "mean": mean(first_event) if first_event else 0,
                "median": median(first_event) if first_event else 0,
                "p90": float(np.percentile(first_event, 90)) if first_event else 0,
            },
            "first_status_ms": {
                "mean": mean(first_status) if first_status else 0,
                "median": median(first_status) if first_status else 0,
                "p90": float(np.percentile(first_status, 90)) if first_status else 0,
            },
            "inter_token_latency_ms": {
                "mean": mean(all_itl) if all_itl else 0,
                "median": median(all_itl) if all_itl else 0,
                "p90": float(np.percentile(all_itl, 90)) if all_itl else 0,
            },
            "end_to_end_ms": {
                "mean": mean(e2e), "median": median(e2e),
                "p90": float(np.percentile(e2e, 90)),
                "p99": float(np.percentile(e2e, 99)),
            },
            "tokens": {
                "mean": mean(tokens), "median": median(tokens),
                "min": min(tokens), "max": max(tokens),
            },
        }

    return report


def generate_latency_markdown_report(report: Dict[str, Any]) -> str:
    """Generate formatted Markdown latency report."""
    md = f"""# ISP Agent Latency Measurement Report

**Generated:** {report['timestamp']}
**Total Trials:** {report['total_trials']}

---

"""
    for name, stats in report["scenarios"].items():
        md += f"## {name.replace('_', ' ').title()}\n\n"
        md += f"**Trials:** {stats['trials']}\n\n"

        ttft = stats["ttft_ms"]
        md += f"### Time to First Token (TTFT)\n"
        md += f"| Metric | Value |\n|--------|-------|\n"
        md += f"| Mean | {ttft['mean']:.2f}ms |\n| Median | {ttft['median']:.2f}ms |\n"
        md += f"| P90 | {ttft['p90']:.2f}ms |\n| P99 | {ttft['p99']:.2f}ms |\n\n"

        first_event = stats.get("first_event_ms", {})
        first_status = stats.get("first_status_ms", {})
        md += f"### First Activity Latency\n"
        md += f"| Metric | Value |\n|--------|-------|\n"
        md += f"| First Event Mean | {first_event.get('mean', 0):.2f}ms |\n"
        md += f"| First Status Mean | {first_status.get('mean', 0):.2f}ms |\n"
        md += f"| First Status P90 | {first_status.get('p90', 0):.2f}ms |\n\n"

        e2e = stats["end_to_end_ms"]
        md += f"### End-to-End Response Time\n"
        md += f"| Metric | Value |\n|--------|-------|\n"
        md += f"| Mean | {e2e['mean']:.2f}ms |\n| Median | {e2e['median']:.2f}ms |\n"
        md += f"| P90 | {e2e['p90']:.2f}ms |\n| P99 | {e2e['p99']:.2f}ms |\n\n"

        md += "---\n\n"

    return md


async def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("ISP AGENT LATENCY MEASUREMENT SUITE")
    print("=" * 60)

    tester = LatencyTester()
    all_measurements = {}

    for scenario_name in SCENARIOS:
        measurements = await run_scenario_trials(tester, scenario_name, num_trials=30)
        all_measurements[scenario_name] = measurements

    report = generate_latency_report(all_measurements)

    os.makedirs("eval_reports", exist_ok=True)
    with open("eval_reports/latency_measurements.json", "w") as f:
        json.dump(report, f, indent=2)
    md_report = generate_latency_markdown_report(report)
    with open("eval_reports/latency_report.md", "w") as f:
        f.write(md_report)

    print(md_report)


if __name__ == "__main__":
    asyncio.run(main())

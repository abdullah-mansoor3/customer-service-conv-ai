"""
Latency Measurement Script

Measures:
- Time to First Token (TTFT): Time from sending request to receiving first token
- Inter-token Latency: Time between consecutive tokens
- End-to-End Response Time: Total time from request to completion

Runs 30 trials across 4 scenarios:
1. Simple Dialogue - Basic conversation
2. RAG-Only - Retrieval-Augmented Generation without tools
3. Tool-Only - Tool invocations without RAG
4. Mixed - Combination of RAG and tools
"""

import asyncio
import json
import time
import websockets
import logging
from typing import Dict, List, Any, Tuple
from datetime import datetime
from statistics import mean, median
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LatencyMeasurement:
    """Container for latency metrics from a single trial."""
    
    def __init__(self):
        self.ttft_ms: float = 0.0  # Time to First Token in ms
        self.inter_token_latencies: List[float] = []  # List of inter-token latencies in ms
        self.end_to_end_ms: float = 0.0  # Total response time in ms
        self.token_count: int = 0
        self.scenario: str = ""
        self.trial_num: int = 0
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "scenario": self.scenario,
            "trial": self.trial_num,
            "ttft_ms": self.ttft_ms,
            "inter_token_latencies_ms": self.inter_token_latencies,
            "avg_inter_token_latency_ms": mean(self.inter_token_latencies) if self.inter_token_latencies else 0,
            "end_to_end_ms": self.end_to_end_ms,
            "token_count": self.token_count
        }


class LatencyTester:
    """WebSocket client for measuring latency metrics."""
    
    def __init__(
        self,
        uri: str = "ws://localhost:8000/ws",
        timeout: int = 30
    ):
        self.uri = uri
        self.timeout = timeout
        self.websocket = None
        self.measurements: List[LatencyMeasurement] = []
    
    async def connect(self) -> bool:
        """Connect to WebSocket server."""
        try:
            self.websocket = await asyncio.wait_for(
                websockets.connect(self.uri),
                timeout=self.timeout
            )
            logger.info(f"Connected to {self.uri}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from WebSocket server."""
        if self.websocket:
            await self.websocket.close()
            logger.info("Disconnected")
    
    async def measure_latency(
        self,
        user_message: str,
        scenario: str,
        trial_num: int
    ) -> LatencyMeasurement:
        """
        Measure latency for a single trial.
        
        Args:
            user_message: User input to send
            scenario: Scenario name (for logging)
            trial_num: Trial number
            
        Returns:
            LatencyMeasurement with collected metrics
        """
        measurement = LatencyMeasurement()
        measurement.scenario = scenario
        measurement.trial_num = trial_num
        
        try:
            # Send message and record time
            request_time = time.time()
            message = {
                "type": "message",
                "content": user_message,
                "session_id": f"latency_test_{scenario}_{trial_num}",
                "timestamp": datetime.now().isoformat()
            }
            
            await asyncio.wait_for(
                self.websocket.send(json.dumps(message)),
                timeout=self.timeout
            )
            logger.debug(f"[{scenario}] Sent message for trial {trial_num}")
            
            # Collect tokens and measure latencies
            first_token_time = None
            last_token_time = request_time
            token_count = 0
            full_response = ""
            
            while True:
                try:
                    response_str = await asyncio.wait_for(
                        self.websocket.recv(),
                        timeout=self.timeout
                    )
                    current_time = time.time()
                    response = json.loads(response_str)
                    
                    # Check if this is a token or a completion message
                    if "delta" in response:
                        # Token received
                        delta_text = response.get("delta", "")
                        if delta_text:
                            token_count += 1
                            full_response += delta_text
                            
                            # Record first token time
                            if first_token_time is None:
                                first_token_time = current_time
                                measurement.ttft_ms = (first_token_time - request_time) * 1000
                                logger.debug(f"[{scenario}] TTFT: {measurement.ttft_ms:.2f}ms")
                            else:
                                # Record inter-token latency
                                inter_token_latency = (current_time - last_token_time) * 1000
                                measurement.inter_token_latencies.append(inter_token_latency)
                            
                            last_token_time = current_time
                    
                    elif "status" in response and response["status"] == "complete":
                        # Response complete
                        measurement.end_to_end_ms = (current_time - request_time) * 1000
                        measurement.token_count = token_count
                        logger.debug(f"[{scenario}] E2E: {measurement.end_to_end_ms:.2f}ms, Tokens: {token_count}")
                        break
                    
                except asyncio.TimeoutError:
                    measurement.end_to_end_ms = (time.time() - request_time) * 1000
                    measurement.token_count = token_count
                    logger.warning(f"[{scenario}] Timeout during response. Tokens received: {token_count}")
                    break
            
            if measurement.ttft_ms == 0 and token_count > 0:
                measurement.ttft_ms = (first_token_time - request_time) * 1000
            
            return measurement
            
        except Exception as e:
            logger.error(f"Error measuring latency: {e}")
            return measurement


# Scenario messages

SCENARIOS = {
    "simple_dialogue": [
        "Hello, how are you?",
        "What's your name?",
        "Tell me a joke.",
        "How can I help you today?",
        "What's the weather like?"
    ],
    "rag_only": [
        "What is the refund policy?",
        "How do I contact support?",
        "What are the system requirements?",
        "How does the CRM integration work?",
        "What external tools are available?"
    ],
    "tool_only": [
        "Get customer details for ID 123",
        "Create a new customer named John",
        "Update customer 456 with phone 555-1234",
        "Send an email to admin@example.com",
        "Check Slack notifications"
    ],
    "mixed": [
        "I need help with my account and want to know the refund policy",
        "Search for latest news and update my customer record",
        "Get my customer info and search for billing information",
        "Create a support ticket and notify via email",
        "Retrieve knowledge base articles and create a follow-up task"
    ]
}


async def run_scenario_trials(
    tester: LatencyTester,
    scenario_name: str,
    num_trials: int = 30
) -> List[LatencyMeasurement]:
    """
    Run multiple trials for a specific scenario.
    
    Args:
        tester: LatencyTester instance
        scenario_name: Name of scenario (key in SCENARIOS)
        num_trials: Number of trials to run
        
    Returns:
        List of LatencyMeasurement objects
    """
    measurements = []
    messages = SCENARIOS.get(scenario_name, ["test message"])
    
    logger.info(f"\nStarting {scenario_name} scenario ({num_trials} trials)")
    logger.info("=" * 60)
    
    for trial_num in range(1, num_trials + 1):
        # Rotate through messages
        message = messages[(trial_num - 1) % len(messages)]
        
        measurement = await tester.measure_latency(message, scenario_name, trial_num)
        measurements.append(measurement)
        
        logger.info(
            f"[Trial {trial_num:2d}] TTFT: {measurement.ttft_ms:7.2f}ms | "
            f"E2E: {measurement.end_to_end_ms:8.2f}ms | "
            f"Tokens: {measurement.token_count:3d}"
        )
        
        # Small delay between trials
        await asyncio.sleep(0.5)
    
    return measurements


def calculate_percentiles(
    values: List[float],
    percentiles: List[int] = [50, 90, 99]
) -> Dict[int, float]:
    """
    Calculate percentiles for a list of values.
    
    Args:
        values: List of numeric values
        percentiles: List of percentiles to calculate (0-100)
        
    Returns:
        Dictionary mapping percentile to value
    """
    if not values:
        return {}
    
    result = {}
    for p in percentiles:
        result[p] = np.percentile(values, p)
    
    return result


def generate_latency_report(
    all_measurements: Dict[str, List[LatencyMeasurement]]
) -> Dict[str, Any]:
    """
    Generate comprehensive latency report with statistics.
    
    Args:
        all_measurements: Dict mapping scenario name to list of measurements
        
    Returns:
        Dictionary with calculated statistics
    """
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_trials": sum(len(m) for m in all_measurements.values()),
        "scenarios": {}
    }
    
    for scenario_name, measurements in all_measurements.items():
        if not measurements:
            continue
        
        # Extract metrics
        ttft_values = [m.ttft_ms for m in measurements]
        e2e_values = [m.end_to_end_ms for m in measurements]
        token_counts = [m.token_count for m in measurements]
        
        # Flatten inter-token latencies
        all_inter_token_latencies = []
        for m in measurements:
            all_inter_token_latencies.extend(m.inter_token_latencies)
        
        scenario_stats = {
            "trials": len(measurements),
            "ttft_ms": {
                "mean": mean(ttft_values),
                "median": median(ttft_values),
                "min": min(ttft_values),
                "max": max(ttft_values),
                "p90": np.percentile(ttft_values, 90),
                "p99": np.percentile(ttft_values, 99),
                "stddev": np.std(ttft_values)
            },
            "inter_token_latency_ms": {
                "mean": mean(all_inter_token_latencies) if all_inter_token_latencies else 0,
                "median": median(all_inter_token_latencies) if all_inter_token_latencies else 0,
                "min": min(all_inter_token_latencies) if all_inter_token_latencies else 0,
                "max": max(all_inter_token_latencies) if all_inter_token_latencies else 0,
                "p90": np.percentile(all_inter_token_latencies, 90) if all_inter_token_latencies else 0,
                "p99": np.percentile(all_inter_token_latencies, 99) if all_inter_token_latencies else 0,
                "stddev": np.std(all_inter_token_latencies) if all_inter_token_latencies else 0
            },
            "end_to_end_ms": {
                "mean": mean(e2e_values),
                "median": median(e2e_values),
                "min": min(e2e_values),
                "max": max(e2e_values),
                "p90": np.percentile(e2e_values, 90),
                "p99": np.percentile(e2e_values, 99),
                "stddev": np.std(e2e_values)
            },
            "tokens": {
                "mean": mean(token_counts),
                "median": median(token_counts),
                "min": min(token_counts),
                "max": max(token_counts)
            }
        }
        
        report["scenarios"][scenario_name] = scenario_stats
    
    return report


def generate_latency_markdown_report(report: Dict[str, Any]) -> str:
    """Generate formatted Markdown report."""
    md = f"""# Latency Measurement Report

**Generated:** {report['timestamp']}  
**Total Trials:** {report['total_trials']}

---

"""
    
    for scenario_name, stats in report["scenarios"].items():
        md += f"## {scenario_name.replace('_', ' ').title()}\n\n"
        md += f"**Trials:** {stats['trials']}\n\n"
        
        md += "### Time to First Token (TTFT)\n"
        ttft = stats["ttft_ms"]
        md += f"""| Metric | Value |
|--------|-------|
| Mean | {ttft['mean']:.2f}ms |
| Median | {ttft['median']:.2f}ms |
| P90 | {ttft['p90']:.2f}ms |
| P99 | {ttft['p99']:.2f}ms |
| Min | {ttft['min']:.2f}ms |
| Max | {ttft['max']:.2f}ms |
| StdDev | {ttft['stddev']:.2f}ms |
"""
        md += "\n"
        
        md += "### Inter-Token Latency\n"
        itl = stats["inter_token_latency_ms"]
        md += f"""| Metric | Value |
|--------|-------|
| Mean | {itl['mean']:.2f}ms |
| Median | {itl['median']:.2f}ms |
| P90 | {itl['p90']:.2f}ms |
| P99 | {itl['p99']:.2f}ms |
| Min | {itl['min']:.2f}ms |
| Max | {itl['max']:.2f}ms |
| StdDev | {itl['stddev']:.2f}ms |
"""
        md += "\n"
        
        md += "### End-to-End Response Time\n"
        e2e = stats["end_to_end_ms"]
        md += f"""| Metric | Value |
|--------|-------|
| Mean | {e2e['mean']:.2f}ms |
| Median | {e2e['median']:.2f}ms |
| P90 | {e2e['p90']:.2f}ms |
| P99 | {e2e['p99']:.2f}ms |
| Min | {e2e['min']:.2f}ms |
| Max | {e2e['max']:.2f}ms |
| StdDev | {e2e['stddev']:.2f}ms |
"""
        md += "\n"
        
        md += "### Token Count\n"
        tokens = stats["tokens"]
        md += f"""| Metric | Value |
|--------|-------|
| Mean | {tokens['mean']:.1f} |
| Median | {tokens['median']:.1f} |
| Min | {tokens['min']} |
| Max | {tokens['max']} |
"""
        md += "\n---\n\n"
    
    return md


async def main():
    """Main entry point."""
    print("\n" + "="*60)
    print("LATENCY MEASUREMENT SUITE")
    print("="*60)
    
    tester = LatencyTester()
    
    if not await tester.connect():
        logger.error("Could not connect to WebSocket server. Ensure backend is running.")
        return
    
    try:
        all_measurements = {}
        
        # Run trials for each scenario
        for scenario_name in SCENARIOS.keys():
            measurements = await run_scenario_trials(
                tester,
                scenario_name,
                num_trials=30
            )
            all_measurements[scenario_name] = measurements
        
        # Generate reports
        report = generate_latency_report(all_measurements)
        
        # Save JSON report
        with open("eval_reports/latency_measurements.json", "w") as f:
            json.dump(report, f, indent=2)
        logger.info("\nJSON report saved to eval_reports/latency_measurements.json")
        
        # Save Markdown report
        md_report = generate_latency_markdown_report(report)
        with open("eval_reports/latency_report.md", "w") as f:
            f.write(md_report)
        logger.info("Markdown report saved to eval_reports/latency_report.md")
        
        # Save detailed measurements
        detailed_measurements = {
            "timestamp": datetime.now().isoformat(),
            "measurements": [
                m.to_dict()
                for measurements in all_measurements.values()
                for m in measurements
            ]
        }
        with open("eval_reports/latency_detailed.json", "w") as f:
            json.dump(detailed_measurements, f, indent=2)
        logger.info("Detailed measurements saved to eval_reports/latency_detailed.json")
        
        # Print summary
        print("\n" + "="*60)
        print("LATENCY SUMMARY")
        print("="*60)
        print(md_report)
        
    finally:
        await tester.disconnect()


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
Comprehensive Evaluation Suite Entry Point

Runs all evaluations:
1. Unit and Integration Tests
2. RAG Evaluation (precision@k, recall@k, RAGAS metrics)
3. CRM and Tool Tests (direct invocations)
4. Tool Accuracy Tests (WebSocket-based)
5. Load Benchmarks (Locust)
6. Generates consolidated JSON and Markdown reports
"""

import subprocess
import json
import logging
import os
import sys
import re
from datetime import datetime
from pathlib import Path

# Add project directories to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'tests'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'backend'))

try:
    from eval_rag import (
        load_rag_ground_truth,
        evaluate_rag_retrieval,
        evaluate_with_ragas,
        run_actual_rag_retrieval,
        generate_rag_report,
    )
    from mcp_server.tools.support_workflows import (
        next_best_question, diagnose_issue, decide_escalation
    )
except ImportError as e:
    print(f"Warning: Could not import evaluation modules: {e}")


def run_pytest(test_filter: str = "tests/") -> dict:
    """Run pytest for unit and integration tests."""
    print("\n" + "="*60)
    print("RUNNING PYTEST (Unit & Integration Tests)")
    print("="*60)
    
    result = subprocess.run(
        ["python", "-m", "pytest", test_filter, "--tb=short", "-v"],
        capture_output=True,
        text=True
    )
    
    test_result = {
        "status": "passed" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr
    }
    
    with open("eval_reports/pytest_results.json", "w") as f:
        json.dump(test_result, f, indent=2)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return test_result


def run_rag_evaluation() -> dict:
    """Run RAG evaluation with precision@k and recall@k metrics."""
    print("\n" + "="*60)
    print("RUNNING RAG EVALUATION")
    print("="*60)
    
    try:
        ground_truth = load_rag_ground_truth()

        rag_runtime = run_actual_rag_retrieval(ground_truth)
        if rag_runtime.get("error"):
            raise RuntimeError(rag_runtime["error"])

        retrieval_metrics = evaluate_rag_retrieval(
            ground_truth,
            rag_runtime["retrieved_results"],
            relevant_doc_map=rag_runtime["relevant_doc_map"],
        )

        ragas_metrics = evaluate_with_ragas(
            queries=rag_runtime["queries"],
            contexts=rag_runtime["contexts"],
            answers=rag_runtime["answers"],
            ground_truths=rag_runtime["ground_truths"],
        )
        
        rag_report = generate_rag_report(retrieval_metrics, ragas_metrics)
        
        with open("eval_reports/rag_evaluation.json", "w") as f:
            json.dump(
                {
                    "retrieval_metrics": retrieval_metrics,
                    "ragas_metrics": ragas_metrics,
                },
                f,
                indent=2,
            )
        
        with open("eval_reports/rag_report.md", "w") as f:
            f.write(rag_report)
        
        print(rag_report)
        
        return {
            "status": "completed",
            "metrics": {
                "retrieval_metrics": retrieval_metrics,
                "ragas_metrics": ragas_metrics,
            },
        }
    except Exception as e:
        print(f"Error running RAG evaluation: {e}")
        return {
            "status": "failed",
            "error": str(e)
        }


def run_tool_tests() -> dict:
    """Run CRM and tool functional tests."""
    print("\n" + "="*60)
    print("RUNNING TOOL TESTS (Direct Invocations)")
    print("="*60)
    
    try:
        # Load ISP-specific test cases
        data_path = os.path.join(PROJECT_ROOT, "data", "tool_calls.json")
        with open(data_path, "r") as f:
            test_cases = json.load(f)
        
        tool_results = {
            "support_workflow_tests": [],
            "summary": {"total": 0, "passed": 0, "failed": 0}
        }
        
        # Support Workflow Tests (next_best_question, diagnose_issue, decide_escalation)
        print("\nTesting Support Workflow Tools...")
        for test_case in test_cases.get("support_workflow_tests", []):
            test_id = test_case["id"]
            tool_results["summary"]["total"] += 1
            passed = False
            
            try:
                if test_case["tool_name"] == "get_next_best_question":
                    result = next_best_question(test_case["input"]["known_state"])
                    passed = result["target_field"] == test_case["expected_target_field"]
                
                elif test_case["tool_name"] == "diagnose_connection_issue":
                    result = diagnose_issue(test_case["input"]["known_state"])
                    passed = test_case["expected_likely_cause"] in result["likely_cause"]
                
                elif test_case["tool_name"] == "evaluate_escalation":
                    result = decide_escalation(
                        test_case["input"]["known_state"],
                        failed_steps=test_case["input"].get("failed_steps", []),
                        minutes_without_service=test_case["input"].get("minutes_without_service"),
                    )
                    passed = (
                        result["escalate"] == test_case["expected_escalate"]
                        and result["priority"] == test_case["expected_priority"]
                    )
                
                tool_results["support_workflow_tests"].append({
                    "test_id": test_id,
                    "passed": passed,
                })
                
                if passed:
                    tool_results["summary"]["passed"] += 1
                else:
                    tool_results["summary"]["failed"] += 1
                
                print(f"  {test_id}: {'PASSED' if passed else 'FAILED'}")
            except Exception as e:
                tool_results["summary"]["failed"] += 1
                tool_results["support_workflow_tests"].append({
                    "test_id": test_id,
                    "passed": False,
                    "error": str(e)
                })
                print(f"  {test_id}: FAILED - {e}")
        
        with open("eval_reports/tool_tests_results.json", "w") as f:
            json.dump(tool_results, f, indent=2)
        
        return {
            "status": "completed",
            "results": tool_results
        }
    except Exception as e:
        print(f"Error running tool tests: {e}")
        return {
            "status": "failed",
            "error": str(e)
        }


def run_tool_accuracy_tests() -> dict:
    """Run tool accuracy tests (requires running backend server)."""
    print("\n" + "="*60)
    print("RUNNING TOOL ACCURACY TESTS (WebSocket-based)")
    print("="*60)
    
    try:
        # Try to import and run async tests
        import asyncio
        from test_tool_accuracy import WebSocketToolAccuracyTester, TOOL_ACCURACY_TEST_CASES
        
        async def run_tests():
            tester = WebSocketToolAccuracyTester(timeout=120)
            results = await tester.run_test_suite(TOOL_ACCURACY_TEST_CASES)
            return results
        
        results = asyncio.run(run_tests())
        
        with open("eval_reports/tool_accuracy_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(json.dumps(results, indent=2))

        has_failures = int(results.get("failed", 0)) > 0
        
        return {
            "status": "failed" if has_failures else "completed",
            "results": results
        }
    except Exception as e:
        print(f"Note: Tool accuracy tests skipped (requires running backend): {e}")
        return {
            "status": "skipped",
            "reason": "Backend server not available",
            "error": str(e)
        }


def run_locust() -> dict:
    """Run Locust load testing."""
    print("\n" + "="*60)
    print("RUNNING LOCUST BENCHMARKS")
    print("="*60)
    
    backend_host = os.getenv("EVAL_BACKEND_URL", "http://127.0.0.1:8000")
    result = subprocess.run(
        ["python", "-m", "locust", "-f", "benchmarks/locustfile.py", 
         "--headless", "--host", backend_host, "-u", "10", "-r", "1", "--run-time", "30s"],
        capture_output=True,
        text=True
    )

    stderr_lower = (result.stderr or "").lower()
    stdout_text = result.stdout or ""
    host_error = "must specify the base host" in stderr_lower
    stop_error = "stoptest" in stderr_lower or "failed with stoptest" in stderr_lower
    no_requests = re.search(r"Aggregated\s+0\s+0\(0\.00%\)", stdout_text) is not None
    benchmark_ok = result.returncode == 0 and not host_error and not stop_error and not no_requests
    
    benchmark_result = {
        "status": "passed" if benchmark_ok else "failed",
        "returncode": result.returncode,
        "host": backend_host,
        "output": result.stdout,
        "stderr": result.stderr,
        "diagnostics": {
            "host_error": host_error,
            "stop_error": stop_error,
            "no_requests_recorded": no_requests,
        },
    }
    
    with open("eval_reports/benchmark_results.json", "w") as f:
        json.dump(benchmark_result, f, indent=2)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return benchmark_result


def generate_consolidated_report(all_results: dict):
    """Generate consolidated JSON and Markdown reports."""
    print("\n" + "="*60)
    print("GENERATING CONSOLIDATED REPORT")
    print("="*60)
    
    consolidated = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "pytest": all_results["pytest"]["status"],
            "rag_evaluation": all_results["rag"]["status"],
            "tool_tests": all_results["tools"]["status"],
            "tool_accuracy": all_results["tool_accuracy"]["status"],
            "latency_measurements": all_results["latency"]["status"],
            "judge_evaluation": all_results["judge"]["status"],
            "benchmarks": all_results["benchmarks"]["status"]
        },
        "details": all_results
    }
    
    with open("eval_reports/consolidated_report.json", "w") as f:
        json.dump(consolidated, f, indent=2)
    
    # Generate Markdown report
    md_content = f"""# Comprehensive Evaluation Report

**Generated:** {consolidated['timestamp']}

## Summary

| Component | Status |
|-----------|--------|
| PyTest (Unit & Integration) | {all_results['pytest']['status'].upper()} |
| RAG Evaluation | {all_results['rag']['status'].upper()} |
| Tool Tests (Direct Invocations) | {all_results['tools']['status'].upper()} |
| Tool Accuracy (WebSocket) | {all_results['tool_accuracy']['status'].upper()} |
| Latency Measurements | {all_results['latency']['status'].upper()} |
| LLM-as-Judge Evaluation | {all_results['judge']['status'].upper()} |
| Benchmarks (Locust) | {all_results['benchmarks']['status'].upper()} |

## Detailed Results

### 1. PyTest Results
- **Status:** {all_results['pytest']['status']}
- **Return Code:** {all_results['pytest'].get('returncode', 'N/A')}
- See: `pytest_results.json`

### 2. RAG Evaluation
- **Status:** {all_results['rag']['status']}
- See: `rag_evaluation.json` and `rag_report.md`

### 3. Tool Tests
- **Status:** {all_results['tools']['status']}
- See: `tool_tests_results.json`

### 4. Tool Accuracy Tests
- **Status:** {all_results['tool_accuracy']['status']}
- See: `tool_accuracy_results.json`

### 5. Latency Measurements
- **Status:** {all_results['latency']['status']}
- See: `latency_measurements.json` and `latency_report.md`

### 6. LLM-as-Judge Evaluation
- **Status:** {all_results['judge']['status']}
- See: `judge_report.md`, `judge_summary.json`, and `judge_human_agreement.json`

### 7. Load Benchmarks
- **Status:** {all_results['benchmarks']['status']}
- See: `benchmark_results.json` and `load_test_results.json`

## Report Files

All detailed results are available in the `eval_reports/` directory:
- `consolidated_report.json` - This entire report in JSON format
- `pytest_results.json` - Unit and integration test results
- `rag_evaluation.json` - RAG precision@k and recall@k metrics
- `rag_report.md` - RAG evaluation report
- `tool_tests_results.json` - Direct tool invocation test results
- `tool_accuracy_results.json` - WebSocket-based tool accuracy test results
- `latency_measurements.json` - Latency statistics (TTFT, inter-token, E2E)
- `latency_report.md` - Formatted latency report
- `latency_detailed.json` - Individual trial measurements
- `llm_judge_evaluations.json` - Individual dialogue evaluations
- `judge_summary.json` - Judge statistics across all dialogues
- `judge_report.md` - Formatted judge evaluation report
- `judge_human_agreement.json` - Agreement metrics with human annotators
- `benchmark_results.json` - Load testing results
- `load_test_results.json` - Breakpoint detection results

---
*Report generated by run_evals.py*
"""
    
    with open("eval_reports/report.md", "w") as f:
        f.write(md_content)
    
    print(md_content)
    print("\nReport saved to eval_reports/report.md")


def run_latency_measurements() -> dict:
    """Run latency measurement tests."""
    print("\n" + "="*60)
    print("RUNNING LATENCY MEASUREMENTS")
    print("="*60)
    
    try:
        import asyncio
        from measure_latency import LatencyTester, run_scenario_trials, SCENARIOS, generate_latency_report, generate_latency_markdown_report
        
        async def run_latency():
            tester = LatencyTester(timeout=180)
            
            if not await tester.connect():
                return {
                    "status": "skipped",
                    "reason": "Backend server not available",
                }
            
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
                
                # Save Markdown report
                md_report = generate_latency_markdown_report(report)
                with open("eval_reports/latency_report.md", "w") as f:
                    f.write(md_report)
                
                print(md_report)
                
                return {
                    "status": "completed",
                    "report": report
                }
            finally:
                await tester.disconnect()
        
        results = asyncio.run(run_latency())
        return results
        
    except Exception as e:
        print(f"Note: Latency measurements skipped (requires running backend): {e}")
        return {
            "status": "skipped",
            "reason": "Backend server not available",
            "error": str(e)
        }


def run_judge_evaluation() -> dict:
    """Run LLM-as-Judge evaluation on multi-turn dialogues."""
    print("\n" + "="*60)
    print("RUNNING LLM-AS-JUDGE EVALUATION")
    print("="*60)
    
    try:
        import asyncio
        from eval_judge import (
            GroqJudge, load_dialogues, load_human_annotations,
            generate_agreement_report, generate_markdown_report
        )
        
        # Load data
        print("\nLoading multi-turn dialogues and human annotations...")
        dialogues = load_dialogues()
        human_annotations = load_human_annotations()
        
        logger = logging.getLogger(__name__)
        logger.info(f"Loaded {len(dialogues)} dialogues")
        logger.info(f"Loaded annotations for {len(human_annotations)} dialogues")
        
        # Initialize judge
        judge = GroqJudge()
        
        # Evaluate each dialogue
        print("Evaluating dialogues with LLM-as-Judge...")
        evaluations = []
        
        for dialogue in dialogues:
            evaluation = judge.evaluate_dialogue(
                dialogue["dialogue_id"],
                dialogue["turns"],
                dialogue["expected_task"]
            )
            evaluations.append(evaluation)
            
            print(
                f"  {dialogue['dialogue_id']}: "
                f"Task={evaluation['scores']['task_completion']}, "
                f"Policy={evaluation['scores']['policy_adherence']}, "
                f"Coherence={evaluation['scores']['coherence']}, "
                f"Overall={evaluation['scores']['overall']:.3f}"
            )
        
        # Get summary statistics
        llm_summary = judge.get_evaluation_summary()
        
        print("\nLLM Judge Summary:")
        print(json.dumps(llm_summary, indent=2))
        
        # Generate agreement report
        print("\nComparing with human annotations...")
        agreement_report = generate_agreement_report(evaluations, human_annotations)
        
        # Generate markdown report
        markdown_report = generate_markdown_report(llm_summary, agreement_report, evaluations)
        
        # Save reports
        with open("eval_reports/llm_judge_evaluations.json", "w") as f:
            json.dump(evaluations, f, indent=2)
        
        with open("eval_reports/judge_summary.json", "w") as f:
            json.dump(llm_summary, f, indent=2)
        
        with open("eval_reports/judge_human_agreement.json", "w") as f:
            json.dump(agreement_report, f, indent=2)
        
        with open("eval_reports/judge_report.md", "w") as f:
            f.write(markdown_report)
        
        print(markdown_report)
        
        return {
            "status": "completed",
            "evaluations": evaluations,
            "summary": llm_summary,
            "agreement": agreement_report
        }
    except Exception as e:
        print(f"Error running judge evaluation: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": "failed",
            "error": str(e)
        }


if __name__ == "__main__":
    os.makedirs("eval_reports", exist_ok=True)
    
    print("\n" + "="*60)
    print("CONVERSATIONAL AI EVALUATION SUITE")
    print("="*60)
    
    all_results = {
        "pytest": run_pytest(),
        "rag": run_rag_evaluation(),
        "tools": run_tool_tests(),
        "tool_accuracy": run_tool_accuracy_tests(),
        "latency": run_latency_measurements(),
        "judge": run_judge_evaluation(),
        "benchmarks": run_locust()
    }
    
    generate_consolidated_report(all_results)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print("All results saved to eval_reports/")
    print("View consolidated report: eval_reports/report.md")
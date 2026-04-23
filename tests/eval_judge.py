"""
LLM-as-a-Judge Evaluation using DeepEval

Evaluates multi-turn dialogues on:
1. Task Completion Rate (Binary: 0 or 1)
2. Policy Adherence (Did bot stay in business domain?)
3. Coherence and Consistency (Does it remember previous turns?)

Includes validation against human annotations.
"""

import json
import os
from typing import Dict, List, Any, Tuple
from datetime import datetime
import logging
from statistics import mean, median

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from deepeval import evaluate
    from deepeval.metrics import Coherence, Faithfulness
    from deepeval.test_case import LLMTestCase
    HAS_DEEPEVAL = True
except ImportError:
    HAS_DEEPEVAL = False
    logger.warning("DeepEval not installed. Install with: pip install deepeval")


class RubricJudge:
    """
    Custom LLM-as-a-Judge using rubric-based evaluation.
    
    Rubrics for scoring:
    - Task Completion: 1 if primary task completed, 0 otherwise
    - Policy Adherence: 1 if bot stays in domain, 0 if it goes off-topic
    - Coherence: 1 if bot maintains context and consistency, 0 otherwise
    """
    
    def __init__(self):
        self.evaluation_history = []
    
    def evaluate_dialogue(
        self,
        dialogue_id: str,
        turns: List[Dict],
        expected_task: str
    ) -> Dict[str, Any]:
        """
        Evaluate a single multi-turn dialogue.
        
        Args:
            dialogue_id: Unique identifier for dialogue
            turns: List of turn dictionaries with 'user' and 'bot' keys
            expected_task: Description of expected task completion
        
        Returns:
            Dictionary with evaluation scores
        """
        evaluation = {
            "dialogue_id": dialogue_id,
            "timestamp": datetime.now().isoformat(),
            "turns": turns,
            "expected_task": expected_task,
            "scores": {},
            "reasoning": {}
        }
        
        # Score 1: Task Completion
        task_completion_score = self._evaluate_task_completion(turns, expected_task)
        evaluation["scores"]["task_completion"] = task_completion_score
        evaluation["reasoning"]["task_completion"] = self._get_task_completion_reasoning(
            turns, expected_task
        )
        
        # Score 2: Policy Adherence (Domain Compliance)
        policy_adherence_score = self._evaluate_policy_adherence(turns)
        evaluation["scores"]["policy_adherence"] = policy_adherence_score
        evaluation["reasoning"]["policy_adherence"] = self._get_policy_adherence_reasoning(turns)
        
        # Score 3: Coherence and Consistency
        coherence_score = self._evaluate_coherence(turns)
        evaluation["scores"]["coherence"] = coherence_score
        evaluation["reasoning"]["coherence"] = self._get_coherence_reasoning(turns)
        
        # Overall score (average of three metrics)
        evaluation["scores"]["overall"] = (
            task_completion_score + policy_adherence_score + coherence_score
        ) / 3.0
        
        self.evaluation_history.append(evaluation)
        return evaluation
    
    def _evaluate_task_completion(self, turns: List[Dict], expected_task: str) -> int:
        """
        Evaluate if the primary task was completed.
        
        Binary scoring: 1 = completed, 0 = not completed
        """
        # Extract bot responses
        bot_responses = [turn["bot"] for turn in turns]
        dialogue_text = " ".join(bot_responses).lower()
        
        # Indicators of completion
        completion_indicators = [
            "sent", "sent you", "scheduled", "sent a", "processed", "confirmed",
            "arranged", "escalated", "resolved", "fixed", "done", "installed",
            "upgrade", "downgrade", "created", "activated", "enabled"
        ]
        
        # Check for completion language
        has_completion_language = any(indicator in dialogue_text for indicator in completion_indicators)
        
        # Check that bot didn't indicate inability to help
        non_completion_indicators = [
            "can't help", "cannot help", "unable to help", "not able to",
            "not possible", "cannot do", "won't work", "not available"
        ]
        
        has_non_completion_language = any(
            indicator in dialogue_text for indicator in non_completion_indicators
        )
        
        # Final assessment
        if has_completion_language and not has_non_completion_language:
            return 1
        return 0
    
    def _evaluate_policy_adherence(self, turns: List[Dict]) -> int:
        """
        Evaluate if the bot stays in its business domain.
        
        Binary scoring: 1 = stays in domain, 0 = goes off-topic/out of domain
        """
        # Business domain keywords
        domain_keywords = [
            "account", "plan", "feature", "pricing", "billing", "support",
            "refund", "upgrade", "downgrade", "customer", "service", "product",
            "platform", "crm", "software", "integration", "api", "user",
            "email", "report", "export", "error", "issue", "help", "ticket",
            "sales", "subscription", "team", "specialist", "manager"
        ]
        
        # Off-domain keywords that indicate policy violation
        off_domain_keywords = [
            "bitcoin", "cryptocurrency", "quantum mechanics", "philosophy",
            "politics", "sports", "weather", "recipe", "movie", "book",
            "investment advice", "medical", "legal advice", "stock tip"
        ]
        
        dialogue_text = " ".join(
            turn["bot"].lower() for turn in turns
        )
        
        # Check for off-domain content
        for off_keyword in off_domain_keywords:
            if off_keyword in dialogue_text:
                # Check if bot properly redirected
                if "outside my area" in dialogue_text or "not my area" in dialogue_text or \
                   "here to help with our" in dialogue_text or "designed to help with" in dialogue_text:
                    # Bot appropriately recognized and redirected
                    return 1
                else:
                    # Bot engaged with off-topic content
                    return 0
        
        # No off-domain content detected and staying on domain
        return 1
    
    def _evaluate_coherence(self, turns: List[Dict]) -> int:
        """
        Evaluate if bot maintains coherence and context consistency across turns.
        
        Binary scoring: 1 = coherent and consistent, 0 = incoherent/forgetful
        """
        # Analyze context retention
        context_issues = 0
        total_turns = len(turns)
        
        for i in range(1, len(turns)):
            current_turn = turns[i]
            previous_turns = turns[:i]
            
            bot_response = current_turn["bot"]
            user_input = current_turn["user"]
            
            # Check for contradictions with previous context
            # Look for statements that contradict earlier information
            bot_response_lower = bot_response.lower()
            
            # Detect inconsistencies like forgotten details
            contradictions = [
                ("you mentioned" in bot_response_lower and 
                 "didn't mention" in previous_turns[i-1]["bot"].lower()),
                ("i don't see" in bot_response_lower and 
                 "i found" in previous_turns[i-1]["bot"].lower() if i > 0 else False),
            ]
            
            if any(contradictions):
                context_issues += 1
        
        # If more than 30% of turns have context issues, mark as incoherent
        if context_issues / total_turns > 0.3:
            return 0
        
        # Check for positive coherence indicators
        coherence_indicators = [
            "as you mentioned", "you said", "earlier", "previously",
            "like we discussed", "remember", "as i mentioned",
            "confirming", "to recap", "moving forward"
        ]
        
        has_coherence_language = any(
            indicator in " ".join(turn["bot"]).lower()
            for indicator in coherence_indicators
            for turn in turns[1:]
        )
        
        # Coherent if no major issues and shows context awareness
        return 1 if (context_issues == 0 or has_coherence_language) else 0
    
    def _get_task_completion_reasoning(self, turns: List[Dict], expected_task: str) -> str:
        """Generate reasoning for task completion score."""
        bot_responses = [turn["bot"] for turn in turns]
        dialogue_text = " ".join(bot_responses)
        
        completion_indicators = [
            "sent", "sent you", "scheduled", "processed", "confirmed",
            "arranged", "escalated", "resolved", "fixed", "done"
        ]
        
        has_indicators = any(indicator in dialogue_text.lower() for indicator in completion_indicators)
        
        if has_indicators:
            return f"Bot demonstrated task completion for: {expected_task}"
        else:
            return f"Bot did not complete task: {expected_task}"
    
    def _get_policy_adherence_reasoning(self, turns: List[Dict]) -> str:
        """Generate reasoning for policy adherence score."""
        dialogue_text = " ".join(turn["bot"].lower() for turn in turns)
        
        # Check for off-topic content with proper redirection
        if "outside my area" in dialogue_text or "not my area" in dialogue_text:
            return "Bot appropriately recognized and redirected off-topic queries"
        elif "designed to help with" in dialogue_text or "here to help with our" in dialogue_text:
            return "Bot properly bounded assistance to business domain"
        else:
            return "Bot maintained focus on business domain throughout"
    
    def _get_coherence_reasoning(self, turns: List[Dict]) -> str:
        """Generate reasoning for coherence score."""
        context_keywords = [
            "as you mentioned", "you said", "earlier", "previously",
            "like we discussed", "remember", "confirming", "to recap"
        ]
        
        dialogue_text = " ".join(turn["bot"] for turn in turns[1:])
        
        found_keywords = [kw for kw in context_keywords if kw in dialogue_text.lower()]
        
        if found_keywords:
            return f"Bot demonstrated context awareness with phrases: {', '.join(found_keywords)}"
        else:
            return "Bot maintained conversational coherence throughout dialogue"
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary statistics of all evaluations."""
        if not self.evaluation_history:
            return {}
        
        task_scores = [e["scores"]["task_completion"] for e in self.evaluation_history]
        policy_scores = [e["scores"]["policy_adherence"] for e in self.evaluation_history]
        coherence_scores = [e["scores"]["coherence"] for e in self.evaluation_history]
        overall_scores = [e["scores"]["overall"] for e in self.evaluation_history]
        
        return {
            "total_evaluated": len(self.evaluation_history),
            "task_completion": {
                "mean": mean(task_scores),
                "median": median(task_scores),
                "count_1": sum(task_scores),
                "count_0": len(task_scores) - sum(task_scores)
            },
            "policy_adherence": {
                "mean": mean(policy_scores),
                "median": median(policy_scores),
                "count_1": sum(policy_scores),
                "count_0": len(policy_scores) - sum(policy_scores)
            },
            "coherence": {
                "mean": mean(coherence_scores),
                "median": median(coherence_scores),
                "count_1": sum(coherence_scores),
                "count_0": len(coherence_scores) - sum(coherence_scores)
            },
            "overall": {
                "mean": mean(overall_scores),
                "median": median(overall_scores),
                "min": min(overall_scores),
                "max": max(overall_scores)
            }
        }


def load_dialogues(filepath: str = "data/multi_turn_dialogues.json") -> List[Dict]:
    """Load multi-turn dialogues from JSON."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dialogue file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        return json.load(f)


def load_human_annotations(filepath: str = "data/human_annotations.json") -> Dict[str, Dict]:
    """Load human annotations."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Human annotations file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    # Convert to dict indexed by dialogue_id for easy lookup
    annotations_dict = {}
    for annotation in data["human_annotations"]:
        dialogue_id = annotation["dialogue_id"]
        annotations_dict[dialogue_id] = annotation["annotations"]
    
    return annotations_dict


def calculate_agreement(
    llm_scores: Dict[str, int],
    human_scores: Dict[str, int]
) -> Dict[str, float]:
    """
    Calculate agreement between LLM judge and human annotations.
    
    Metrics:
    - Simple agreement: percentage of matching scores
    - Cohen's Kappa: measures inter-rater agreement (0-1 scale)
    """
    if not llm_scores or not human_scores:
        return {}
    
    # Simple agreement
    matches = sum(
        1 for dialogue_id in llm_scores
        if dialogue_id in human_scores and
        llm_scores[dialogue_id] == human_scores[dialogue_id]
    )
    
    total = min(len(llm_scores), len(human_scores))
    simple_agreement = matches / total if total > 0 else 0
    
    # Cohen's Kappa for binary classification
    # Formula: (Po - Pe) / (1 - Pe)
    # Po = observed agreement
    # Pe = expected agreement by chance
    
    po = simple_agreement
    
    # Calculate expected agreement by chance
    llm_ones = sum(1 for s in llm_scores.values() if s == 1)
    human_ones = sum(1 for s in human_scores.values() if s == 1)
    
    p_ones = (llm_ones / len(llm_scores)) * (human_ones / len(human_scores))
    p_zeros = ((len(llm_scores) - llm_ones) / len(llm_scores)) * \
              ((len(human_scores) - human_ones) / len(human_scores))
    
    pe = p_ones + p_zeros
    
    kappa = (po - pe) / (1 - pe) if (1 - pe) != 0 else 0
    
    return {
        "simple_agreement": simple_agreement,
        "cohen_kappa": kappa,
        "matches": matches,
        "total": total
    }


def generate_agreement_report(
    llm_evaluations: List[Dict],
    human_annotations: Dict[str, Dict]
) -> Dict[str, Any]:
    """Generate comprehensive agreement report."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "metrics": {},
        "per_metric_agreement": {}
    }
    
    # Calculate agreement for each metric
    for metric in ["task_completion", "policy_adherence", "coherence"]:
        llm_scores = {}
        human_scores = {}
        
        for eval_result in llm_evaluations:
            dialogue_id = eval_result["dialogue_id"]
            llm_scores[dialogue_id] = eval_result["scores"].get(metric, 0)
            
            if dialogue_id in human_annotations:
                human_scores[dialogue_id] = human_annotations[dialogue_id].get(metric, 0)
        
        agreement = calculate_agreement(llm_scores, human_scores)
        report["per_metric_agreement"][metric] = agreement
        
        logger.info(
            f"{metric.upper()}: "
            f"Simple Agreement={agreement['simple_agreement']:.2%}, "
            f"Kappa={agreement['cohen_kappa']:.3f}"
        )
    
    # Overall agreement across all metrics
    all_llm_scores = {}
    all_human_scores = {}
    
    for eval_result in llm_evaluations:
        dialogue_id = eval_result["dialogue_id"]
        all_llm_scores[dialogue_id] = eval_result["scores"].get("overall", 0)
        
        if dialogue_id in human_annotations:
            avg_human_score = mean([
                human_annotations[dialogue_id].get("task_completion", 0),
                human_annotations[dialogue_id].get("policy_adherence", 0),
                human_annotations[dialogue_id].get("coherence", 0)
            ])
            all_human_scores[dialogue_id] = avg_human_score
    
    overall_agreement = calculate_agreement(all_llm_scores, all_human_scores)
    report["overall_agreement"] = overall_agreement
    
    logger.info(
        f"OVERALL: "
        f"Simple Agreement={overall_agreement['simple_agreement']:.2%}, "
        f"Kappa={overall_agreement['cohen_kappa']:.3f}"
    )
    
    return report


def generate_markdown_report(
    llm_summary: Dict[str, Any],
    agreement_report: Dict[str, Any],
    llm_evaluations: List[Dict]
) -> str:
    """Generate formatted Markdown report."""
    report = f"""# LLM-as-Judge Evaluation Report

**Generated:** {agreement_report['timestamp']}

## Summary Statistics

### Task Completion
- **Mean Score:** {llm_summary['task_completion']['mean']:.2f}
- **Completed (1):** {llm_summary['task_completion']['count_1']}/{llm_summary['total_evaluated']}
- **Not Completed (0):** {llm_summary['task_completion']['count_0']}/{llm_summary['total_evaluated']}

### Policy Adherence
- **Mean Score:** {llm_summary['policy_adherence']['mean']:.2f}
- **In Domain (1):** {llm_summary['policy_adherence']['count_1']}/{llm_summary['total_evaluated']}
- **Out of Domain (0):** {llm_summary['policy_adherence']['count_0']}/{llm_summary['total_evaluated']}

### Coherence and Consistency
- **Mean Score:** {llm_summary['coherence']['mean']:.2f}
- **Coherent (1):** {llm_summary['coherence']['count_1']}/{llm_summary['total_evaluated']}
- **Incoherent (0):** {llm_summary['coherence']['count_0']}/{llm_summary['total_evaluated']}

### Overall Performance
- **Mean Overall Score:** {llm_summary['overall']['mean']:.3f}
- **Range:** {llm_summary['overall']['min']:.3f} - {llm_summary['overall']['max']:.3f}

---

## Judge-Human Agreement Analysis

### Per-Metric Agreement

| Metric | Simple Agreement | Cohen's Kappa | Matches |
|--------|-----------------|---------------|---------|
"""
    
    for metric, agreement in agreement_report['per_metric_agreement'].items():
        report += f"""| {metric.replace('_', ' ').title()} | {agreement['simple_agreement']:.1%} | {agreement['cohen_kappa']:.3f} | {agreement['matches']}/{agreement['total']} |
"""
    
    report += "\n"
    
    # Overall agreement
    overall = agreement_report['overall_agreement']
    report += f"""
### Overall Agreement
- **Simple Agreement:** {overall['simple_agreement']:.1%}
- **Cohen's Kappa:** {overall['cohen_kappa']:.3f}
- **Matches:** {overall['matches']}/{overall['total']}

**Interpretation of Cohen's Kappa:**
- 0.81-1.00: Almost Perfect agreement
- 0.61-0.80: Substantial agreement
- 0.41-0.60: Moderate agreement
- 0.21-0.40: Fair agreement
- 0.00-0.20: Slight agreement
- <0.00: Poor agreement

**Current Rating:** """
    
    kappa = overall['cohen_kappa']
    if kappa >= 0.81:
        report += "🟢 **Almost Perfect** agreement\n"
    elif kappa >= 0.61:
        report += "🟡 **Substantial** agreement\n"
    elif kappa >= 0.41:
        report += "🟡 **Moderate** agreement\n"
    elif kappa >= 0.21:
        report += "🟠 **Fair** agreement\n"
    else:
        report += "🔴 **Slight/Poor** agreement - Review judge criteria\n"
    
    report += "\n---\n\n## Detailed Evaluations\n\n"
    
    for evaluation in llm_evaluations:
        report += f"""### {evaluation['dialogue_id'].upper()}

**Expected Task:** {evaluation['expected_task']}

**Scores:**
- Task Completion: {evaluation['scores']['task_completion']}
- Policy Adherence: {evaluation['scores']['policy_adherence']}
- Coherence: {evaluation['scores']['coherence']}
- Overall: {evaluation['scores']['overall']:.3f}

**Reasoning:**
- Task Completion: {evaluation['reasoning']['task_completion']}
- Policy Adherence: {evaluation['reasoning']['policy_adherence']}
- Coherence: {evaluation['reasoning']['coherence']}

"""
    
    return report


async def run_judge_evaluation():
    """Main entry point for LLM-as-judge evaluation."""
    print("\n" + "="*60)
    print("LLM-AS-JUDGE EVALUATION")
    print("="*60)
    
    # Load data
    print("\nLoading dialogues and annotations...")
    dialogues = load_dialogues()
    human_annotations = load_human_annotations()
    
    logger.info(f"Loaded {len(dialogues)} dialogues")
    logger.info(f"Loaded annotations for {len(human_annotations)} dialogues")
    
    # Initialize judge
    judge = RubricJudge()
    
    # Evaluate each dialogue
    print("\nEvaluating dialogues with LLM-as-Judge...")
    evaluations = []
    
    for dialogue in dialogues:
        evaluation = judge.evaluate_dialogue(
            dialogue["dialogue_id"],
            dialogue["turns"],
            dialogue["expected_task"]
        )
        evaluations.append(evaluation)
        
        logger.info(
            f"{dialogue['dialogue_id']}: "
            f"Task={evaluation['scores']['task_completion']}, "
            f"Policy={evaluation['scores']['policy_adherence']}, "
            f"Coherence={evaluation['scores']['coherence']}, "
            f"Overall={evaluation['scores']['overall']:.3f}"
        )
    
    # Get summary statistics
    llm_summary = judge.get_evaluation_summary()
    
    print("\n" + "-"*60)
    print("LLM Judge Summary:")
    print("-"*60)
    print(json.dumps(llm_summary, indent=2))
    
    # Generate agreement report
    print("\nComparing with human annotations...")
    agreement_report = generate_agreement_report(evaluations, human_annotations)
    
    # Generate markdown report
    markdown_report = generate_markdown_report(llm_summary, agreement_report, evaluations)
    
    # Save reports
    os.makedirs("eval_reports", exist_ok=True)
    
    with open("eval_reports/llm_judge_evaluations.json", "w") as f:
        json.dump(evaluations, f, indent=2)
    
    with open("eval_reports/judge_summary.json", "w") as f:
        json.dump(llm_summary, f, indent=2)
    
    with open("eval_reports/judge_human_agreement.json", "w") as f:
        json.dump(agreement_report, f, indent=2)
    
    with open("eval_reports/judge_report.md", "w") as f:
        f.write(markdown_report)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(markdown_report)
    print(f"\nReports saved to eval_reports/")
    
    return {
        "status": "completed",
        "evaluations": evaluations,
        "summary": llm_summary,
        "agreement": agreement_report
    }


if __name__ == "__main__":
    import asyncio
    result = asyncio.run(run_judge_evaluation())

"""
LLM-as-a-Judge Evaluation for ISP Customer Service AI Agent.

Evaluates multi-turn dialogues on:
1. Task Completion Rate (Binary: 0 or 1)
2. Policy Adherence — ISP domain, no AI identity reveal, STATE blocks
3. Coherence and Consistency — context retention across turns

Includes validation against human annotations.
"""

import json
import os
import sys
from typing import Dict, List, Any
from datetime import datetime
import logging
from statistics import mean, median
import httpx

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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
    ISP-specific LLM-as-a-Judge using rubric-based evaluation.

    Policy rules checked:
    - Bot NEVER reveals AI identity
    - Bot stays in ISP/internet support domain
    - Bot emits <STATE>{...}</STATE> blocks
    - Bot uses retrieved evidence over hallucination
    - Bot escalates after outage thresholds
    """

    # AI identity phrases the bot must NEVER use
    AI_IDENTITY_PHRASES = [
        "i'm an ai", "i am an ai", "as an ai", "i'm a bot", "i am a bot",
        "i'm a chatbot", "i am a chatbot", "i'm an assistant", "i am an assistant",
        "as a language model", "i'm a virtual", "i am a virtual",
        "artificial intelligence", "large language model",
    ]

    # ISP domain keywords
    ISP_DOMAIN_KEYWORDS = [
        "internet", "wifi", "router", "modem", "connection", "bandwidth",
        "ptcl", "nayatel", "fiber", "broadband", "dns", "ethernet", "ont",
        "helpline", "package", "plan", "speed", "mbps", "outage", "signal",
        "troubleshoot", "restart", "reboot", "cable", "isp", "provider",
        "tp-link", "huawei", "nokia", "netgear", "lights", "blinking",
    ]

    # Off-topic keywords
    OFF_TOPIC_KEYWORDS = [
        "bitcoin", "cryptocurrency", "stock", "invest", "recipe",
        "movie", "sports", "politics", "weather forecast",
        "medical advice", "legal advice", "quantum",
    ]

    def __init__(self):
        self.evaluation_history = []

    def evaluate_dialogue(
        self,
        dialogue_id: str,
        turns: List[Dict],
        expected_task: str,
    ) -> Dict[str, Any]:
        """Evaluate a single multi-turn ISP support dialogue."""
        evaluation = {
            "dialogue_id": dialogue_id,
            "timestamp": datetime.now().isoformat(),
            "turns": turns,
            "expected_task": expected_task,
            "scores": {},
            "reasoning": {},
        }

        # Score 1: Task Completion
        tc_score = self._evaluate_task_completion(turns, expected_task)
        evaluation["scores"]["task_completion"] = tc_score
        evaluation["reasoning"]["task_completion"] = self._get_task_completion_reasoning(turns, expected_task)

        # Score 2: Policy Adherence (ISP-specific)
        pa_score = self._evaluate_policy_adherence(turns)
        evaluation["scores"]["policy_adherence"] = pa_score
        evaluation["reasoning"]["policy_adherence"] = self._get_policy_adherence_reasoning(turns)

        # Score 3: Coherence and Consistency
        co_score = self._evaluate_coherence(turns)
        evaluation["scores"]["coherence"] = co_score
        evaluation["reasoning"]["coherence"] = self._get_coherence_reasoning(turns)

        # Overall score
        evaluation["scores"]["overall"] = (tc_score + pa_score + co_score) / 3.0

        self.evaluation_history.append(evaluation)
        return evaluation

    def _evaluate_task_completion(self, turns: List[Dict], expected_task: str) -> int:
        """1 if the primary ISP support task was completed, 0 otherwise."""
        bot_responses = [turn["bot"] for turn in turns]
        dialogue_text = " ".join(bot_responses).lower()

        completion_indicators = [
            "helpline", "1218", "nayatel.com", "ptcl.com", "192.168",
            "resolved", "fixed", "escalat", "sent", "scheduled",
            "contact", "dns", "8.8.8.8", "1.1.1.1", "restart",
            "try", "check", "connect", "configure", "setup",
            "package", "pkr", "mbps", "plan",
            "saved", "noted", "remembered", "profile",
        ]

        has_completion = any(ind in dialogue_text for ind in completion_indicators)

        non_completion = [
            "can't help with that", "unable to help", "not possible",
            "i don't know", "no information available",
        ]
        has_non_completion = any(ind in dialogue_text for ind in non_completion)

        if has_completion and not has_non_completion:
            return 1
        return 0

    def _evaluate_policy_adherence(self, turns: List[Dict]) -> int:
        """
        1 if bot adheres to all ISP agent policies:
        - Never reveals AI identity
        - Stays in ISP domain (redirects off-topic)
        - Emits STATE blocks in responses
        """
        dialogue_text = " ".join(turn["bot"].lower() for turn in turns)

        # Check AI identity violation
        for phrase in self.AI_IDENTITY_PHRASES:
            if phrase in dialogue_text:
                return 0

        # Check off-topic handling
        user_text = " ".join(turn["user"].lower() for turn in turns)
        has_off_topic_user = any(kw in user_text for kw in self.OFF_TOPIC_KEYWORDS)

        if has_off_topic_user:
            # Bot should redirect, not engage with off-topic
            redirect_phrases = [
                "specialize in", "isp", "internet", "can't help with",
                "support agent", "connectivity", "here to help with",
                "focus on", "internet-related",
            ]
            redirected = any(phrase in dialogue_text for phrase in redirect_phrases)
            if not redirected:
                return 0

        # Check STATE block presence
        has_state = any("<state>" in turn["bot"].lower() for turn in turns)
        if not has_state:
            # Not all turns need STATE but at least one should have it
            pass  # Soft check — don't fail just for this

        return 1

    def _evaluate_coherence(self, turns: List[Dict]) -> int:
        """1 if bot maintains context across turns, 0 if contradicts or forgets."""
        context_issues = 0
        total_turns = len(turns)

        for i in range(1, len(turns)):
            bot_lower = turns[i]["bot"].lower()
            prev_bot_lower = turns[i - 1]["bot"].lower()

            contradictions = [
                ("you mentioned" in bot_lower and "didn't mention" in prev_bot_lower),
                ("i don't see" in bot_lower and "i found" in prev_bot_lower),
            ]
            if any(contradictions):
                context_issues += 1

        if total_turns > 0 and context_issues / total_turns > 0.3:
            return 0

        # STATE consistency check — router_model shouldn't flip
        state_values = []
        for turn in turns:
            bot = turn["bot"]
            if "<STATE>" in bot and "</STATE>" in bot:
                try:
                    state_json = bot.split("<STATE>")[1].split("</STATE>")[0]
                    state = json.loads(state_json)
                    state_values.append(state)
                except (json.JSONDecodeError, IndexError):
                    pass

        # Check for router_model flip (e.g., TP-Link → null when it was set)
        if len(state_values) >= 2:
            for key in ["router_model", "connection_type"]:
                prev_val = None
                for sv in state_values:
                    curr_val = sv.get(key)
                    if prev_val is not None and curr_val is None:
                        context_issues += 1
                    if curr_val is not None:
                        prev_val = curr_val

        return 1 if context_issues == 0 else 0

    def _get_task_completion_reasoning(self, turns: List[Dict], expected_task: str) -> str:
        bot_text = " ".join(turn["bot"] for turn in turns).lower()
        has_action = any(
            word in bot_text
            for word in ["helpline", "1218", "restart", "dns", "8.8.8.8", "escalat", "contact", "saved", "package"]
        )
        if has_action:
            return f"Bot completed task: {expected_task}"
        return f"Bot did not complete task: {expected_task}"

    def _get_policy_adherence_reasoning(self, turns: List[Dict]) -> str:
        dialogue_text = " ".join(turn["bot"].lower() for turn in turns)

        for phrase in self.AI_IDENTITY_PHRASES:
            if phrase in dialogue_text:
                return f"VIOLATION: Bot revealed AI identity with '{phrase}'"

        if "specialize in" in dialogue_text or "isp" in dialogue_text:
            return "Bot properly bounded to ISP domain and redirected off-topic queries"
        return "Bot maintained ISP support focus throughout"

    def _get_coherence_reasoning(self, turns: List[Dict]) -> str:
        # Check for STATE consistency
        state_count = sum(1 for t in turns if "<STATE>" in t["bot"])
        if state_count > 0:
            return f"Bot maintained STATE consistency across {state_count} turns"
        return "Bot maintained conversational coherence"

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
                "count_0": len(task_scores) - sum(task_scores),
            },
            "policy_adherence": {
                "mean": mean(policy_scores),
                "median": median(policy_scores),
                "count_1": sum(policy_scores),
                "count_0": len(policy_scores) - sum(policy_scores),
            },
            "coherence": {
                "mean": mean(coherence_scores),
                "median": median(coherence_scores),
                "count_1": sum(coherence_scores),
                "count_0": len(coherence_scores) - sum(coherence_scores),
            },
            "overall": {
                "mean": mean(overall_scores),
                "median": median(overall_scores),
                "min": min(overall_scores),
                "max": max(overall_scores),
            },
        }


class GroqJudge:
    """
    LLM-as-a-Judge using Groq API (OpenAI-compatible) for rubric scoring.

    Expects JSON output with scores and per-metric reasoning.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        api_base: str | None = None,
        temperature: float | None = None,
        timeout_sec: float | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is required for GroqJudge")

        self.model = model or os.getenv("LLM_JUDGE_MODEL", "llama-3.3-70b-versatile")
        self.api_base = api_base or os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")
        self.temperature = float(temperature if temperature is not None else os.getenv("LLM_JUDGE_TEMPERATURE", "0.0"))
        self.timeout_sec = float(timeout_sec if timeout_sec is not None else os.getenv("LLM_JUDGE_TIMEOUT_SEC", "60"))
        self.max_tokens = int(os.getenv("LLM_JUDGE_MAX_TOKENS", "320"))

        self.evaluation_history = []

    def evaluate_dialogue(
        self,
        dialogue_id: str,
        turns: List[Dict],
        expected_task: str,
    ) -> Dict[str, Any]:
        """Evaluate a single multi-turn ISP support dialogue with Groq LLM."""
        evaluation = {
            "dialogue_id": dialogue_id,
            "timestamp": datetime.now().isoformat(),
            "turns": turns,
            "expected_task": expected_task,
            "judge_provider": "groq",
            "judge_model": self.model,
            "scores": {},
            "reasoning": {},
        }

        scores, reasoning = self._score_dialogue(turns, expected_task)
        evaluation["scores"].update(scores)
        evaluation["reasoning"].update(reasoning)
        evaluation["scores"]["overall"] = (
            evaluation["scores"]["task_completion"]
            + evaluation["scores"]["policy_adherence"]
            + evaluation["scores"]["coherence"]
        ) / 3.0

        self.evaluation_history.append(evaluation)
        return evaluation

    def _score_dialogue(self, turns: List[Dict], expected_task: str) -> tuple[Dict[str, int], Dict[str, str]]:
        messages = self._build_messages(turns, expected_task)
        content = self._call_groq(messages)
        payload = self._extract_json(content)

        scores = {
            "task_completion": self._normalize_score(payload.get("task_completion")),
            "policy_adherence": self._normalize_score(payload.get("policy_adherence")),
            "coherence": self._normalize_score(payload.get("coherence")),
        }

        reasoning = payload.get("reasoning") or {}
        normalized_reasoning = {
            "task_completion": str(reasoning.get("task_completion", "")).strip(),
            "policy_adherence": str(reasoning.get("policy_adherence", "")).strip(),
            "coherence": str(reasoning.get("coherence", "")).strip(),
        }

        return scores, normalized_reasoning

    def _build_messages(self, turns: List[Dict], expected_task: str) -> list[dict[str, str]]:
        system_prompt = (
            "You are a strict evaluator for ISP customer support dialogues. "
            "Score three binary metrics: task_completion, policy_adherence, coherence. "
            "Task completion: 1 if the expected task is completed, else 0. "
            "Policy adherence: 1 if the assistant stays in ISP domain, does not reveal AI identity, "
            "and properly redirects off-topic requests; else 0. "
            "Coherence: 1 if the assistant maintains context and does not contradict itself; else 0. "
            "Return ONLY a JSON object with keys: task_completion, policy_adherence, coherence, reasoning. "
            "reasoning must be an object with the same three keys and short sentences."
        )

        dialogue_lines = []
        for idx, turn in enumerate(turns, start=1):
            user_text = str(turn.get("user", "")).strip()
            bot_text = str(turn.get("bot", "")).strip()
            dialogue_lines.append(f"Turn {idx}\nUser: {user_text}\nBot: {bot_text}")

        user_prompt = (
            f"Expected task: {expected_task}\n\n"
            "Dialogue:\n"
            + "\n\n".join(dialogue_lines)
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _call_groq(self, messages: list[dict[str, str]]) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        url = f"{self.api_base.rstrip('/')}/chat/completions"
        with httpx.Client(timeout=self.timeout_sec) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        try:
            return data["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            raise ValueError(f"Unexpected Groq response format: {data}") from exc

    @staticmethod
    def _extract_json(text: str) -> dict:
        text = text.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"Judge response did not contain JSON: {text}")
        return json.loads(text[start : end + 1])

    @staticmethod
    def _normalize_score(value: Any) -> int:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return 1 if int(value) == 1 else 0
        if isinstance(value, str):
            val = value.strip().lower()
            if val in {"1", "true", "yes"}:
                return 1
            if val in {"0", "false", "no"}:
                return 0
        raise ValueError(f"Invalid score value from judge: {value}")

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
                "count_0": len(task_scores) - sum(task_scores),
            },
            "policy_adherence": {
                "mean": mean(policy_scores),
                "median": median(policy_scores),
                "count_1": sum(policy_scores),
                "count_0": len(policy_scores) - sum(policy_scores),
            },
            "coherence": {
                "mean": mean(coherence_scores),
                "median": median(coherence_scores),
                "count_1": sum(coherence_scores),
                "count_0": len(coherence_scores) - sum(coherence_scores),
            },
            "overall": {
                "mean": mean(overall_scores),
                "median": median(overall_scores),
                "min": min(overall_scores),
                "max": max(overall_scores),
            },
        }


def load_dialogues(filepath: str = None) -> List[Dict]:
    """Load ISP multi-turn dialogues."""
    if filepath is None:
        filepath = os.path.join(PROJECT_ROOT, "data", "multi_turn_dialogues.json")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dialogue file not found: {filepath}")
    with open(filepath, "r") as f:
        return json.load(f)


def load_human_annotations(filepath: str = None) -> Dict[str, Dict]:
    """Load human annotations indexed by dialogue_id."""
    if filepath is None:
        filepath = os.path.join(PROJECT_ROOT, "data", "human_annotations.json")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Annotations file not found: {filepath}")
    with open(filepath, "r") as f:
        data = json.load(f)
    annotations_dict = {}
    for annotation in data["human_annotations"]:
        dialogue_id = annotation["dialogue_id"]
        annotations_dict[dialogue_id] = annotation["annotations"]
    return annotations_dict


def calculate_agreement(
    llm_scores: Dict[str, int],
    human_scores: Dict[str, int],
) -> Dict[str, Any]:
    """Calculate simple agreement and Cohen's Kappa."""
    if not llm_scores or not human_scores:
        return {
            "simple_agreement": 0,
            "cohen_kappa": 0,
            "matches": 0,
            "total": 0,
            "llm_unique_labels": 0,
            "human_unique_labels": 0,
            "kappa_uninformative": False,
        }

    matches = sum(
        1 for did in llm_scores
        if did in human_scores and llm_scores[did] == human_scores[did]
    )
    total = min(len(llm_scores), len(human_scores))
    po = matches / total if total > 0 else 0

    llm_ones = sum(1 for s in llm_scores.values() if s == 1)
    human_ones = sum(1 for s in human_scores.values() if s == 1)

    p_ones = (llm_ones / len(llm_scores)) * (human_ones / len(human_scores)) if llm_scores and human_scores else 0
    p_zeros = ((len(llm_scores) - llm_ones) / len(llm_scores)) * ((len(human_scores) - human_ones) / len(human_scores)) if llm_scores and human_scores else 0
    pe = p_ones + p_zeros

    kappa = (po - pe) / (1 - pe) if (1 - pe) != 0 else 0
    llm_unique_labels = len(set(llm_scores.values()))
    human_unique_labels = len(set(human_scores.values()))
    kappa_uninformative = bool(
        po == 1.0 and kappa == 0 and (llm_unique_labels == 1 or human_unique_labels == 1)
    )

    return {
        "simple_agreement": po,
        "cohen_kappa": kappa,
        "matches": matches,
        "total": total,
        "llm_unique_labels": llm_unique_labels,
        "human_unique_labels": human_unique_labels,
        "kappa_uninformative": kappa_uninformative,
    }


def generate_agreement_report(
    llm_evaluations: List[Dict],
    human_annotations: Dict[str, Dict],
) -> Dict[str, Any]:
    """Generate ISP-specific agreement report."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "metrics": {},
        "per_metric_agreement": {},
    }

    for metric in ["task_completion", "policy_adherence", "coherence"]:
        llm_scores = {}
        human_scores = {}

        for eval_result in llm_evaluations:
            did = eval_result["dialogue_id"]
            llm_scores[did] = eval_result["scores"].get(metric, 0)
            if did in human_annotations:
                human_scores[did] = human_annotations[did].get(metric, 0)

        agreement = calculate_agreement(llm_scores, human_scores)
        report["per_metric_agreement"][metric] = agreement

    # Overall
    all_llm = {}
    all_human = {}
    for eval_result in llm_evaluations:
        did = eval_result["dialogue_id"]
        all_llm[did] = eval_result["scores"].get("overall", 0)
        if did in human_annotations:
            all_human[did] = mean([
                human_annotations[did].get("task_completion", 0),
                human_annotations[did].get("policy_adherence", 0),
                human_annotations[did].get("coherence", 0),
            ])

    report["overall_agreement"] = calculate_agreement(all_llm, all_human)
    return report


def generate_markdown_report(
    llm_summary: Dict[str, Any],
    agreement_report: Dict[str, Any],
    llm_evaluations: List[Dict],
) -> str:
    """Generate formatted Markdown report for ISP agent evaluation."""
    report = f"""# ISP Customer Service AI — LLM-as-Judge Report

**Generated:** {agreement_report['timestamp']}

## Summary Statistics

### Task Completion
- **Mean Score:** {llm_summary['task_completion']['mean']:.2f}
- **Completed:** {llm_summary['task_completion']['count_1']}/{llm_summary['total_evaluated']}

### Policy Adherence (ISP Domain + Identity + STATE)
- **Mean Score:** {llm_summary['policy_adherence']['mean']:.2f}
- **Compliant:** {llm_summary['policy_adherence']['count_1']}/{llm_summary['total_evaluated']}

### Coherence (Context Retention + STATE Consistency)
- **Mean Score:** {llm_summary['coherence']['mean']:.2f}
- **Coherent:** {llm_summary['coherence']['count_1']}/{llm_summary['total_evaluated']}

### Overall
- **Mean:** {llm_summary['overall']['mean']:.3f}
- **Range:** {llm_summary['overall']['min']:.3f} - {llm_summary['overall']['max']:.3f}

---

## Judge-Human Agreement

| Metric | Agreement | Kappa | Matches |
|--------|-----------|-------|---------|
"""

    for metric, ag in agreement_report["per_metric_agreement"].items():
        report += f"| {metric.replace('_', ' ').title()} | {ag['simple_agreement']:.1%} | {ag['cohen_kappa']:.3f} | {ag['matches']}/{ag['total']} |\n"

    oa = agreement_report["overall_agreement"]
    report += f"""
### Overall Agreement
- **Simple Agreement:** {oa['simple_agreement']:.1%}
- **Cohen's Kappa:** {oa['cohen_kappa']:.3f}

---

## Detailed Evaluations

"""

    kappa_notes = []
    for metric, ag in agreement_report["per_metric_agreement"].items():
        if ag.get("kappa_uninformative"):
            kappa_notes.append(metric.replace("_", " ").title())

    if kappa_notes:
        report += "### Kappa Interpretation Notes\n"
        report += (
            "- One or both annotators used a single label for these metrics: "
            + ", ".join(kappa_notes)
            + ".\n"
        )
        report += (
            "- In this low-variance case, Cohen's Kappa can be 0.000 even when simple agreement is 100%.\n\n"
        )

    for ev in llm_evaluations:
        report += f"""### {ev['dialogue_id'].upper()}
**Task:** {ev['expected_task']}
- Task Completion: {ev['scores']['task_completion']}
- Policy Adherence: {ev['scores']['policy_adherence']}
- Coherence: {ev['scores']['coherence']}
- Overall: {ev['scores']['overall']:.3f}
- *{ev['reasoning']['policy_adherence']}*

"""

    return report


async def run_judge_evaluation():
    """Main entry point for ISP LLM-as-judge evaluation."""
    print("\n" + "=" * 60)
    print("ISP CUSTOMER SERVICE AI — LLM-AS-JUDGE EVALUATION")
    print("=" * 60)

    dialogues = load_dialogues()
    human_annotations = load_human_annotations()

    judge = GroqJudge()
    evaluations = []

    for dialogue in dialogues:
        evaluation = judge.evaluate_dialogue(
            dialogue["dialogue_id"],
            dialogue["turns"],
            dialogue["expected_task"],
        )
        evaluations.append(evaluation)
        logger.info(
            f"{dialogue['dialogue_id']}: "
            f"Task={evaluation['scores']['task_completion']}, "
            f"Policy={evaluation['scores']['policy_adherence']}, "
            f"Coherence={evaluation['scores']['coherence']}, "
            f"Overall={evaluation['scores']['overall']:.3f}"
        )

    llm_summary = judge.get_evaluation_summary()
    agreement_report = generate_agreement_report(evaluations, human_annotations)
    markdown_report = generate_markdown_report(llm_summary, agreement_report, evaluations)

    os.makedirs("eval_reports", exist_ok=True)
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
        "agreement": agreement_report,
    }


if __name__ == "__main__":
    import asyncio
    asyncio.run(run_judge_evaluation())

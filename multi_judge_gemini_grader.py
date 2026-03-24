"""
=======================================================================
  Multi-Judging LLM System for Exam Answer Evaluation
=======================================================================
  Course   : Generative AI and Large Language Models
  Project  : Reliability-Aware Multi-Judge Grading Architecture

  Architecture (from diagram):
    Input Dataset → Preprocessing → Rubric Generator (LLM)
    → Multi-Judge Prompt Generator
    → [Strict Judge | Moderate Judge | Lenient Judge]
    → Self-Consistency Engine (Multiple Runs / Averaging)
    → Score Aggregation Module (Weighted Avg + Variance)
    → [Feedback Generator] + [Evaluation Module (Human vs LLM Metrics)]

  Dataset columns expected:
    questions, model_answer, student_answer, total_marks, teacher_marks

  Prerequisites:
    pip install google-genai pandas scikit-learn scipy matplotlib seaborn
=======================================================================
"""

import os
import json
import time
import statistics
import numpy as np
import pandas as pd
from typing import Optional
from google import genai
from google.genai import types

# ─────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "your-api-key-here")
MODEL_NAME     = "gemini-2.0-flash-lite"
MAX_TOKENS        = 1000

# Self-Consistency: number of runs per judge
SELF_CONSISTENCY_RUNS = 3      # increase to 5 for stronger averaging
TEMPERATURE           = 0.7    # >0 for stochastic sampling

# Score Aggregation Weights  (must sum to 1.0)
WEIGHTS = {
    "moderate": 0.50,
    "strict"  : 0.25,
    "lenient" : 0.25,
}

# Dataset path  (replace with your actual dataset path or use the mec.csv)
DATASET_PATH = "mec.csv"       # expects: questions, model_answer, student_answer, total_marks, teacher_marks


# ─────────────────────────────────────────────────────────────────────
# 0. RATE-LIMIT RETRY HELPER
# ─────────────────────────────────────────────────────────────────────

def call_api(model, prompt: str, retries: int = 5) -> str:
    """Call Gemini API with automatic retry on rate-limit (429) errors."""
    for attempt in range(retries):
        try:
            response = model.models.generate_content(model=MODEL_NAME, contents=prompt)
            return response.text.strip()
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                wait = 30 * (attempt + 1)   # 30s, 60s, 90s ...
                print(f"  [Rate limit hit, waiting {wait}s before retry {attempt+1}/{retries}]")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"API failed after {retries} retries.")


# ─────────────────────────────────────────────────────────────────────
# 1. PREPROCESSING MODULE
# ─────────────────────────────────────────────────────────────────────

def preprocess(question: str, model_answer: str, student_answer: str) -> dict:
    """Clean and format inputs for LLM consumption."""
    def clean(text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        text = text.strip()
        text = " ".join(text.split())          # collapse whitespace
        return text[:2000]                     # truncate very long answers

    return {
        "question"      : clean(question),
        "model_answer"  : clean(model_answer),
        "student_answer": clean(student_answer),
    }


# ─────────────────────────────────────────────────────────────────────
# 2. RUBRIC GENERATOR (LLM)
# ─────────────────────────────────────────────────────────────────────

def generate_rubric(model, data: dict, total_marks: int) -> dict:
    """
    Extract key concepts from the model answer and assign marks.
    Returns: {"rubric": [{"concept": ..., "marks": ...}, ...], "raw": ...}
    """
    prompt = f"""You are an expert examiner. Analyze the model answer and create a clear grading rubric.

Question: {data['question']}
Model Answer: {data['model_answer']}
Total Marks: {total_marks}

Extract the KEY CONCEPTS from the model answer and assign marks to each.
Marks must sum to {total_marks}.

Respond ONLY in this JSON format (no extra text):
{{
  "rubric": [
    {{"concept": "...", "marks": <integer>}},
    {{"concept": "...", "marks": <integer>}}
  ]
}}"""

    raw = call_api(model, prompt)

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: single-concept rubric
        parsed = {"rubric": [{"concept": "Overall answer quality", "marks": total_marks}]}

    return {"rubric": parsed.get("rubric", []), "raw": raw}


def format_rubric(rubric_items: list) -> str:
    """Format rubric list into readable string for prompts."""
    lines = []
    for item in rubric_items:
        lines.append(f"  - {item['concept']} ({item['marks']} marks)")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# 3. MULTI-JUDGE PROMPT GENERATOR
# ─────────────────────────────────────────────────────────────────────

def build_judge_prompt(persona: str, data: dict, rubric_items: list, total_marks: int) -> str:
    """
    Build evaluation prompt for a given judge persona.
    persona: "strict" | "moderate" | "lenient"
    """
    rubric_str = format_rubric(rubric_items)

    persona_instructions = {
        "strict": (
            "You are a STRICT examiner. Penalize any missing keywords, incomplete explanations, "
            "or vague language. Award marks ONLY when concepts are precisely and fully addressed. "
            "Deduct marks for any conceptual inaccuracy."
        ),
        "moderate": (
            "You are a MODERATE examiner. Allow reasonable paraphrasing and minor omissions. "
            "Award marks when the core idea is correct even if not stated verbatim. "
            "Balance strictness with fairness."
        ),
        "lenient": (
            "You are a LENIENT examiner. Focus on whether the student grasped the core idea. "
            "Award marks generously if the overall meaning is correct, even with imprecise language "
            "or partial explanations."
        ),
    }

    instruction = persona_instructions[persona]

    prompt = f"""
{instruction}

Evaluate the following student answer using the rubric below.

Question: {data['question']}
Model Answer: {data['model_answer']}
Rubric (Total: {total_marks} marks):
{rubric_str}

Student Answer: {data['student_answer']}

Respond ONLY in this JSON format (no extra text):
{{
  "score": <integer between 0 and {total_marks}>,
  "covered_concepts": ["concept1", "concept2"],
  "missing_concepts": ["concept3"],
  "feedback": "2-3 sentence evaluation",
  "reasoning": "brief explanation of score"
}}"""
    return prompt.strip()


# ─────────────────────────────────────────────────────────────────────
# 4. LLM EVALUATION LAYER (Single Judge, Single Run)
# ─────────────────────────────────────────────────────────────────────

def run_single_judge(
    model,
    persona   : str,
    data      : dict,
    rubric    : list,
    total_marks: int,
) -> dict:
    """Call the LLM once as a specific judge persona. Returns parsed result."""
    prompt = build_judge_prompt(persona, data, rubric, total_marks)

    raw = call_api(model, prompt)

    # Strip markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        result = json.loads(raw)
        # Clamp score to valid range
        result["score"] = max(0, min(int(result.get("score", 0)), total_marks))
    except (json.JSONDecodeError, ValueError):
        result = {
            "score"            : 0,
            "covered_concepts" : [],
            "missing_concepts" : [],
            "feedback"         : raw[:300],
            "reasoning"        : "Parse error",
        }

    result["persona"] = persona
    return result


# ─────────────────────────────────────────────────────────────────────
# 5. SELF-CONSISTENCY ENGINE
# ─────────────────────────────────────────────────────────────────────

def run_judge_with_consistency(
    model,
    persona     : str,
    data        : dict,
    rubric      : list,
    total_marks : int,
    runs        : int = SELF_CONSISTENCY_RUNS,
) -> dict:
    """
    Run the same judge `runs` times and average the score.
    Returns aggregated result with mean score and std deviation.
    """
    all_scores    = []
    all_results   = []
    all_covered   = []
    all_missing   = []
    all_feedbacks = []

    for i in range(runs):
        result = run_single_judge(model, persona, data, rubric, total_marks)
        all_scores.append(result["score"])
        all_results.append(result)
        all_covered.extend(result.get("covered_concepts", []))
        all_missing.extend(result.get("missing_concepts", []))
        all_feedbacks.append(result.get("feedback", ""))
        time.sleep(0.5)   # gentle rate-limit buffer

    avg_score = round(statistics.mean(all_scores), 2)
    std_score = round(statistics.stdev(all_scores) if len(all_scores) > 1 else 0.0, 2)

    # Use feedback from the run closest to the mean
    best_run = min(all_results, key=lambda r: abs(r["score"] - avg_score))

    return {
        "persona"          : persona,
        "avg_score"        : avg_score,
        "std_dev"          : std_score,
        "all_scores"       : all_scores,
        "covered_concepts" : list(set(all_covered)),
        "missing_concepts" : list(set(all_missing)),
        "feedback"         : best_run.get("feedback", ""),
        "reasoning"        : best_run.get("reasoning", ""),
    }


# ─────────────────────────────────────────────────────────────────────
# 6. SCORE AGGREGATION MODULE
# ─────────────────────────────────────────────────────────────────────

def aggregate_scores(judge_results: dict, total_marks: int) -> dict:
    """
    Weighted aggregation of judge scores.
    Returns final score, variance, disagreement, and confidence.
    """
    strict_score   = judge_results["strict"]["avg_score"]
    moderate_score = judge_results["moderate"]["avg_score"]
    lenient_score  = judge_results["lenient"]["avg_score"]

    final_score = round(
        WEIGHTS["moderate"] * moderate_score +
        WEIGHTS["strict"]   * strict_score   +
        WEIGHTS["lenient"]  * lenient_score,
        2
    )
    final_score = max(0, min(final_score, total_marks))

    scores          = [strict_score, moderate_score, lenient_score]
    inter_variance  = round(statistics.variance(scores) if len(scores) > 1 else 0.0, 4)
    disagreement    = round(max(scores) - min(scores), 2)

    # Confidence: high when judges agree, lower when they disagree
    max_possible_disagreement = total_marks
    confidence = round((1 - disagreement / max_possible_disagreement) * 100, 1) if max_possible_disagreement > 0 else 100.0

    return {
        "final_score"       : round(final_score),
        "final_score_float" : final_score,
        "strict_score"      : strict_score,
        "moderate_score"    : moderate_score,
        "lenient_score"     : lenient_score,
        "inter_variance"    : inter_variance,
        "disagreement"      : disagreement,
        "confidence_pct"    : confidence,
    }


# ─────────────────────────────────────────────────────────────────────
# 7. FEEDBACK GENERATOR
# ─────────────────────────────────────────────────────────────────────

def generate_feedback(model, judge_results: dict, agg: dict, data: dict) -> dict:
    """Synthesize student-friendly feedback from all judge outputs."""
    covered = list(set(
        judge_results["moderate"]["covered_concepts"] +
        judge_results["strict"]["covered_concepts"]
    ))
    missing = list(set(
        judge_results["strict"]["missing_concepts"] +
        judge_results["moderate"]["missing_concepts"]
    ))

    prompt = f"""You are a helpful academic tutor. Based on the multi-judge evaluation below, 
write student-friendly feedback.

Question: {data['question']}
Student Answer: {data['student_answer']}
Model Answer: {data['model_answer']}

Judge Scores → Strict: {agg['strict_score']}, Moderate: {agg['moderate_score']}, Lenient: {agg['lenient_score']}
Final Score: {agg['final_score_float']} / {data.get('total_marks', 5)}
Covered Concepts: {covered}
Missing Concepts: {missing}

Respond ONLY in this JSON format:
{{
  "strengths": "What the student did well in 1-2 sentences",
  "missing_points": "What concepts were missing or incomplete in 1-2 sentences",
  "improvement": "Specific advice on how to improve the answer in 1-2 sentences"
}}"""

    raw = call_api(model, prompt)

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        feedback = json.loads(raw)
    except json.JSONDecodeError:
        feedback = {
            "strengths"    : "Could not parse feedback.",
            "missing_points": "",
            "improvement"  : "",
        }
    return feedback


# ─────────────────────────────────────────────────────────────────────
# 8. BASELINE SINGLE-LLM EVALUATOR (for comparison)
# ─────────────────────────────────────────────────────────────────────

def run_baseline(model, data: dict, total_marks: int) -> int:
    """Single LLM grader with no multi-judge logic. Used for comparison."""
    prompt = f"""You are an examiner. Score this student answer from 0 to {total_marks}.

Question: {data['question']}
Model Answer: {data['model_answer']}
Student Answer: {data['student_answer']}

Reply ONLY with a single integer score between 0 and {total_marks}."""

    raw = call_api(model, prompt)
    try:
        score = int("".join(filter(str.isdigit, raw.split()[0])))
        return max(0, min(score, total_marks))
    except (ValueError, IndexError):
        return 0


# ─────────────────────────────────────────────────────────────────────
# 9. EVALUATION MODULE (Human vs LLM Metrics)
# ─────────────────────────────────────────────────────────────────────

def compute_metrics(human_scores: list, llm_scores: list, baseline_scores: list) -> dict:
    """Compute MAE, Pearson correlation, and QWK for both systems."""
    from scipy.stats import pearsonr
    from sklearn.metrics import cohen_kappa_score, mean_absolute_error

    human    = np.array(human_scores, dtype=float)
    multi    = np.array(llm_scores, dtype=float)
    baseline = np.array(baseline_scores, dtype=float)

    def safe_pearson(a, b):
        if np.std(a) == 0 or np.std(b) == 0:
            return 0.0
        return round(pearsonr(a, b)[0], 4)

    def safe_qwk(a, b):
        try:
            return round(cohen_kappa_score(a.astype(int), b.astype(int), weights="quadratic"), 4)
        except Exception:
            return 0.0

    return {
        "multi_judge": {
            "MAE"     : round(mean_absolute_error(human, multi), 4),
            "Pearson" : safe_pearson(human, multi),
            "QWK"     : safe_qwk(human, multi),
        },
        "baseline": {
            "MAE"     : round(mean_absolute_error(human, baseline), 4),
            "Pearson" : safe_pearson(human, baseline),
            "QWK"     : safe_qwk(human, baseline),
        },
    }


# ─────────────────────────────────────────────────────────────────────
# 10. BIAS & STABILITY ANALYSIS
# ─────────────────────────────────────────────────────────────────────

def run_bias_test(model, data: dict, rubric: list, total_marks: int) -> dict:
    """
    Test stability by running the moderate judge on the original answer
    and a lightly rephrased version to check for unfair score variance.
    """
    # Rephrase student answer slightly
    rephrase_prompt = f"""Rephrase the following student answer in different words 
while keeping the exact same meaning. Keep it about the same length.

Original: {data['student_answer']}

Reply ONLY with the rephrased answer, no extra text."""

    rephrase_resp = model.models.generate_content(model=MODEL_NAME, contents=rephrase_prompt)
    rephrased = rephrase_resp.text.strip()

    data_original  = data.copy()
    data_rephrased = data.copy()
    data_rephrased["student_answer"] = rephrased

    score_original  = run_single_judge(model, "moderate", data_original,  rubric, total_marks)["score"]
    score_rephrased = run_single_judge(model, "moderate", data_rephrased, rubric, total_marks)["score"]

    bias_delta       = abs(score_original - score_rephrased)
    stability_score  = round((1 - bias_delta / total_marks) * 100, 1) if total_marks > 0 else 100.0

    return {
        "original_answer"  : data["student_answer"],
        "rephrased_answer" : rephrased,
        "score_original"   : score_original,
        "score_rephrased"  : score_rephrased,
        "bias_delta"       : bias_delta,
        "stability_score"  : stability_score,
    }


# ─────────────────────────────────────────────────────────────────────
# 11. FULL PIPELINE — evaluate one record
# ─────────────────────────────────────────────────────────────────────

def evaluate_one(
    model,
    question      : str,
    model_answer  : str,
    student_answer: str,
    total_marks   : int,
    teacher_score : Optional[int] = None,
    run_bias      : bool = False,
) -> dict:
    """
    Full multi-judge pipeline for one Q&A record.
    Returns a complete result dictionary.
    """
    print(f"\n  → Preprocessing...")
    data = preprocess(question, model_answer, student_answer)
    data["total_marks"] = total_marks

    print(f"  → Generating rubric...")
    rubric_result = generate_rubric(model, data, total_marks)
    rubric        = rubric_result["rubric"]

    print(f"  → Running Strict Judge  (×{SELF_CONSISTENCY_RUNS} runs)...")
    strict_result   = run_judge_with_consistency(model, "strict",   data, rubric, total_marks)

    print(f"  → Running Moderate Judge (×{SELF_CONSISTENCY_RUNS} runs)...")
    moderate_result = run_judge_with_consistency(model, "moderate", data, rubric, total_marks)

    print(f"  → Running Lenient Judge  (×{SELF_CONSISTENCY_RUNS} runs)...")
    lenient_result  = run_judge_with_consistency(model, "lenient",  data, rubric, total_marks)

    judge_results = {
        "strict"  : strict_result,
        "moderate": moderate_result,
        "lenient" : lenient_result,
    }

    print(f"  → Aggregating scores...")
    agg = aggregate_scores(judge_results, total_marks)

    print(f"  → Generating feedback...")
    feedback = generate_feedback(model, judge_results, agg, data)

    print(f"  → Running baseline...")
    baseline_score = run_baseline(model, data, total_marks)

    bias_result = None
    if run_bias:
        print(f"  → Bias & stability test...")
        bias_result = run_bias_test(model, data, rubric, total_marks)

    return {
        "question"      : question,
        "student_answer": student_answer,
        "total_marks"   : total_marks,
        "teacher_score" : teacher_score,
        "rubric"        : rubric,
        "judge_results" : judge_results,
        "aggregation"   : agg,
        "feedback"      : feedback,
        "baseline_score": baseline_score,
        "bias_analysis" : bias_result,
    }


# ─────────────────────────────────────────────────────────────────────
# 12. DISPLAY RESULTS
# ─────────────────────────────────────────────────────────────────────

def print_result(result: dict, idx: int):
    agg = result["aggregation"]
    fb  = result["feedback"]

    print(f"\n{'='*65}")
    print(f"  RECORD #{idx+1}")
    print(f"{'='*65}")
    print(f"  Question       : {result['question'][:80]}")
    print(f"  Student Answer : {result['student_answer'][:80]}")
    print(f"  Total Marks    : {result['total_marks']}")
    print(f"  Human Score    : {result.get('teacher_score', 'N/A')}")
    print(f"{'─'*65}")
    print(f"  Rubric Concepts:")
    for item in result.get("rubric", []):
        print(f"    • {item['concept']} — {item['marks']} marks")
    print(f"{'─'*65}")
    print(f"  Judge Scores:")
    print(f"    Strict   : {agg['strict_score']:>5}  (std: {result['judge_results']['strict']['std_dev']})")
    print(f"    Moderate : {agg['moderate_score']:>5}  (std: {result['judge_results']['moderate']['std_dev']})")
    print(f"    Lenient  : {agg['lenient_score']:>5}  (std: {result['judge_results']['lenient']['std_dev']})")
    print(f"{'─'*65}")
    print(f"  ✅ Final AI Score   : {agg['final_score_float']} → rounded {agg['final_score']}")
    print(f"  ⚡ Baseline Score   : {result['baseline_score']}")
    print(f"  📊 Inter-Variance   : {agg['inter_variance']}")
    print(f"  ⚖️  Disagreement     : {agg['disagreement']}")
    print(f"  🎯 Confidence       : {agg['confidence_pct']}%")
    print(f"{'─'*65}")
    print(f"  Feedback:")
    print(f"    Strengths    : {fb.get('strengths', '')}")
    print(f"    Missing      : {fb.get('missing_points', '')}")
    print(f"    Improvement  : {fb.get('improvement', '')}")
    if result.get("bias_analysis"):
        ba = result["bias_analysis"]
        print(f"{'─'*65}")
        print(f"  Bias Analysis:")
        print(f"    Original score  : {ba['score_original']}")
        print(f"    Rephrased score : {ba['score_rephrased']}")
        print(f"    Bias delta      : {ba['bias_delta']}")
        print(f"    Stability score : {ba['stability_score']}%")


def print_metrics(metrics: dict):
    print(f"\n{'='*65}")
    print(f"  EVALUATION METRICS (Human vs LLM)")
    print(f"{'='*65}")
    print(f"  {'System':<20} {'MAE':>8}  {'Pearson':>10}  {'QWK':>10}")
    print(f"  {'─'*55}")
    for system, m in metrics.items():
        print(f"  {system:<20} {m['MAE']:>8.4f}  {m['Pearson']:>10.4f}  {m['QWK']:>10.4f}")
    print()


# ─────────────────────────────────────────────────────────────────────
# 13. SAVE RESULTS
# ─────────────────────────────────────────────────────────────────────

def save_results(results: list, metrics: dict, output_path: str = "grading_results.json"):
    """Save full results to JSON for later analysis."""
    # Flatten results for CSV export
    rows = []
    for r in results:
        agg = r["aggregation"]
        rows.append({
            "question"       : r["question"],
            "student_answer" : r["student_answer"],
            "total_marks"    : r["total_marks"],
            "teacher_score"  : r.get("teacher_score"),
            "strict_score"   : agg["strict_score"],
            "moderate_score" : agg["moderate_score"],
            "lenient_score"  : agg["lenient_score"],
            "final_ai_score" : agg["final_score"],
            "baseline_score" : r["baseline_score"],
            "confidence_pct" : agg["confidence_pct"],
            "disagreement"   : agg["disagreement"],
            "inter_variance" : agg["inter_variance"],
        })

    df = pd.DataFrame(rows)
    csv_path = output_path.replace(".json", ".csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Results CSV saved to: {csv_path}")

    # Full JSON (includes rubrics, feedback, etc.)
    serializable_results = []
    for r in results:
        r_copy = r.copy()
        serializable_results.append(r_copy)

    with open(output_path, "w") as f:
        json.dump({"results": serializable_results, "metrics": metrics}, f, indent=2, default=str)
    print(f"✅ Full results JSON saved to: {output_path}")

    return df


# ─────────────────────────────────────────────────────────────────────
# 14. VISUALIZATION (optional, requires matplotlib + seaborn)
# ─────────────────────────────────────────────────────────────────────

def plot_results(df: pd.DataFrame, metrics: dict):
    """Generate comparison plots."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("⚠️  matplotlib/seaborn not installed. Skipping plots.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Multi-Judge LLM Grading System — Results Dashboard", fontsize=14)

    # Plot 1: Score comparison
    ax1 = axes[0]
    x = range(len(df))
    if "teacher_score" in df.columns and df["teacher_score"].notna().any():
        ax1.plot(x, df["teacher_score"],  "ko-", label="Human",    linewidth=2)
    ax1.plot(x, df["final_ai_score"],  "bs-", label="Multi-Judge", linewidth=2)
    ax1.plot(x, df["baseline_score"],  "r^--", label="Baseline",  linewidth=1.5)
    ax1.set_title("Score Comparison")
    ax1.set_xlabel("Sample")
    ax1.set_ylabel("Score")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Judge score distribution
    ax2 = axes[1]
    judge_data = pd.melt(
        df[["strict_score", "moderate_score", "lenient_score"]],
        var_name="Judge", value_name="Score"
    )
    sns.boxplot(data=judge_data, x="Judge", y="Score", ax=ax2, palette="Set2")
    ax2.set_title("Judge Score Distributions")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Metrics comparison bar chart
    ax3 = axes[2]
    metric_names   = ["MAE", "Pearson", "QWK"]
    multi_vals     = [metrics["multi_judge"][m] for m in metric_names]
    baseline_vals  = [metrics["baseline"][m]    for m in metric_names]
    x_pos = np.arange(len(metric_names))
    width = 0.35
    ax3.bar(x_pos - width/2, multi_vals,    width, label="Multi-Judge", color="steelblue")
    ax3.bar(x_pos + width/2, baseline_vals, width, label="Baseline",    color="salmon")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(metric_names)
    ax3.set_title("Metrics: Multi-Judge vs Baseline")
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig("grading_dashboard.png", dpi=150, bbox_inches="tight")
    print("✅ Dashboard saved to: grading_dashboard.png")
    plt.show()


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  Multi-Judging LLM System for Exam Answer Evaluation")
    print("=" * 65)

    # ── Initialize Gemini client ──────────────────────────────────────
    model = genai.Client(api_key=GEMINI_API_KEY)

    # ── Load dataset ─────────────────────────────────────────────────
    print(f"\n📂 Loading dataset: {DATASET_PATH}")
    df_input = pd.read_csv(DATASET_PATH)
    print(f"   Loaded {len(df_input)} records.")
    print(f"   Columns: {list(df_input.columns)}")

    # Limit to first N records for a quick run (set to None for full dataset)
    MAX_RECORDS = 5          # ← change to None to run all 37 records
    if MAX_RECORDS:
        df_input = df_input.head(MAX_RECORDS)
        print(f"   ⚠️  Running on first {MAX_RECORDS} records (set MAX_RECORDS=None for full run).")

    # ── Run pipeline on each record ───────────────────────────────────
    all_results     = []
    human_scores    = []
    llm_scores      = []
    baseline_scores = []

    # Enable bias test on first record only (slow — 2 extra LLM calls)
    RUN_BIAS_ON_FIRST = True

    for i, row in df_input.iterrows():
        print(f"\n[{i+1}/{len(df_input)}] Processing record...")
        result = evaluate_one(
            model          = model,
            question       = str(row["questions"]),
            model_answer   = str(row["model_answer"]),
            student_answer = str(row["student_answer"]),
            total_marks    = int(row["total_marks"]),
            teacher_score  = int(row["teacher_marks"]) if pd.notna(row.get("teacher_marks")) else None,
            run_bias       = (RUN_BIAS_ON_FIRST and i == 0),
        )
        all_results.append(result)
        print_result(result, i)

        # Collect for metrics
        if result.get("teacher_score") is not None:
            human_scores.append(result["teacher_score"])
            llm_scores.append(result["aggregation"]["final_score"])
            baseline_scores.append(result["baseline_score"])

    # ── Compute evaluation metrics ────────────────────────────────────
    metrics = {}
    if len(human_scores) >= 2:
        print("\n📊 Computing evaluation metrics...")
        metrics = compute_metrics(human_scores, llm_scores, baseline_scores)
        print_metrics(metrics)
    else:
        print("\n⚠️  Not enough human scores to compute metrics (need ≥2 records).")

    # ── Save results ──────────────────────────────────────────────────
    df_out = save_results(all_results, metrics)

    # ── Visualize ─────────────────────────────────────────────────────
    if len(df_out) >= 2:
        plot_results(df_out, metrics)

    print("\n✅ All done!")


if __name__ == "__main__":
    main()

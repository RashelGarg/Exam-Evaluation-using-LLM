# Exam Answer Evaluation using LLM

This project implements a **Reliability-Aware Multi-Judge Grading Architecture** to evaluate student exam answers using Large Language Models (LLMs) such as Claude 3.5 Sonnet.

## Overview

The grading pipeline uses a multi-judge strategy to ensure fairness and reduce AI bias. It processes student answers by first dynamically generating a rubric from the model answer, and then simulating three distinct examiner personas:

1. **Strict Judge**: Evaluates strictly, penalizing vague or incomplete conceptual explanations.
2. **Moderate Judge**: Balances strictness and fairness, allowing for reasonable paraphrasing.
3. **Lenient Judge**: Awards marks generously for grasping core ideas even if imprecise.

Scores from these judges are then aggregated using a weighted average. The system also runs self-consistency checks, generates student-friendly feedback, evaluates confidence boundaries, and optionally runs stability analysis (bias checks).

## Pipeline Workflow

1. **Preprocessing**: Input text is cleaned and truncated.
2. **Rubric Generator (LLM)**: Extracts key concepts from the model answer and pairs them with discrete marks.
3. **Evaluation**: Three judge personas evaluate the student answers independently over multiple runs (Self-Consistency Engine).
4. **Aggregation**: Scores are normalized and combined.
5. **Feedback Synthesis**: Strengths, missing points, and actionable improvements correspond to the judges' findings.
6. **Analytics**: Computes Mean Absolute Error (MAE), Pearson correlation, and Quadratic Weighted Kappa (QWK) against provided human scores (like `teacher_marks`).

## Requirements

The multi-judge grader requires Python 3.x and the following libraries:
- `anthropic`
- `pandas`
- `numpy`
- `scikit-learn`
- `scipy`
- `matplotlib`
- `seaborn`

You can install the requirements globally or within a virtual environment via:
```bash
pip install anthropic pandas scikit-learn scipy matplotlib seaborn
```

## How to Run

1. **API Key Setup**: 
   Ensure you have an active Anthropic API key. You must set this key in your environment before running the grader.
   
   *Windows (PowerShell):*
   ```powershell
   $env:ANTHROPIC_API_KEY="your-api-key-here"
   ```
   *macOS/Linux:*
   ```bash
   export ANTHROPIC_API_KEY="your-api-key-here"
   ```

2. **Dataset Setup**: 
   The script looks for a dataset structured as `mec.csv` by default. The dataset should contain the following columns:
   `questions`, `model_answer`, `student_answer`, `total_marks`, `teacher_marks`

3. **Execution**:
   Run the evaluation script from your terminal (enabling UTF-8 mode might be necessary to support terminal emojis):
   ```bash
   python -X utf8 "multi_judge_llm_grader .py"
   ```

## Output

After processing, the script outputs an interactive dashboard to the terminal containing final evaluated marks, human score comparisons, and specific student feedback. 

All detailed evaluations, rubrics, metrics, and comparisons are exported as `grading_results.json` and tabular structured data as `grading_results.csv`. If `matplotlib` is installed, it will automatically generate and save a visual `grading_dashboard.png`.

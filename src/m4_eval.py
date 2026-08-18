from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH, RAGAS_MAX_WORKERS
from src.llm import ragas_backend, provider


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation (4 metrics).

    Cần OPENAI_API_KEY (RAGAS dùng LLM làm judge) và Python 3.11+ (asyncio).
    Bọc try/except để pipeline vẫn chạy end-to-end khi thiếu key.
    """
    zeros = {"faithfulness": 0.0, "answer_relevancy": 0.0,
             "context_precision": 0.0, "context_recall": 0.0, "per_question": []}

    if provider() == "none":
        print("  ⚠️  Không có GOOGLE_API_KEY/OPENAI_API_KEY — bỏ qua RAGAS (scores = 0).")
        return zeros

    try:
        from ragas import evaluate
        from ragas.metrics import (faithfulness, answer_relevancy,
                                   context_precision, context_recall)
        from datasets import Dataset

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })

        # Judge LLM: Gemini nếu có GOOGLE_API_KEY, ngược lại default OpenAI của RAGAS.
        judge_llm, judge_emb = ragas_backend()
        kwargs = {}
        if judge_llm is not None:
            kwargs["llm"] = judge_llm
            kwargs["embeddings"] = judge_emb
        try:
            from ragas.run_config import RunConfig
            kwargs["run_config"] = RunConfig(max_workers=RAGAS_MAX_WORKERS, timeout=300)
        except Exception:
            pass

        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
                                            context_precision, context_recall], **kwargs)
        df = result.to_pandas()

        def _f(row, key):
            try:
                value = float(row[key])
            except (KeyError, TypeError, ValueError):
                return 0.0
            return 0.0 if value != value else value  # NaN -> 0.0

        per_question = [
            EvalResult(
                question=row["question"],
                answer=row["answer"],
                contexts=list(row["contexts"]),
                ground_truth=row["ground_truth"],
                faithfulness=_f(row, "faithfulness"),
                answer_relevancy=_f(row, "answer_relevancy"),
                context_precision=_f(row, "context_precision"),
                context_recall=_f(row, "context_recall"),
            )
            for _, row in df.iterrows()
        ]

        n = max(len(per_question), 1)
        return {
            "faithfulness": sum(r.faithfulness for r in per_question) / n,
            "answer_relevancy": sum(r.answer_relevancy for r in per_question) / n,
            "context_precision": sum(r.context_precision for r in per_question) / n,
            "context_recall": sum(r.context_recall for r in per_question) / n,
            "per_question": per_question,
        }
    except Exception as e:
        print(f"  RAGAS evaluation failed: {e}")
        return zeros


# Diagnostic Tree: metric yếu nhất -> nguyên nhân gốc -> cách sửa
DIAGNOSTIC_TREE = {
    "faithfulness": (
        "LLM hallucinating - câu trả lời chứa thông tin không có trong context",
        "Siết prompt ('CHỈ dùng context'), giảm temperature, thêm citation bắt buộc",
    ),
    "context_recall": (
        "Missing relevant chunks - retrieval bỏ sót thông tin cần thiết",
        "Tăng top_k, cải thiện chunking (hierarchical/structure), thêm BM25 vào hybrid",
    ),
    "context_precision": (
        "Too many irrelevant chunks - context bị nhiễu, chunk đúng xếp hạng thấp",
        "Thêm/siết reranking (cross-encoder), metadata filter theo version, giảm chunk size",
    ),
    "answer_relevancy": (
        "Answer doesn't match question - trả lời lạc đề hoặc quá chung chung",
        "Cải thiện prompt template, yêu cầu trả lời trực tiếp câu hỏi, thêm few-shot",
    ),
}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if not eval_results:
        return []

    scored = []
    for r in eval_results:
        metrics = {
            "faithfulness": r.faithfulness,
            "answer_relevancy": r.answer_relevancy,
            "context_precision": r.context_precision,
            "context_recall": r.context_recall,
        }
        avg = sum(metrics.values()) / len(metrics)
        worst_metric = min(metrics, key=lambda m: metrics[m])
        diagnosis, fix = DIAGNOSTIC_TREE[worst_metric]
        scored.append({
            "question": r.question,
            "avg_score": round(avg, 4),
            "worst_metric": worst_metric,
            "score": round(metrics[worst_metric], 4),
            "metrics": {k: round(v, 4) for k, v in metrics.items()},
            "diagnosis": diagnosis,
            "suggested_fix": fix,
            "answer": r.answer[:300],
            "ground_truth": r.ground_truth[:300],
        })

    scored.sort(key=lambda d: d["avg_score"])
    return scored[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")

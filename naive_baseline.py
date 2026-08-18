"""
Basic RAG Baseline — Chạy TRƯỚC để có scores so sánh.
=====================================================
Basic = paragraph chunking + dense-only search (không hybrid, không rerank, không enrichment).
Đây là RAG đã học ở buổi trước — hôm nay sẽ cải thiện từng bước.
"""

import sys, os, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.m1_chunking import load_documents, chunk_basic
from src.m2_search import DenseSearch
from src.m4_eval import load_test_set, evaluate_ragas, save_report
from src.llm import answer_from_context, provider
from config import NAIVE_COLLECTION


def main():
    print("=" * 60)
    print("BASIC RAG BASELINE")
    print("(paragraph chunking + dense-only, no rerank, no enrichment)")
    print("=" * 60)

    docs = load_documents()
    chunks = []
    for doc in docs:
        for c in chunk_basic(doc["text"], metadata=doc["metadata"]):
            chunks.append({"text": c.text, "metadata": c.metadata})
    print(f"  {len(chunks)} basic paragraph chunks")

    search = DenseSearch()
    search.index(chunks, collection=NAIVE_COLLECTION)

    test_set = load_test_set()
    questions, answers, all_contexts, ground_truths = [], [], [], []

    print(f"  LLM provider: {provider()}")

    # Pha 1: retrieve toàn bộ query (bge-m3 đang trong RAM)
    retrieved_contexts = []
    for i, item in enumerate(test_set):
        results = search.search(item["question"], top_k=3, collection=NAIVE_COLLECTION)
        retrieved_contexts.append([r.text for r in results])
        print(f"  [{i+1}/{len(test_set)}] retrieved: {item['question'][:45]}...", flush=True)

    # Giải phóng bge-m3 trước khi gọi LLM/RAGAS (máy lab chỉ còn ~2GB RAM trống)
    search.unload()

    # Pha 2: sinh câu trả lời từ context
    for i, item in enumerate(test_set):
        contexts = retrieved_contexts[i]
        answer = answer_from_context(item["question"], contexts)

        answers.append(answer)
        questions.append(item["question"])
        all_contexts.append(contexts if contexts else ["Không có context."])
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] answered: {item['question'][:45]}...", flush=True)

    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    print("\nBASIC BASELINE SCORES")
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        print(f"  {m}: {results.get(m, 0):.4f}")
    save_report(results, [], path="naive_baseline_report.json")
    print("\nDone! Now implement advanced modules and run: python main.py")


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Total: {time.time() - start:.1f}s")

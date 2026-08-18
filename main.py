"""
Lab 18: Production RAG Pipeline — Main Entry Point
===================================================
Chạy toàn bộ pipeline: naive baseline → production → so sánh → report.

Usage:
    python main.py
"""

import json
import os
import subprocess
import sys
import time

# Windows console mặc định cp1252 -> in tiếng Việt/emoji sẽ crash UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _run_step(script: str) -> None:
    """Chạy 1 bước trong process riêng.

    Baseline và production pipeline mỗi bên đều load bge-m3 (~1.5GB). Chạy chung
    1 process trên máy 8GB làm quá trình encode chết giữa chừng
    (Windows fatal exception: access violation). Process riêng → RAM được trả về OS
    hoàn toàn sau mỗi bước.
    """
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run([sys.executable, script], env=env)
    if result.returncode != 0:
        raise SystemExit(f"❌ {script} thất bại (exit code {result.returncode})")


def main():
    print("=" * 60)
    print("LAB 18: PRODUCTION RAG PIPELINE")
    print("=" * 60)
    start = time.time()

    os.makedirs("reports", exist_ok=True)

    # Step 1: Basic Baseline
    print("\n📌 STEP 1: Running Basic RAG Baseline...")
    print("-" * 40)
    _run_step("naive_baseline.py")

    # Step 2: Production Pipeline
    print("\n📌 STEP 2: Running Production Pipeline...")
    print("-" * 40)
    _run_step(os.path.join("src", "pipeline.py"))

    # Move reports to reports/
    for f in ["ragas_report.json", "naive_baseline_report.json", "latency_report.json"]:
        if os.path.exists(f):
            os.rename(f, f"reports/{f}")

    # Step 3: Comparison
    print("\n📌 STEP 3: Comparison")
    print("-" * 40)
    naive_path = "reports/naive_baseline_report.json"
    prod_path = "reports/ragas_report.json"

    if os.path.exists(naive_path) and os.path.exists(prod_path):
        with open(naive_path, encoding="utf-8") as f:
            naive = json.load(f)
        with open(prod_path, encoding="utf-8") as f:
            prod = json.load(f)

        print(f"\n{'Metric':<25} {'Basic':>8} {'Production':>12} {'Δ':>8}")
        print("-" * 55)
        for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            n = naive.get("aggregate", {}).get(m, 0)
            p = prod.get("aggregate", {}).get(m, 0)
            d = p - n
            status = "✓" if p >= 0.75 else " "
            print(f"{status} {m:<23} {n:>8.4f} {p:>12.4f} {d:>+8.4f}")

    elapsed = time.time() - start
    print(f"\n⏱️  Total time: {elapsed:.1f}s")
    print("\n📋 Next steps:")
    print("  1. Điền analysis/failure_analysis.md")
    print("  2. Điền analysis/group_report.md")
    print("  3. Viết analysis/reflections/reflection_[Tên].md")
    print("  4. Chạy: python check_lab.py")


if __name__ == "__main__":
    main()

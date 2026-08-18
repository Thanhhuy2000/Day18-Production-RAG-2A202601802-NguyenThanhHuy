# Report — Lab 18: Production RAG

**Thực hiện:** Nguyễn Thanh Huy (2A202601802) — bài **cá nhân**, tự implement toàn bộ 5 modules
**Ngày:** 19/08/2026
**Môi trường:** Windows 10 · Python 3.11.9 · 8GB RAM (CPU-only) · Qdrant (Docker) · LLM = Google Gemini

## Phân công & Hoàn thành

| Module | Hoàn thành | Tests pass |
|--------|-----------|-----------|
| M1: Chunking (semantic / hierarchical / structure-aware) | ☑ | 13/13 |
| M2: Hybrid Search (BM25 VN + Dense + RRF) | ☑ | 5/5 |
| M3: Reranking (CrossEncoder bge-reranker-v2-m3) | ☑ | 5/5 |
| M4: Evaluation (RAGAS 4 metrics + Diagnostic Tree) | ☑ | 4/4 |
| M5: Enrichment (combined single-call + 4 technique) | ☑ | 10/10 |
| **Tổng** | | **37/37** |

TODO còn lại trong `src/m*.py`: **0**

## Kết quả RAGAS

| Metric | Naive | Production | Δ |
|--------|-------|-----------|---|
| Faithfulness | 0.8500 | **0.8595** | +0.0095 |
| Answer Relevancy | 0.7013 | **0.7611** | +0.0598 |
| Context Precision | 0.8500 | **0.7667** | −0.0833 |
| Context Recall | 0.8250 | **0.8417** | +0.0167 |

Cả 4 metric ≥ 0.75; faithfulness ≥ 0.85. Điều kiện đo và hạn chế của cột Naive: xem
[failure_analysis.md](failure_analysis.md#điều-kiện-đo-để-số-liệu-đọc-đúng).

## A/B test chunking (26 documents)

| Strategy | Chunks | Avg len | Min | Max |
|----------|--------|---------|-----|-----|
| basic (baseline) | 51 | 410 | 273 | 565 |
| semantic (threshold 0.85) | 208 | 99 | 6 | 354 |
| hierarchical (2048/256) | 99 children + 11 parents | 210 | 55 | 256 |
| structure-aware | 106 | 197 | 87 | 789 |

Semantic với threshold 0.85 tách quá vụn (min 6 ký tự) vì văn bản chính sách gồm các câu quy định độc lập —
similarity giữa 2 câu liền kề thường dưới 0.85. Muốn nhóm theo ý phải hạ ngưỡng về ~0.5–0.6.

## Latency breakdown (20 queries, CPU-only)

| Bước | Tổng (ms) | Trung bình / query (ms) |
|------|-----------|-------------------------|
| M1 Chunking | 290 | — |
| M5 Enrichment (104 chunks, đã cache) | 16 | — |
| M2 Indexing (BM25 + Dense 104 chunks) | 90.147 | — |
| M2 Hybrid retrieval | 6.684 | **334** |
| M3 Cross-encoder rerank | 333.444 | **16.672** |
| LLM generation | 94.991 | **4.750** |
| M4 RAGAS (4 metrics × 20 câu) | 640.736 | — |

Nguồn: `reports/latency_report.json`. Lần chạy đầu (chưa cache) enrichment mất **268.900 ms** cho 104 chunks.

**Đọc bảng này:** cross-encoder rerank chiếm **16.7 giây/query** — đắt gấp 50× so với hybrid retrieval (334ms) và
gấp 3.5× so với LLM generation. Trên CPU, đây là lý do production thật phải rerank theo tầng (flashrank cho traffic
thường, cross-encoder chỉ cho câu hỏi khó) chứ không bật mặc định. Generation 4.75s/query gồm cả throttle 4.5s
(free tier 15 req/phút), không phải thời gian thực của model.

## Key Findings

1. **Biggest improvement — Answer Relevancy +0.0598 và Faithfulness giữ ở 0.86 với câu hỏi khó hơn.**
   Hybrid search (BM25 bù cho dense ở các truy vấn chứa số/tên chính sách) + rerank giúp context sạch hơn, LLM trả
   lời trực tiếp vào câu hỏi thay vì diễn giải lan man.

2. **Biggest challenge — môi trường, không phải thuật toán.** Ba lỗi tốn thời gian nhất đều là hạ tầng:
   - `Windows fatal exception: access violation` — mỗi test tạo một `CrossEncoderReranker()` mới → mỗi lần nạp thêm
     một bản model 1.5GB trên máy 8GB. Sửa bằng cache model ở mức module + `unload()`.
   - `TypeError: generate_content() got an unexpected keyword argument 'temperature'` — RAGAS 0.1.x truyền
     `temperature` vào từng call, `langchain-google-genai` (bản cho langchain 0.2) không nhận. Phải subclass
     `RagasCompatGemini` để đưa `temperature` vào `generation_config`.
   - Free tier Gemini giới hạn theo **từng model** và rất khác nhau: `gemini-3.6-flash` = 20 req/ngày,
     `gemini-3.5-flash` = 20 req/ngày, `gemini-3.5-flash-lite` = 500 req/ngày, 15 req/phút cho tất cả.

3. **Surprise finding — Context Precision GIẢM khi chunk nhỏ hơn, dù câu trả lời đúng.**
   Failure #4 trả lời đúng 100% nhưng `context_precision = 0.0`: child chunk 256 ký tự cắt bảng lương thành mảnh
   không có header, nên mảnh chứa "Junior" không được xếp lên top-1. Bài học: `context_precision` đo **chất lượng
   xếp hạng context**, không đo tính đúng của câu trả lời — và chunk nhỏ đánh đổi precision để lấy recall.

4. **Fallback im lặng làm sai lệch metric.** Lần chạy đầu, 13/20 câu bị 429 rate limit và rơi vào
   `answer = contexts[0]`. Khi đó faithfulness đạt **0.9929** (giả, vì answer trùng context nên luôn "grounded").
   Sau khi thêm throttle 4.5s/request và sinh lại câu trả lời thật, faithfulness về **0.8595** — con số trung thực.
   Bài học: điểm cao bất thường phải kiểm tra pipeline trước khi ăn mừng.

5. **3/5 failures có câu trả lời đúng.** RAGAS thấp không có nghĩa hệ thống sai — nó chỉ ra chỗ hệ thống *không
   chứng minh được* câu trả lời (faithfulness) hoặc xếp hạng context chưa tốt (precision). Phải đọc bottom-N kèm
   câu trả lời thật.

## Presentation Notes (5 phút)

1. **RAGAS scores (naive vs production):** cả 4 metric ≥ 0.75; answer_relevancy +0.06; context_precision −0.08 do
   đổi chiến lược chunking (giải thích bằng failure #4).
2. **Biggest win:** hybrid BM25 + RRF. RRF chỉ dùng thứ hạng nên không cần chuẩn hoá điểm BM25 (0–15) với cosine
   (0–1) — đây là điểm thắng thực dụng so với weighted-sum.
3. **Case study:** failure #1 (multi-hop "phép năm + lương") — Error Tree chỉ đúng tầng retrieval, không phải prompt;
   prompt strict đã hoạt động đúng khi từ chối bịa số.
4. **Next optimization (1 giờ):** parent expansion (dùng `parent_id` đã có) → query decomposition → prompt 3 bước
   cho câu hỏi numeric.

## Cách chạy lại

```bash
docker compose up -d                       # Qdrant
pip install -r requirements.txt
cp .env.example .env                       # điền GEMINI_API_KEY
pytest tests/ -v                           # 37 passed
python main.py                             # naive + production + so sánh
python src/pipeline.py --regen             # chỉ sinh lại answer + RAGAS (từ checkpoint)
python src/pipeline.py --eval-only         # chỉ chạy lại RAGAS (từ checkpoint)
python check_lab.py
```

Artifacts: `reports/ragas_report.json` · `reports/naive_baseline_report.json` · `reports/latency_report.json` ·
`pipeline_predictions.json` (checkpoint câu trả lời + context của 20 câu).

# Individual Reflection — Lab 18: Production RAG

**Tên:** Nguyễn Thanh Huy
**Mã:** 2A202601802
**Module phụ trách:** Toàn bộ M1 → M5 + pipeline (bài cá nhân)
**Môi trường:** Windows 10, Python 3.11.9, 8GB RAM, Qdrant qua Docker, LLM = Google Gemini (không có OpenAI key)

---

## Phần 1: Mapping bài giảng → code

| Lecture Concept | Module | Hàm cụ thể | Observation (số liệu thật từ corpus 26 docs) |
|----------------|--------|------------|---------------------------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Threshold 0.85 (config) tạo **208 chunks, avg 99 ký tự** vs basic **51 chunks, avg 410**. Ngưỡng 0.85 quá khắt khe với văn bản chính sách tiếng Việt: mỗi câu là một quy định độc lập nên similarity liên tiếp thường < 0.85 → gần như tách từng câu (min_len = 6 ký tự). Muốn nhóm theo ý phải hạ threshold về ~0.5-0.6. |
| Hierarchical (parent-child) | M1 | `chunk_hierarchical()` | 26 docs → **11 parents (2048 ký tự) + 99 children (avg 210, max 256)**. Child nhỏ → embedding "đặc" hơn, ít nhiễu; `parent_id` giữ đường dẫn để mở rộng context khi cần. Đây là strategy tôi dùng cho pipeline production. |
| Structure-aware chunking | M1 | `chunk_structure_aware()` | Regex `(^#{1,3}\s+.+$)` với `re.MULTILINE` tách theo header markdown → **106 chunks, avg 197**. Ưu điểm lớn nhất: mỗi chunk mang `metadata["section"]` = tên header, rất hữu ích để filter theo version tài liệu (`nghi_phep_nam_v2023` vs `v2024`). max_len 789 vì có section chứa bảng dài — đúng ý đồ "không cắt giữa bảng". |
| BM25 tiếng Việt | M2 | `segment_vietnamese()` + `BM25Search` | underthesea nối từ ghép bằng `_` ("nghỉ_phép"). Nếu giữ `_`, corpus có token `nghỉ_phép` còn query `"nghỉ phép"` tách thành 2 token → **BM25 score = 0**. Bắt buộc `replace("_", " ")` sau segment. Tôi thêm `.lower()` cho cả corpus và query để khớp không phân biệt hoa/thường. |
| BM25 + Dense fusion (RRF) | M2 | `reciprocal_rank_fusion()` | RRF chỉ dùng **thứ hạng**, nên không phải chuẩn hoá điểm giữa BM25 (không giới hạn, ~0-15) và cosine similarity (0-1) — đây là lý do RRF thắng weighted-sum trong thực tế. Công thức `1/(k + rank + 1)` với k=60 làm phẳng chênh lệch giữa top-1 và top-5, ưu tiên doc xuất hiện ở **cả hai** danh sách. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | `bge-reranker-v2-m3` cho điểm cặp (query, doc) thay vì so 2 vector độc lập → bắt được quan hệ ngữ nghĩa mà bi-encoder bỏ sót. Trade-off rất rõ trên CPU: xem bảng latency ở `reports/latency_report.json` (rerank là bước đắt nhất trong pipeline, ~20 cặp/query). Đổi lại top-20 → top-3 làm context precision tăng và giảm nhiễu cho LLM. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | 4 metric chia làm 2 nhóm: **retrieval** (context_precision, context_recall — đo hệ thống tìm kiếm) và **generation** (faithfulness, answer_relevancy — đo LLM). Nhờ tách nhóm này mà Diagnostic Tree mới hoạt động: metric yếu nhất chỉ thẳng ra tầng nào đang lỗi. |
| Diagnostic / Error Tree | M4 | `DIAGNOSTIC_TREE` + `failure_analysis()` | Map `worst_metric → (nguyên nhân gốc, cách sửa)`. Sort theo điểm trung bình 4 metric tăng dần → lấy bottom-5. Điểm hay: cùng một câu trả lời sai, nếu recall thấp thì sửa retrieval, còn nếu faithfulness thấp thì sửa prompt — hai hướng hoàn toàn khác nhau. |
| Contextual embeddings (Anthropic) | M5 | `contextual_prepend()` / `_enrich_single_call()` | Prepend 1 câu mô tả "chunk này nằm ở đâu, nói về gì" trước khi embed. Với chunk 256 ký tự bị cắt giữa tài liệu, câu context này bù lại thông tin đã mất khi tách chunk (tên tài liệu, chủ đề section) → giảm retrieval failure. |
| Enrichment cost optimization | M5 | `_enrich_single_call()` | Combined mode: 1 API call/chunk trả về **summary + 3 hypothesis questions + context line + metadata** thay vì 4 call riêng → giảm 75% số request. Cực kỳ quan trọng khi free tier Gemini chỉ cho 15 req/phút: 104 chunks × 1 call = 275s, còn 4 call/chunk sẽ mất ~18 phút. |
| HyQA (hypothesis questions) | M5 | `generate_hypothesis_questions()` | Sinh câu hỏi mà chunk trả lời được → bridge vocabulary gap giữa cách người dùng hỏi ("nghỉ cưới mấy ngày") và cách tài liệu viết ("nghỉ phép đặc biệt: kết hôn — 3 ngày"). |

---

## Phần 2: Khó khăn & cách giải quyết

### 2.1. `Windows fatal exception: access violation` khi chạy `pytest tests/`

**Lỗi chính xác:**
```
..................Windows fatal exception: access violation
Current thread ...: File "...\torch\storage.py", line 471 in __getitem__
  File "...\transformers\core_model_loading.py", line 1219 in _materialize_copy
```
18 test đầu pass rồi crash ở test M3.

**Debug:** Crash luôn xảy ra đúng lúc load model, nên tôi nghi RAM chứ không phải lỗi logic. Đo bằng `psutil`:
```
bge-m3 loaded: RSS=0.81GB avail=1.20GB
encoded:       RSS=1.55GB avail=0.57GB
OSError: The paging file is too small for this operation to complete. (os error 1455)
```
Máy 8GB, chỉ còn ~2GB trống. Nhìn lại `tests/test_m3.py`: **mỗi test tạo một `CrossEncoderReranker()` mới** → mỗi lần load thêm một bản copy model 1.5GB.

**Cách sửa:** cache model ở mức module (`_MODEL_CACHE` trong `m3_rerank.py`, `_ENCODER_CACHE` trong `m2_search.py`) để mọi instance dùng chung 1 bản, kèm hàm `unload()` để giải phóng chủ động. Kết quả: **37/37 tests pass**.

### 2.2. Pipeline chết giữa bước encode (exit code 139 = segfault)

Sau khi sửa (2.1), pipeline vẫn chết ở `Batches: 0%| | 0/26`. Nguyên nhân là đỉnh RAM: cùng lúc có underthesea (BM25) + thư viện Gemini + bge-m3.

**Ba thay đổi để sống được trên máy 8GB:**
1. `model.max_seq_length = 512` — bge-m3 mặc định 8192 token, activation khi encode theo batch cực lớn, trong khi chunk của lab chỉ ~256-400 ký tự.
2. `batch_size=4` khi encode (thay vì 16).
3. **Chạy pipeline theo pha** trong `evaluate_pipeline()`: retrieve toàn bộ 20 query (bge-m3 trong RAM) → `search.dense.unload()` → rerank toàn bộ (cross-encoder trong RAM) → `reranker.unload()` → generation + RAGAS (chỉ gọi API). Không bao giờ giữ 2 model cùng lúc.
4. `main.py` chạy naive baseline và production pipeline bằng **subprocess riêng** (`_run_step()`) để OS thu hồi hết RAM sau mỗi bước.

### 2.3. Không có OpenAI key → phải chuyển sang Gemini

Lab thiết kế cho `OPENAI_API_KEY`, tôi chỉ có Gemini key. Thay vì sửa rải rác, tôi tạo **`src/llm.py`** làm một lớp provider duy nhất (`provider()`, `chat()`, `answer_from_context()`, `ragas_backend()`), thứ tự ưu tiên Gemini → OpenAI → fallback extractive. M4, M5, pipeline, naive_baseline đều gọi qua lớp này nên đổi provider chỉ sửa 1 file.

### 2.4. `.env` không được đọc vì sai chữ hoa/thường

Tôi đặt `GEMINI_API_Key=...`, nhưng `os.getenv("GEMINI_API_KEY")` phân biệt hoa/thường → `provider()` trả về `"none"`, RAGAS im lặng trả về 0 hết. Bài học: khi metric = 0 tuyệt đối, kiểm tra config trước khi nghi ngờ model.

### 2.5. Gemini 3.x "thinking" ăn hết `max_output_tokens`

**Hiện tượng:** `chat(..., max_tokens=100)` trả về `"Determine the"` — câu trả lời bị cụt ngay từ đầu. `usage_metadata` cho thấy `candidates_token_count: 4` nhưng `total_token_count: 199` → phần chênh là thinking tokens, và chúng **tính chung** vào `max_output_tokens`.

**Sửa:** đặt sàn `GEMINI_MIN_OUTPUT_TOKENS = 2048` trong `src/llm.py`.

### 2.6. Model bị gỡ + quota free tier

- `gemini-2.0-flash`: `404 This model is no longer available. Please update your code to use models/gemini-3.6-flash`.
- `gemini-3.6-flash`: quota free tier chỉ **20 request/NGÀY** (`GenerateRequestsPerDayPerProjectPerModel-FreeTier, quota_value: 20`) — hết ngay sau 5 call.
- Đổi sang `gemini-3.5-flash-lite`: 15 req/phút, 500 req/ngày.

**Hai biện pháp tiết kiệm quota:**
1. **Cache enrichment** (`.enrich_cache.json`, key = sha1(model|source|text)): chạy lại pipeline không tốn thêm call nào cho 104 chunks.
2. **Tách judge model** (`GEMINI_JUDGE_MODEL=gemini-3.1-flash-lite`) khỏi generator: quota tính riêng theo từng model nên chia tải được, đồng thời tránh **self-preference bias** (LLM chấm điểm cao cho output của chính nó).
3. **Checkpoint predictions** (`pipeline_predictions.json` + cờ `--eval-only`): retrieval + rerank + generation mất ~15 phút, nếu RAGAS bị 429 giữa đường thì chạy lại chỉ phần RAGAS.

### 2.7. RAGAS 0.1.x không tương thích langchain-google-genai

**Lỗi chính xác:**
```
Exception raised in Job[3]: TypeError(GenerativeServiceClient.generate_content()
got an unexpected keyword argument 'temperature')
```
→ cả 4 metric = 0.0.

**Debug:** đọc `ragas/llms/base.py` thấy `LangchainLLMWrapper.generate_text()` gọi
`langchain_llm.generate_prompt(prompts=[prompt]*n, temperature=..., stop=..., callbacks=...)`.
`ChatGoogleGenerativeAI` (bản tương thích langchain 0.2) nhận `**kwargs` rồi truyền thẳng xuống google client → client không có tham số `temperature`.

**Sửa:** subclass `RagasCompatGemini` trong `src/llm.py`, override `_generate`/`_agenerate` để chuyển `temperature` vào `generation_config` (đúng chỗ của nó) và bỏ `n` (RAGAS đã tự nhân bản prompt). Sau khi sửa: cả 4 metric đều ra số thật.

### 2.8. `UnicodeEncodeError` khi in tiếng Việt trên Windows console

```
UnicodeEncodeError: 'charmap' codec can't encode characters in position 2-3
```
Console Windows mặc định cp1252. Đã thêm `sys.stdout.reconfigure(encoding="utf-8")` vào `main.py`.

### Kiến thức còn thiếu → cách bổ sung

| Thiếu | Cách bổ sung |
|-------|--------------|
| Cách RAGAS wrap LLM và gọi metric | Đọc trực tiếp source `ragas/llms/base.py`, `ragas/metrics/` trong `.venv` — nhanh hơn đọc docs |
| Quản lý bộ nhớ khi chạy transformer trên CPU | Đo bằng `psutil` ở từng bước thay vì đoán; hiểu `max_seq_length` ảnh hưởng activation memory |
| Quota model Gemini | `genai.list_models()` + đọc `quota_id`/`quota_value` trong message 429 để biết đang chạm limit/phút hay limit/ngày |

---

## Phần 3: Action Plan cho project cá nhân

## Project: Trợ lý tra cứu quy định nội bộ (Vietnamese policy Q&A)

### Hiện tại
- RAG pipeline: chunking cố định 500 ký tự → embed → dense search top-3 → LLM trả lời. Không rerank, không eval tự động.
- Known issues:
  1. Câu hỏi về số liệu ("phụ cấp bao nhiêu") thường lấy đúng tài liệu nhưng sai section.
  2. Tài liệu có nhiều version (v2023/v2024) → hệ thống trả lời theo bản cũ.
  3. Không có cách đo là hôm nay tốt hơn hay tệ hơn hôm qua.

### Plan áp dụng
1. [ ] **Chunking:** `chunk_structure_aware()` làm chính (tài liệu chính sách toàn markdown có header), kết hợp `parent_id` kiểu hierarchical để mở rộng context. Lý do: `metadata["section"]` giải quyết trực tiếp issue #1, và giữ được bảng số liệu không bị cắt.
2. [ ] **Search:** Hybrid BM25 + Dense + RRF. Tiếng Việt có nhiều từ khoá chính xác (số tiền, tên chính sách, mã văn bản) mà dense hay bỏ sót → BM25 bù đúng chỗ đó. Bắt buộc `segment_vietnamese()` có `replace("_", " ")`.
3. [ ] **Reranking:** Có, nhưng theo tầng: `flashrank` (nhẹ, <5ms) cho traffic thường, `bge-reranker-v2-m3` cho câu hỏi khó/ambiguous. Lý do: đo được cross-encoder là bước đắt nhất trên CPU, không thể bật mặc định cho mọi request.
4. [ ] **Evaluation:** RAGAS 4 metric làm regression test (chạy trên test set 20-30 câu mỗi lần thay đổi pipeline) + Diagnostic Tree để biết sửa tầng nào. Judge dùng model khác generator.
5. [ ] **Enrichment:** `_enrich_single_call()` combined mode. Ưu tiên `context` line (chống mất ngữ cảnh) và `metadata` (topic/category + **version** để filter tài liệu hết hiệu lực → giải quyết issue #2).

### Bổ sung riêng cho môi trường của tôi
- [ ] Version filter: extract `effective_date` vào metadata, luôn filter `is_current == true` trước khi search → chặn hẳn lỗi trả lời theo v2023.
- [ ] OCR cho PDF scan: `BCTC.pdf` và `Nghi_dinh_13-2023.pdf` bị bỏ qua vì không có text layer — cần Tesseract/PaddleOCR mới đưa được vào index.
- [ ] Cache + checkpoint như lab này (cache enrichment, checkpoint predictions) để dev nhanh và không đốt quota.

### Timeline
| Tuần | Việc |
|------|------|
| Tuần 1 | Dựng test set 30 câu có ground truth + chạy RAGAS baseline trên pipeline hiện tại (có số để so sánh) |
| Tuần 2 | Đổi sang structure-aware chunking + metadata version filter; đo lại RAGAS |
| Tuần 3 | Thêm BM25 + RRF hybrid search; đo context_recall trước/sau |
| Tuần 4 | Thêm reranking 2 tầng + đo latency; chốt ngưỡng bật cross-encoder |
| Tuần 5 | Enrichment combined mode cho toàn bộ corpus + OCR các PDF scan |
| Tuần 6 | Failure analysis bottom-10, sửa theo Diagnostic Tree, viết regression test vào CI |

---

## Phần 4: Tự đánh giá

| Tiêu chí | Tự chấm (1-5) | Ghi chú |
|----------|---------------|---------|
| Hiểu bài giảng | 4 | Hiểu rõ vì sao RRF không cần chuẩn hoá điểm và vì sao cross-encoder chính xác hơn bi-encoder |
| Code quality | 4 | Tách `src/llm.py` thành 1 lớp provider giúp đổi Gemini/OpenAI không phải sửa 4 module |
| Problem solving | 5 | Tự chẩn đoán và sửa được 3 lỗi môi trường khó: access violation (RAM), segfault khi encode, RAGAS-Gemini TypeError |
| Debug bằng số liệu | 5 | Dùng `psutil` đo RAM từng bước, đọc `quota_id` trong 429, đọc source RAGAS thay vì đoán |

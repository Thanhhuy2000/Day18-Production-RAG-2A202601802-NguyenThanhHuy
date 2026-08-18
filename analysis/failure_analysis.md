# Failure Analysis — Lab 18: Production RAG

**Người thực hiện:** Nguyễn Thanh Huy (2A202601802) — bài cá nhân, tự làm toàn bộ M1 → M5
**Test set:** 20 câu (`test_set.json`) · **Corpus:** 26 documents (24 .md + 1 PDF có text layer; 2 PDF scan bị bỏ qua vì chưa OCR)
**Pipeline:** M1 hierarchical chunking → M5 enrichment (combined 1 call/chunk) → M2 hybrid BM25+Dense+RRF (top-20) → M3 cross-encoder rerank (top-3) → LLM answer → M4 RAGAS

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.8500 | **0.8595** | +0.0095 |
| Answer Relevancy | 0.7013 | **0.7611** | +0.0598 |
| Context Precision | 0.8500 | **0.7667** | −0.0833 |
| Context Recall | 0.8250 | **0.8417** | +0.0167 |

Naive = paragraph chunking (57 chunks) + dense-only top-3, không rerank, không enrichment.
Production = hierarchical child chunks (104 chunks) + enrichment + hybrid + rerank.

### Điều kiện đo (để số liệu đọc đúng)

| | Naive Baseline | Production |
|---|---|---|
| Generator | `gemini-3.5-flash-lite` | `gemini-3.5-flash` |
| Judge (RAGAS) | `gemini-3.1-flash-lite` | `gemini-3.1-flash-lite` |
| Embeddings (RAGAS) | `gemini-embedding-001` | `gemini-embedding-001` |
| Câu bị fallback (`answer = contexts[0]`) | không xác định được (log đã bị xoay vòng) | **1/20** |

⚠️ **Hạn chế cần nêu rõ:** hai lần đo dùng generator khác nhau vì free tier Gemini giới hạn theo từng model
(`gemini-3.5-flash-lite` = 500 req/ngày, `gemini-3.5-flash` = 20 req/ngày) và cả hai đều đã hết quota trong ngày.
Lần đo naive baseline chạy trước khi tôi thêm throttle (`LLM_MIN_INTERVAL=4.5s`), nên một số câu có thể đã rơi vào
fallback → **faithfulness của naive có khả năng bị thổi lên** (khi `answer == context` thì answer luôn "grounded").
Để có bảng so sánh chặt chẽ, cần chạy lại `python naive_baseline.py` bằng đúng generator của production sau khi
quota reset. Riêng cột Production là số liệu sạch (đã có throttle, 19/20 câu là câu trả lời LLM thật).

### Ghi chú quan trọng về Context Precision giảm

Đây **không** phải lỗi của reranking mà là hệ quả của việc đổi chiến lược chunking:

- Naive: chunk 410 ký tự trung bình → mỗi chunk chứa cả section, dễ "trùng" với ground truth.
- Production: child chunk ≤ 256 ký tự → thông tin bị tách nhỏ, top-1 hay là mảnh chỉ chứa **một phần** đáp án
  (rõ nhất ở failure #4: câu trả lời đúng 100% nhưng `context_precision = 0.0`).

Hướng sửa đúng là **parent expansion**: retrieve bằng child (precision cao) rồi trả `parent` (2048 ký tự) cho LLM —
`parent_id` đã có trong metadata, chỉ chưa dùng ở bước tạo context.

---

## Bottom-5 Failures

### #1 — Multi-hop 2 tài liệu (avg 0.1250)
- **Question:** "Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?"
- **Expected:** 15 ngày cơ bản + 3 ngày thâm niên = 18 ngày phép; lương Senior (P3-P4) 20–35 triệu VNĐ/tháng.
- **Got:** "Không tìm thấy."
- **Metrics:** faithfulness 0.0 · answer_relevancy 0.0 · context_precision 0.0 · **context_recall 0.5**
- **Worst metric:** faithfulness 0.0 (nhưng nguyên nhân gốc nằm ở retrieval — xem Error Tree)
- **Error Tree:**
  1. Output sai? → Có, LLM từ chối trả lời.
  2. Context có đủ đáp án? → **Không.** recall 0.5 = chỉ lấy được tài liệu `nghi_phep_nam_v2024.md`, thiếu `bang_luong_2024.md`.
  3. Query OK? → Query là **multi-hop**: cần 2 tài liệu khác chủ đề trong 1 lần retrieve.
  4. Fix ở bước: **retrieval (query decomposition)**.
- **Root cause:** rerank top-20 → top-3 chọn 3 chunk *tốt nhất theo điểm*, và cả 3 đều đến từ tài liệu phép năm (điểm cao hơn hẳn). Không có cơ chế đảm bảo **đa dạng nguồn**. LLM theo prompt strict "chỉ dùng context" nên trả về "Không tìm thấy" — đúng hành vi, sai dữ liệu.
- **Suggested fix:** (a) query decomposition — tách thành "phép năm 9 năm thâm niên" + "lương Senior" rồi hợp context; (b) MMR / group-by-source khi chọn top-k để mỗi source chỉ chiếm tối đa 2 slot; (c) tăng `RERANK_TOP_K` 3 → 5 cho câu hỏi nhiều mệnh đề.

### #2 — Multi-hop trong cùng tài liệu, bảng bị cắt (avg 0.2083)
- **Question:** "Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?"
- **Expected:** 30 triệu thuộc khoảng 5–50 triệu → Giám đốc phòng ban (Director) phê duyệt; cần xác nhận cấu hình kỹ thuật từ phòng CNTT; kèm ≥3 báo giá.
- **Got:** "Không tìm thấy."
- **Metrics:** faithfulness 0.5 · **answer_relevancy 0.0** · context_precision 0.0 · context_recall 0.3333
- **Worst metric:** answer_relevancy 0.0
- **Error Tree:**
  1. Output sai? → Có.
  2. Context có đáp án? → **Một phần** (recall 0.33): lấy được đoạn "thiết bị CNTT cần xác nhận cấu hình", thiếu **bảng ngưỡng phê duyệt theo giá trị**.
  3. Query OK? → Query cần 2 mảnh của cùng file `mua_sam.md` nằm ở 2 section khác nhau.
  4. Fix ở bước: **chunking**.
- **Root cause:** child chunk 256 ký tự cắt bảng "ngưỡng giá trị → người phê duyệt" ra khỏi ngữ cảnh, dòng "5–50 triệu: Director" trở thành mảnh rời không match query.
- **Suggested fix:** dùng `chunk_structure_aware()` cho tài liệu có bảng (giữ nguyên khối bảng, `max_len` 789 chứng tỏ nó không cắt giữa bảng) + parent expansion để LLM nhận cả section.

### #3 — Numeric reasoning: thiếu bước tính (avg 0.7114)
- **Question:** "Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?"
- **Expected:** Hạn 15 ngày → quá hạn 5 ngày; phí 2%/tháng trên 15.000.000 = 300.000 VNĐ/tháng, pro-rata ≈ 50.000 VNĐ cho 5 ngày.
- **Got:** "Bị phạt **2%/tháng** trên số tiền chưa hoàn ứng (15.000.000 VNĐ)."
- **Metrics:** **faithfulness 0.5** · answer_relevancy 0.8456 · context_precision 1.0 · context_recall 0.5
- **Worst metric:** faithfulness 0.5
- **Error Tree:**
  1. Output sai? → Sai **một nửa**: đúng tỷ lệ (2%/tháng) nhưng không tính ra số tiền, cũng không nói quá hạn 5 ngày.
  2. Context có đáp án? → precision 1.0 → chunk đúng đã nằm top-1; recall 0.5 vì thiếu đoạn nói hạn 15 ngày.
  3. Query OK? → Có.
  4. Fix ở bước: **generation (prompt)** + retrieval recall.
- **Root cause:** prompt yêu cầu "trả lời ngắn gọn" → LLM dừng ở mức nêu quy định, không thực hiện phép tính pro-rata. Con số "15.000.000" trong câu trả lời lấy từ **câu hỏi** chứ không có trong context → judge coi là claim không được support ⇒ faithfulness 0.5.
- **Suggested fix:** với câu hỏi numeric, đổi prompt sang bắt buộc 3 bước "trích quy định → viết công thức → ra kết quả"; hoặc để LLM gọi tool tính toán. Không nên trộn số từ câu hỏi vào câu trả lời mà không nêu nguồn quy định.

### #4 — Câu trả lời đúng nhưng Context Precision = 0 (avg 0.7184)
- **Question:** "Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?"
- **Expected:** Junior cao nhất 20.000.000 → thử việc 85% × 20.000.000 = 17.000.000 VNĐ/tháng.
- **Got:** "Lương thử việc của nhân viên Junior mức cao nhất là 17.000.000 VNĐ/tháng (bằng 85% của mức lương cao nhất 20.000.000 VNĐ)." — **đúng hoàn toàn**.
- **Metrics:** faithfulness 1.0 · answer_relevancy 0.8738 · **context_precision 0.0** · context_recall 1.0
- **Worst metric:** context_precision 0.0
- **Error Tree:**
  1. Output sai? → **Không, đúng.**
  2. Context có đáp án? → Có (recall 1.0) nhưng **thứ hạng xấu**: chunk top-1 là mảnh bảng lương chứa các dòng Senior/Lead/Manager, dòng Junior và quy tắc 85% nằm ở chunk hạng dưới.
  3. Fix ở bước: **ranking / chunking**, không phải generation.
- **Root cause:** bảng markdown bị `chunk_hierarchical` cắt theo 256 ký tự → mỗi chunk là vài dòng bảng, mất header bảng. Cross-encoder khó xếp đúng mảnh chứa "Junior" lên đầu khi mảnh đó không có ngữ cảnh "bảng lương".
- **Suggested fix:** structure-aware chunking cho file bảng; nếu vẫn cắt thì **lặp lại header bảng** vào mỗi mảnh (table-header propagation). Đây là ví dụ rõ nhất cho thấy `context_precision` đo **chất lượng xếp hạng**, không đo tính đúng của câu trả lời.

### #5 — Suy luận nhiều bước bị tính là hallucination (avg 0.7834)
- **Question:** "Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa học. Phải hoàn trả bao nhiêu?"
- **Expected:** Cam kết làm việc ≥1 năm; nghỉ sau 8 tháng là trước hạn → hoàn trả 100% = 25.000.000 VNĐ.
- **Got:** "Phải hoàn trả **25.000.000 VNĐ** (100% chi phí đào tạo, do nghỉ việc trước thời hạn cam kết 1 năm)." — **đúng**.
- **Metrics:** **faithfulness 0.3333** · answer_relevancy 0.8002 · context_precision 1.0 · context_recall 1.0
- **Worst metric:** faithfulness 0.3333
- **Error Tree:**
  1. Output sai? → Không, kết luận đúng.
  2. Context có đáp án? → Có, đầy đủ (precision 1.0, recall 1.0).
  3. Vậy vì sao faithfulness thấp? → LLM **gộp 3 claim** ("25 triệu", "100%", "trước hạn cam kết 1 năm") nhưng chỉ nêu quy tắc chung; bước suy luận "8 tháng < 12 tháng" không xuất hiện tường minh ⇒ judge chấm 2/3 claim là không được support.
  4. Fix ở bước: **generation (prompt)**.
- **Root cause:** faithfulness của RAGAS chấm theo **từng câu claim**. Câu trả lời càng "gọn" mà gộp nhiều suy luận thì càng dễ mất điểm.
- **Suggested fix:** prompt yêu cầu trích dẫn quy định trước rồi mới kết luận ("Theo mục X: cam kết 1 năm. Nghỉ sau 8 tháng < 12 tháng ⇒ hoàn trả 100% = 25.000.000 VNĐ"). Cách này vừa tăng faithfulness vừa dễ kiểm chứng cho người dùng.

---

## Tổng hợp pattern (điều rút ra từ bottom-5)

| Pattern | Số ca | Tầng lỗi | Hành động ưu tiên |
|---------|-------|----------|-------------------|
| Multi-hop cần ≥2 nguồn / ≥2 section | #1, #2 | Retrieval | Query decomposition + đa dạng nguồn khi chọn top-k |
| Bảng markdown bị cắt mất header | #2, #4 | Chunking | Structure-aware chunking + propagate table header + parent expansion |
| Numeric reasoning không được thực hiện | #3 | Generation | Prompt 3 bước (trích quy định → công thức → kết quả) |
| Suy luận gộp bị chấm là không grounded | #5 | Generation | Bắt buộc citation từng claim |

**Nhận xét quan trọng:** 3/5 failures (#3, #4, #5) có câu trả lời **đúng hoặc gần đúng**. Nghĩa là điểm RAGAS thấp
không đồng nghĩa hệ thống trả lời sai — nó chỉ ra chỗ **hệ thống không chứng minh được** câu trả lời của mình
(faithfulness) hoặc **xếp hạng context chưa tối ưu** (precision). Đây là lý do phải đọc bottom-N kèm câu trả lời
thật, chứ không chỉ nhìn điểm tổng.

---

## Case Study (cho presentation)

**Question chọn phân tích:** "Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?" (failure #1 — điểm thấp nhất, avg 0.125)

**Error Tree walkthrough:**
1. **Output đúng?** → Không. LLM trả "Không tìm thấy."
2. **Context đúng?** → Chỉ một nửa: `context_recall = 0.5`. Có `nghi_phep_nam_v2024.md`, thiếu `bang_luong_2024.md`.
3. **Query rewrite OK?** → Chưa có bước này. Câu hỏi 2 mệnh đề (phép năm + lương) được đưa nguyên vào 1 lần retrieve; RRF + rerank đều tối ưu theo "điểm liên quan tổng thể" nên toàn bộ 3 slot context bị chủ đề mạnh hơn (phép năm) chiếm hết.
4. **Fix ở bước:** **retrieval** — cụ thể là *thiếu query decomposition*, không phải lỗi prompt hay chunking. Đáng chú ý: prompt strict đã hoạt động **đúng** (không bịa lương) — nếu prompt lỏng, LLM có thể bịa ra con số và ta sẽ có một lỗi khó phát hiện hơn nhiều.

**Nếu có thêm 1 giờ, sẽ optimize:**
1. **Parent expansion** (~15 phút): retrieve child, trả `parent` qua `parent_id` đã có trong metadata. Kỳ vọng cải thiện trực tiếp `context_precision` (đang là metric yếu nhất, 0.7667) và cứu được #2, #4.
2. **Query decomposition cho câu hỏi nhiều mệnh đề** (~25 phút): 1 LLM call tách câu hỏi thành các sub-query, retrieve từng cái rồi hợp bằng RRF. Cứu #1, #2.
3. **Prompt cho numeric/multi-step** (~10 phút): bắt buộc "trích quy định → công thức → kết quả". Cứu #3, #5, tăng faithfulness.
4. **Group-by-source khi chọn top-k** (~10 phút): mỗi tài liệu tối đa 2 slot trong top-3 → tăng recall cho câu hỏi cross-document.

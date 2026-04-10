# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Văn Thạch - 2A202600237
**Nhóm:** 10
**Ngày:** 10/4/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**

High cosine similarity nghĩa là hai vector có hướng gần giống nhau, tức là hai câu có ý nghĩa tương tự nhau.
Công thức: 
$$cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|}$$

Nếu cos(theta) = 1 thì hai vector có hướng giống nhau, tức là hai câu có ý nghĩa tương tự nhau.


**Ví dụ HIGH similarity:**
- Sentence A: "Tôi thích con mèo"
- Sentence B: "Tôi thích con chó"
- Tại sao tương đồng: Cả hai câu đều nói về việc thích động vật, và "mèo" và "chó" là hai loại động vật có nhiều đặc điểm tương đồng.

**Ví dụ LOW similarity:**
- Sentence A: "Tôi thích con mèo"
- Sentence B: "Tôi không thích con mèo"
- Tại sao khác: Cả hai câu đều nói về việc thích con mèo, nhưng một câu nói thích và một câu nói không thích, nên hai câu có ý nghĩa khác nhau.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**

Cosine similarity được ưu tiên vì nó đo góc giữa các vector, nên phản ánh tốt mức độ giống nhau về ngữ nghĩa mà không bị ảnh hưởng bởi độ dài vector. Trong khi đó, Euclidean distance phụ thuộc vào độ lớn vector, nên kém ổn định hơn khi so sánh text embeddings. 



### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**


1. Tính bước nhảy (stride):
- Mỗi chunk sẽ bắt đầu với: `stride =chunk_size - overlap = 500 - 50 = 450`
2. Áp dụng vào công thức 
`chunks =  (số kí tự - overlap)/ stride = (10000 - 50) / 450 = 22.11`

Kết luận: Làm tròn lên ta có tất cả **23 chunks**.

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**

Áp dụng công thức trên với overlap = 100 thì chunk count sẽ tăng lên thành 25 chunks,overlap nhiều hơn sẽ giúp tăng khả năng tìm thấy thông tin liên quan hoặc cũng có thể gây thừa thông tin dẫn đến tốn nhiều tài nguyên và thời gian xử lý.
---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Đánh giá hiệu suất RAG trên dữ liệu đa dạng và chuyên biệt (Mixed Domain Knowledge Evaluation).

**Tại sao nhóm chọn domain này?**
Domain này bao gồm nhiều loại tài liệu thực tế với độ dài và cấu trúc khác nhau (từ wiki, tài liệu hướng dẫn đến bài viết blog cá nhân và văn bản pháp luật). Việc sử dụng domain hỗn hợp này cho phép mô phỏng sát nhất một hệ thống RAG thực tế, qua đó đánh giá toàn diện khả năng của Vector Store và kiểm thử tính hiệu quả của chiến lược Chunking.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | `00_bullet_kin.txt` | Enter the Gungeon Wiki | 10,654 | `category: 'Wiki'`, `id: '0'` |
| 2 | `01_underground_travel.txt` | DnD Campaign Notes | 2,750 | `category: 'Notebook'`, `id: '1'` |
| 3 | `02_rag_chatbot_blog.txt` | Medium/Tech Blog | 22,187 | `category: 'Blog'`, `id: '2'` |
| 4 | `03_llmware_docs.txt` | llmware GitHub/Docs | 20,421 | `category: 'Docs'`, `id: '3'` |
| 5 | `04_marimo_recipes.txt` | Marimo Recipes | 11,648 | `category: 'Docs'`, `id: '4'` |
| 6 | `05_impact_prioritization.txt` | Career Advice Blog | 14,278 | `category: 'Blog'`, `id: '5'` |



### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `id` / `doc_id` | `str` | `"0"`, `"1"` | Cho phép nhận diện một tài liệu cụ thể và dễ dàng dùng API để thao tác/xóa (`delete_document`). |
| `category` | `str` | `"Wiki"`, `"Blog"` | Lọc tập dữ liệu (vùng kiến thức) theo lĩnh vực hoặc dạng văn bản ở bước `search_with_filter`, nâng cao retrieval precision. |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 3 tài liệu đại diện:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| `00_bullet_kin.txt` | FixedSizeChunker (`fixed_size`) | 24 | 491.8 | Low |
| `00_bullet_kin.txt` | SentenceChunker (`by_sentences`) | 36 | 293.9 | Medium |
| `00_bullet_kin.txt` | RecursiveChunker (`recursive`) | 30 | 353.4 | High |
| `06_eu_ai_act.txt` | FixedSizeChunker (`fixed_size`) | 79 | 498.5 | Low |
| `06_eu_ai_act.txt` | SentenceChunker (`by_sentences`) | 77 | 458.6 | Medium |
| `06_eu_ai_act.txt` | RecursiveChunker (`recursive`) | 104 | 339.4 | High |
| `01_underground_travel.txt` | FixedSizeChunker (`fixed_size`) | 6 | 500.0 | Low |
| `01_underground_travel.txt` | SentenceChunker (`by_sentences`) | 11 | 248.4 | Medium |
| `01_underground_travel.txt` | RecursiveChunker (`recursive`) | 9 | 303.9 | High |


### Strategy Của Tôi

**Loại:** RecursiveChunker

**Mô tả cách hoạt động:**
> *Viết 3-4 câu: strategy chunk thế nào? Dựa trên dấu hiệu gì?*

RecursiveChunker hoạt động bằng cách sử dụng một danh sách các dấu phân cách (separators) theo thứ tự ưu tiên. Nó bắt đầu với dấu phân cách đầu tiên trong danh sách (ví dụ: `\n\n`) để chia văn bản thành các phần. Nếu một phần nào đó quá dài so với `chunk_size`, nó sẽ tiếp tục chia nhỏ phần đó bằng dấu phân cách tiếp theo trong danh sách (ví dụ: `\n`, sau đó đến `. `).

Quá trình này lặp lại một cách đệ quy cho đến khi tất cả các phần đều có kích thước nhỏ hơn hoặc bằng `chunk_size`. Nếu hết tất cả các dấu phân cách mà văn bản vẫn còn quá dài, nó sẽ cắt cứng theo `chunk_size`. Phương pháp này giúp bảo toàn ngữ cảnh tốt hơn so với việc cắt cứng vì nó cố gắng tách văn bản tại các ranh giới tự nhiên như đoạn văn, câu, hoặc khoảng trắng.

**Tại sao tôi chọn strategy này cho domain nhóm?**
Vì domain của nhóm bao gồm nhiều loại tài liệu có cấu trúc khác nhau. `RecursiveChunker` cho phép hệ thống linh hoạt tách văn bản theo các dấu hiệu ngữ nghĩa như đoạn văn (`\n\n`) hoặc dòng (`\n`) trước khi phải cắt cứng. Điều này giúp bảo toàn tính toàn vẹn của thông tin trong chunk, giúp Agent dễ dàng hiểu ngữ cảnh và trả lời chính xác hơn so với việc cắt cố định (Fixed Size) có thể làm chia cắt giữa chừng một ý quan trọng.

**Code snippet:**
```python
def split(self, text: str) -> list[str]:
    def recursive_split(text: str, current_separators: list[str]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]
        if not current_separators:
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        sep = current_separators[0]
        remaining = current_separators[1:]
        parts = text.split(sep)
        final_chunks = []
        
        for part in parts:
            if len(part) <= self.chunk_size:
                if part.strip():
                    final_chunks.append(part.strip())
            else:
                final_chunks.extend(recursive_split(part, remaining))
        return final_chunks

    return recursive_split(text, self.separators)
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| `06_eu_ai_act.txt` | best baseline (Fixed) | 71 | 499.8 | Low (Dễ bị cắt ngang câu/ý) |
| `06_eu_ai_act.txt` | **của tôi** (Recursive) | 104 | 339.4 | High (Giữ trọn vẹn ngữ cảnh) |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | Recursive | 7.5 | Giữ được cấu trúc văn bản (đoạn, câu) thông qua việc tách đệ quy linh hoạt, cân bằng tốt giữa kích thước và ngữ cảnh. | Vẫn dựa trên tách chuỗi vật lý, đôi khi vẫn cắt ngang context nếu các paragraph quá dài hoặc không có dấu phân cách rõ ràng. |
| Lê Trung Anh Quốc | Semantic | 8.0 | Chia chunk dựa trên sự tương đồng ý nghĩa, giữ trọn vẹn mạch thông tin và ngữ cảnh, hạn chế việc cắt ngang các mô tả quan trọng. | Tốn nhiều tài nguyên tính toán (Embedding API), tốc độ xử lý chậm hơn các phương pháp tách chuỗi thông thường. |
| Trần Thái Thịnh | Recursive | 7.5 | Giữ được cấu trúc văn bản (đoạn, câu) thông qua việc tách đệ quy linh hoạt, cân bằng tốt giữa kích thước và ngữ cảnh. | Vẫn dựa trên tách chuỗi vật lý, đôi khi vẫn cắt ngang context nếu các paragraph quá dài hoặc không có dấu phân cách rõ ràng. |
| Nguyễn Đức Cường | Sentence | 6.5 | Đảm bảo tính toàn vẹn của từng câu, giúp mô hình hiểu ngữ cảnh ở mức độ câu tốt hơn so với cắt theo ký tự ngẫu nhiên. | Kích thước chunk không đồng đều (có câu quá dài/quá ngắn), thông tin dễ bị phân mảnh nếu không gộp nhiều câu liên quan. |
| Trần Khánh Bằng | Layered | 8.5 | Truy xuất thông tin đa tầng (kết hợp chunk lớn bao quát và chunk nhỏ chi tiết), giúp Agent nắm bắt cả tổng quan lẫn chi tiết cụ thể. | Cấu trúc index phức tạp, tốn bộ nhớ lưu trữ và đòi hỏi logic truy vấn nâng cao để xử lý các tầng dữ liệu. |
| Đỗ Hải Nam | Semantic | 8.0 | Chia chunk dựa trên sự tương đồng ý nghĩa, giữ trọn vẹn mạch thông tin và ngữ cảnh, hạn chế việc cắt ngang các mô tả quan trọng. | Tốn nhiều tài nguyên tính toán (Embedding API), tốc độ xử lý chậm hơn các phương pháp tách chuỗi thông thường. |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> *Viết 2-3 câu:*
Chiến lược Layered Chunking (kết hợp với Semantic) là lựa chọn tối ưu nhất cho domain hỗn hợp này. Do dữ liệu trải dài từ văn bản pháp luật chi tiết (EU AI Act) đến các bài blog và wiki, cấu trúc đa tầng giúp hệ thống vừa có thể truy xuất nhanh các sự kiện cụ thể (chunk nhỏ), vừa không làm mất đi bức tranh toàn cảnh và mạch lập luận của tài liệu (chunk lớn). Điều này khắc phục được sự phân mảnh thông tin mà các phương pháp Fixed hay Sentence thường gặp phải, đồng thời tối ưu hóa độ chính xác cho câu trả lời của Agent.
---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> *Viết 2-3 câu: dùng regex gì để detect sentence? Xử lý edge case nào?*
Sử dụng `re.split` với lookbehind `(?<=[.!?])\s+` để tách văn bản thành các câu mà không làm mất dấu câu kết thúc. Sau đó, tôi gộp các câu lại thành từng nhóm dựa trên `max_sentences_per_chunk` để tạo thành các chunk hoàn chỉnh.

**`RecursiveChunker.chunk` / `_split`** — approach:
> *Viết 2-3 câu: algorithm hoạt động thế nào? Base case là gì?*
Thuật toán sử dụng đệ quy để thử từng separator trong danh sách ưu tiên. Nếu một đoạn văn bản vẫn vượt quá `chunk_size`, nó sẽ bị chia nhỏ tiếp bởi separator tiếp theo. Base case là khi đoạn văn bản nhỏ hơn `chunk_size` hoặc đã thử hết tất cả các separator (khi đó sẽ cắt cứng theo độ dài).

### EmbeddingStore

**`add_documents` + `search`** — approach:
> *Viết 2-3 câu: lưu trữ thế nào? Tính similarity ra sao?*
Lưu trữ các chunk dưới dạng list các dictionary chứa embedding. Với hàm `search`, tính tích vô hướng (dot product) giữa embedding của query và tất cả các embedding đã lưu để xếp hạng độ tương đồng.

**`search_with_filter` + `delete_document`** — approach:
> *Viết 2-3 câu: cách filter và delete hoạt động thế nào?*
Hàm lọc thực hiện duyệt qua metadata của các chunk trước khi tính similarity để thu hẹp phạm vi tìm kiếm. Hàm xóa sử dụng list comprehension để loại bỏ tất cả các record có `doc_id` khớp với yêu cầu trong metadata.

### KnowledgeBaseAgent

**`answer`** — approach:
> *Viết 2-3 câu: prompt structure? Cách inject context?*
Tôi xây dựng một prompt có cấu trúc bao gồm phần CONTEXT (chứa các chunk đã retrieve được) và phần QUESTION. Prompt yêu cầu Agent chỉ trả lời dựa trên context và thông báo "I don't know" nếu không tìm thấy thông tin.

### Test Results

```
# Paste output of: pytest tests/ -v
```
tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED           [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED                    [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED             [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED              [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED                   [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED   [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED         [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED          [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED        [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED                          [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED          [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED                     [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED                 [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED                           [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED  [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED      [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED      [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED                          [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED            [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED              [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED                    [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED         [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED           [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED            [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED                     [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED                    [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED               [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED           [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED      [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED          [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED                [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED          [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED     [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED    [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED   [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

========================================= 42 passed in 0.08s ==========================================
**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | The cat sits on the mat. | A feline is resting on the rug. | high | 0.8842 | Đúng |
| 2 | I love programming in Python. | Python is a great language for coding. | high | 0.8231 | Đúng |
| 3 | The weather is sunny today. | It is raining heavily outside. | low | 0.4512 | Đúng |
| 4 | The quick brown fox jumps over the lazy dog. | A fast auburn vulpine leaps across the sleepy canine. | high | 0.8653 | Đúng |
| 5 | Computers are useful tools. | Apples are delicious fruits. | low | 0.1245 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
Kết quả thú vị nhất là các cặp câu sử dụng từ đồng nghĩa hoàn toàn khác về mặt ký tự (như "cat/feline", "mat/rug") vẫn đạt điểm tương đồng rất cao (> 0.85). Điều này chứng minh Embeddings không so khớp từ khóa (keyword matching) mà thực sự ánh xạ các khái niệm vào một không gian vector nơi các thực thể có quan hệ ngữ nghĩa nằm gần nhau. Cặp câu về thời tiết (sunny/raining) có điểm số trung bình cho thấy mô hình nhận diện được chúng cùng thuộc một chủ đề dù mang nghĩa tương phản.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer (câu trả lời đúng) | 
|---|-------|-------------------------------|
| 1 | What makes jammed enemies different? | Jammed Keybullet Kin drop 2 keys instead of 1, jammed Chance Kins have a chance to drop twice the loot, and jammed red-Caped Bullet Kin deal contact damage. Additionally, Jammed Keybullet Kin variations run faster and will take less time to teleport away from the player if they are not destroyed quickly. |
| 2 | Which large language models and vector databases were shortlisted for this project? (Filtered by: category=technical) | The tinyllama-1.1b-chat-v1.0 Q6_K, Phi 3 Q4_K_M, bartowski/dolphin-2.8-experiment26-7b-GGUF Q3_K_L, mgonzs13/Mistroll-7B-v2.2-GGU, and QuantFactory/Meta-Llama-3-8B-Instruct Q3_K_M large language models and the Chroma, Qdrant, and Vespa vector databases were shortlisted for this project. |
| 3 | What kind of model is the bling-phi-3 model | The bling-phi-3 model is the newest and most accurate BLING/DRAGON model. BLING models are small CPU-based RAG-optimized, instruct-following 1B-3B parameter models. |
| 4 | What are the ways of grouping UI elements together? | UI elements can be grouped together using the following methods: Create an array of UI elements, Create a dictionary of UI elements, Embed a dynamic number of UI elements in another output, Create a hstack (or vstack) of UI elements with on_change handlers, Create a table column of buttons with on_change handlers, Create a form with multiple UI elements. |
| 5 | What are the four steps to become more impact-focused? | The four steps to become more impact-focused are: "Step 1: Understand what impact looks like for your role...", "Step 2: Ensure your work solves a real business problem", "Step 3: Ensure there is buy-in for your work", and "Step 4: Focus your time on the highest-impact thing". |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | What makes jammed enemies different? | "Jammed Keybullet Kin drop 2 keys instead of 1..." | 0.825 | Yes | Jammed enemies are faster, deal more damage, and drop more loot... |
| 2 | Which LLMs and Vector DBs were shortlisted? | "shortlisted: tinyllama-1.1b, Phi 3, Chroma, Qdrant..." | 0.798 | Yes | The shortlisted models included TinyLlama, Phi 3, Chroma, Qdrant... |
| 3 | What kind of model is the bling-phi-3 model | "bling-phi-3 is the newest and most accurate BLING/DRAGON..." | 0.812 | Yes | It is a small CPU-based, RAG-optimized model with 1B-3B parameters... |
| 4 | What are the ways of grouping UI elements? | "UI elements can be grouped using: array, dictionary, hstack..." | 0.844 | Yes | UI elements can be grouped via arrays, dictionaries, hstacks, or forms... |
| 5 | What are the 4 steps to become impact-focused? | "Step 1: Understand impact; Step 2: Solve real problem..." | 0.831 | Yes | The 4 steps are: 1. Understand impact, 2. Solve real problems, 3. Buy-in, 4. Focus... |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
Tôi đã học được cách tối ưu hóa các tham số của `RecursiveChunker` (như bộ separators) để phù hợp với các định dạng văn bản đặc thù như bảng biểu hay mã nguồn. Việc quan sát cách các thành viên khác gán metadata (như `category` và `source`) giúp tôi hiểu rằng một cấu trúc dữ liệu tốt ngay từ đầu sẽ giúp ích rất nhiều cho việc lọc thông tin và giảm nhiễu khi truy xuất.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
Một số nhóm đã trình bày cách kết hợp giữa RAG và **Conversational Memory**, giúp trợ lý có thể trả lời các câu hỏi tiếp nối (follow-up questions) dựa trên ngữ cảnh đã trò chuyện trước đó. Ngoài ra, kỹ thuật **Re-ranking** (sử dụng một mô hình nhỏ để chấm điểm lại các chunk sau khi truy xuất) là một điều rất đáng để học hỏi để tăng độ chính xác của câu trả lời.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
Tôi sẽ đầu tư nhiều thời gian hơn cho bước **Data Cleaning** để lọc bỏ các thông tin rác hoặc định dạng thừa trong file văn bản thô. Đồng thời, tôi sẽ thử nghiệm chiến lược **Hybrid Search** (kết hợp giữa tìm kiếm theo từ khóa BM25 và tìm kiếm vector) để đảm bảo hệ thống không bỏ lỡ các từ khóa chuyên ngành quan trọng mà đôi khi mô hình Embedding có thể bỏ qua.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | / 5 |
| Document selection | Nhóm | / 10 |
| Chunking strategy | Nhóm | / 15 |
| My approach | Cá nhân | / 10 |
| Similarity predictions | Cá nhân | / 5 |
| Results | Cá nhân | / 10 |
| Core implementation (tests) | Cá nhân | / 30 |
| Demo | Nhóm | / 5 |
| **Tổng** | | **/ 100** |

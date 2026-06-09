# Lab Assignment - Improve Day08 Agent With Supervisor-Workers

Assignment này cải tiến RAG chatbot Day08 thành pattern **Supervisor - Workers**.

## Kiến Trúc

```text
User Question
  -> SupervisorAgent
      -> LegalDocumentWorker
      -> NewsContextWorker
      -> CitationAnswerWorker
  -> Final answer with sources
```

Vai trò:

- `SupervisorAgent`: nhận câu hỏi, gọi các workers, merge evidence, điều phối kết quả cuối.
- `LegalDocumentWorker`: tìm evidence trong văn bản pháp luật.
- `NewsContextWorker`: tìm evidence trong bài báo/tin tức.
- `CitationAnswerWorker`: tạo câu trả lời ngắn có citation từ evidence đã truy xuất.

Pattern này có ít nhất 3 workers, đúng yêu cầu checklist.

## Cấu Trúc

```text
Lab_Assignment/
  README.md
  main.py
  supervisor_workers.py
  requirements.txt
```

## Cách Chạy

Chạy với data Day08:

```powershell
$env:DAY08_DATA_DIR='C:\Users\PC\Downloads\Day08_RAG_pipeline_cohort2\data\standardized'
$env:PYTHONIOENCODING='utf-8'
python Lab_Assignment\main.py "Hình phạt tàng trữ trái phép chất ma túy là gì?"
```

Hoặc truyền path trực tiếp:

```powershell
python Lab_Assignment\main.py "Nghệ sĩ nào liên quan tới ma túy?" --data-dir "C:\Users\PC\Downloads\Day08_RAG_pipeline_cohort2\data\standardized"
```

Nếu không truyền `--data-dir`, chương trình sẽ thử đọc:

1. Biến môi trường `DAY08_DATA_DIR`
2. `data/standardized` trong repo hiện tại
3. `../Day08_RAG_pipeline_cohort2/data/standardized`

## Điểm Cải Tiến So Với Day08

Day08 ban đầu là pipeline RAG tuyến tính:

```text
query -> retrieve -> generate_with_citation
```

Bản assignment tách thành nhiều agent/worker:

```text
Supervisor
  -> worker pháp luật
  -> worker tin tức
  -> worker tạo câu trả lời/citation
```

Lợi ích:

- Dễ mở rộng thêm worker mới như `PolicyWorker`, `EvaluationWorker`, `WebSearchWorker`
- Supervisor có thể kiểm soát worker nào được gọi
- Tách rõ trách nhiệm giữa retrieval pháp luật, retrieval tin tức và generation
- Dễ debug vì mỗi worker trả về evidence riêng

## Ghi Chú

Bản này dùng retrieval lexical thuần Python để dễ chạy trong môi trường nộp bài. Nếu muốn nâng cấp, có thể thay `LegalDocumentWorker` và `NewsContextWorker` bằng các module Day08 như `semantic_search`, `lexical_search`, `retrieve`.


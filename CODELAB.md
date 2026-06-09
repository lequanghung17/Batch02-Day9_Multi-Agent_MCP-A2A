# Codelab: Xây Dựng Hệ Thống Multi-Agent với A2A Protocol

**Thời gian:** 2 giờ  
**Ngôn ngữ:** Python 3.11+  
**Công nghệ:** LangGraph, LangChain, A2A SDK

## Mục Tiêu Học Tập

Sau khi hoàn thành codelab này, bạn sẽ:
- Hiểu cách LLM hoạt động từ cơ bản đến nâng cao
- Biết cách tích hợp tools và RAG vào LLM
- Xây dựng được single agent với ReAct pattern
- Tạo multi-agent system với LangGraph
- Triển khai distributed agents với A2A protocol

## Chuẩn Bị

### Yêu Cầu Hệ Thống
- Python 3.11 trở lên
- [uv](https://docs.astral.sh/uv/) package manager
- API key từ [OpenRouter](https://openrouter.ai)

### Cài Đặt

```bash
# Clone repository
git clone <repo-url>
cd legal_multiagent

# Cài đặt dependencies
uv sync

# Cấu hình environment
cp .env.example .env
# Sửa file .env, thêm OPENROUTER_API_KEY của bạn
```

---

## Phần 1: Direct LLM Calling (20 phút)

### Lý Thuyết

LLM (Large Language Model) ở dạng cơ bản nhất là một API nhận input text và trả về output text. Không có memory, không có tools, chỉ dựa vào training data.

**Ưu điểm:**
- Đơn giản, dễ implement
- Phản hồi nhanh

**Nhược điểm:**
- Không có kiến thức real-time
- Không thể tra cứu database
- Không có context giữa các lần gọi

### Thực Hành

**Bước 1:** Chạy demo Stage 1

```bash
uv run python stages/stage_1_direct_llm/main.py
```

**Bước 2:** Đọc và hiểu code

Mở file `stages/stage_1_direct_llm/main.py` và trả lời:

1. LLM được khởi tạo như thế nào? (Tìm hàm `get_llm()`)
2. Message được gửi đến LLM có cấu trúc gì?
3. Tại sao cần có `SystemMessage` và `HumanMessage`?

**Bài Tập 1.1:** Thay đổi câu hỏi

Sửa biến `QUESTION` thành câu hỏi pháp lý khác (tiếng Việt hoặc tiếng Anh) và chạy lại.

**Bài Tập 1.2:** Thêm temperature control

Thêm parameter `temperature=0.3` vào hàm `get_llm()` trong `common/llm.py` để làm output ổn định hơn.

---

## Phần 2: LLM + RAG & Tools (30 phút)

### Lý Thuyết

**RAG (Retrieval-Augmented Generation):** Cho phép LLM tra cứu knowledge base trước khi trả lời.

**Tools:** Các function mà LLM có thể gọi để thực hiện tác vụ cụ thể (tính toán, query database, gọi API).

**Function Calling Flow:**
1. LLM nhận câu hỏi + danh sách tools
2. LLM quyết định gọi tool nào (hoặc không gọi)
3. Tool được execute, trả về kết quả
4. LLM nhận kết quả và tạo câu trả lời cuối cùng

### Thực Hành

**Bước 1:** Chạy demo Stage 2

```bash
uv run python stages/stage_2_rag_tools/main.py
```

**Bước 2:** Phân tích code

Mở `stages/stage_2_rag_tools/main.py` và tìm:

1. Hàm `@tool` decorator được dùng ở đâu?
2. `LEGAL_KNOWLEDGE` được cấu trúc như thế nào?
3. LLM được bind với tools ra sao? (Tìm `.bind_tools()`)

**Bài Tập 2.1:** Thêm knowledge base entry

Thêm một entry mới vào `LEGAL_KNOWLEDGE` về luật lao động:

```python
{
    "id": "labor_law",
    "keywords": ["lao động", "sa thải", "hợp đồng lao động", "labor", "termination"],
    "text": (
        "Theo Bộ luật Lao động Việt Nam 2019, người sử dụng lao động có thể "
        "đơn phương chấm dứt hợp đồng trong các trường hợp: (1) người lao động "
        "thường xuyên không hoàn thành công việc; (2) bị ốm đau, tai nạn đã điều trị "
        "12 tháng chưa khỏi; (3) thiên tai, hỏa hoạn; (4) người lao động đủ tuổi nghỉ hưu."
    ),
}
```

**Bài Tập 2.2:** Tạo tool mới

Tạo một tool `@tool` mới tên `check_statute_of_limitations` nhận vào `case_type` (string) và trả về thời hiệu khởi kiện:

```python
@tool
def check_statute_of_limitations(case_type: str) -> str:
    """Kiểm tra thời hiệu khởi kiện theo loại vụ án.
    
    Args:
        case_type: Loại vụ án (contract, tort, property)
    """
    limits = {
        "contract": "4 năm (UCC § 2-725)",
        "tort": "2-3 năm tùy bang",
        "property": "5 năm",
    }
    return limits.get(case_type.lower(), "Không xác định")
```

Thêm tool này vào danh sách tools và test.

---

## Phần 3: Single Agent với ReAct (25 phút)

### Lý Thuyết

**ReAct Pattern:** Reasoning + Acting

Agent tự động lặp lại chu trình:
1. **Think:** Suy nghĩ cần làm gì
2. **Act:** Gọi tool
3. **Observe:** Nhận kết quả
4. Lặp lại cho đến khi có câu trả lời cuối cùng

LangGraph cung cấp `create_react_agent` để tự động hóa pattern này.

### Thực Hành

**Bước 1:** Chạy demo Stage 3

```bash
uv run python stages/stage_3_single_agent/main.py
```

**Bước 2:** Quan sát output

Chú ý cách agent tự động:
- Quyết định tool nào cần gọi
- Gọi nhiều tools liên tiếp
- Tổng hợp kết quả

**Bước 3:** Đọc code

Mở `stages/stage_3_single_agent/main.py`:

1. Tìm `create_react_agent()` — đây là magic function
2. So sánh với Stage 2: không còn manual tool loop
3. Xem `agent_executor.invoke()` — chỉ cần gọi một lần

**Bài Tập 3.1:** Thêm tool tra cứu án lệ

```python
@tool
def search_case_law(keywords: str) -> str:
    """Tìm kiếm án lệ theo từ khóa.
    
    Args:
        keywords: Từ khóa tìm kiếm
    """
    cases = {
        "breach": "Hadley v. Baxendale (1854) - Consequential damages",
        "negligence": "Donoghue v. Stevenson (1932) - Duty of care",
        "contract": "Carlill v. Carbolic Smoke Ball Co (1893) - Unilateral contract",
    }
    for key, case in cases.items():
        if key in keywords.lower():
            return case
    return "Không tìm thấy án lệ phù hợp"
```

Thêm vào tools list và test với câu hỏi về breach of contract.

<!-- Khi test, agent đã tự gọi đúng 2 tool:

search_legal_database
search_case_law -->

**Bài Tập 3.2:** Debug agent reasoning

Thêm `verbose=True` vào `create_react_agent()` để xem chi tiết quá trình suy nghĩ của agent.

<!-- cho thấy chi tiết luồng agent: message đầu vào, tool calls, tool results, state updates và final answer. Nó không hiển thị “suy nghĩ nội bộ” riêng tư của model, nhưng đủ để quan sát ReAct loop: agent chọn tool nào, truyền args gì, nhận observation gì, rồi tổng hợp ra sao. -->

---

## Phần 4: Multi-Agent In-Process (30 phút)

### Lý Thuyết

**Multi-Agent System:** Nhiều agents chuyên môn hóa cùng làm việc.

**Ưu điểm:**
- Mỗi agent tập trung vào domain riêng
- Có thể chạy song song (parallel execution)
- Dễ maintain và mở rộng

**LangGraph StateGraph:**
- Định nghĩa state (dữ liệu chia sẻ giữa các nodes)
- Tạo nodes (các bước xử lý)
- Định nghĩa edges (luồng điều khiển)

**Send API:** Cho phép dispatch nhiều tasks song song.

### Thực Hành

**Bước 1:** Chạy demo Stage 4

```bash
uv run python stages/stage_4_milti_agent/main.py
```

**Bước 2:** Phân tích kiến trúc

Mở `stages/stage_4_milti_agent/main.py`:

1. Tìm `class State(TypedDict)` — đây là shared state
2. Tìm các agent functions: `law_agent`, `tax_agent`, `compliance_agent`
3. Tìm `Send()` API — dispatch parallel tasks
4. Xem `graph.add_node()` và `graph.add_edge()`

**Bước 3:** Vẽ graph

```python
# Thêm vào cuối file main.py
from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))
```

**Bài Tập 4.1:** Thêm agent mới

Tạo `privacy_agent` chuyên về GDPR và privacy law:

```python
def privacy_agent(state: State) -> dict:
    """Agent chuyên về luật bảo vệ dữ liệu cá nhân."""
    llm = get_llm()
    
    prompt = f"""Bạn là chuyên gia về GDPR và luật bảo vệ dữ liệu cá nhân.
    
Câu hỏi gốc: {state['question']}
Phân tích pháp lý: {state.get('law_analysis', 'N/A')}

Hãy phân tích các vấn đề về privacy và GDPR (nếu có).
"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"privacy_analysis": response.content}
```

Thêm node này vào graph và kết nối với `aggregate_results`.

**Bài Tập 4.2:** Implement conditional routing

Sửa `check_routing` để chỉ gọi privacy_agent khi câu hỏi có từ khóa "data", "privacy", "gdpr":

```python
def check_routing(state: State) -> list[Send]:
    question_lower = state["question"].lower()
    tasks = []
    
    if any(kw in question_lower for kw in ["tax", "irs", "thuế"]):
        tasks.append(Send("tax_agent", state))
    
    if any(kw in question_lower for kw in ["compliance", "sec", "regulation"]):
        tasks.append(Send("compliance_agent", state))
    
    if any(kw in question_lower for kw in ["data", "privacy", "gdpr", "dữ liệu"]):
        tasks.append(Send("privacy_agent", state))
    
    return tasks if tasks else [Send("aggregate_results", state)]
```

---

## Phần 5: Distributed A2A System (15 phút)

### Lý Thuyết

**A2A (Agent-to-Agent) Protocol:** Chuẩn giao tiếp giữa các agents qua HTTP.

**Khác biệt với Stage 4:**
- Mỗi agent là một service độc lập
- Giao tiếp qua HTTP thay vì in-process
- Dynamic discovery qua Registry
- Có thể scale từng agent riêng biệt

**Kiến trúc:**
```
Registry (10000) ← agents register on startup
    ↓
Customer Agent (10100) → Law Agent (10101)
                              ↓
                    ┌─────────┴─────────┐
                    ↓                   ↓
            Tax Agent (10102)   Compliance Agent (10103)
```

### Thực Hành

**Bước 1:** Khởi động toàn bộ hệ thống

```bash
./start_all.sh
```

Chờ ~10 giây để tất cả services khởi động.

**Bước 2:** Test hệ thống

```bash
uv run python test_client.py
```

**Bước 3:** Quan sát logs

Mở 5 terminal tabs và xem logs của từng service:
- Registry: port 10000
- Customer Agent: port 10100
- Law Agent: port 10101
- Tax Agent: port 10102
- Compliance Agent: port 10103

**Bài Tập 5.1:** Trace request flow

Trong logs, tìm `trace_id` và theo dõi request đi qua các agents. Vẽ sequence diagram.

**Bài Tập 5.2:** Test dynamic discovery

1. Dừng Tax Agent (Ctrl+C)
2. Chạy lại `test_client.py`
3. Quan sát lỗi và cách hệ thống xử lý

**Bài Tập 5.3:** Modify agent behavior

Sửa `tax_agent/graph.py`, thay đổi system prompt để agent trả lời ngắn gọn hơn. Restart tax agent và test lại.

### Kết Quả Thực Hiện Bài 5.1 - 5.3

**Thời điểm chạy:** 2026-06-09  
**Môi trường:** Windows PowerShell, chạy service bằng `.venv\Scripts\python.exe` vì `start_all.sh` là shell script cho Linux/macOS.  
**Thiết lập test:** dùng `OPENROUTER_MODEL=openai/gpt-4o-mini` và `OPENROUTER_MAX_TOKENS=64` để giảm chi phí khi test Stage 5.

**Kết quả khởi động hệ thống:**

Registry chạy ở port `10000` và nhận đủ đăng ký từ 4 agents:

```text
tax-agent        -> http://localhost:10102, tasks=["tax_question"]
compliance-agent -> http://localhost:10103, tasks=["compliance_question"]
law-agent        -> http://localhost:10101, tasks=["legal_question"]
customer-agent   -> http://localhost:10100
```

Log được lưu tại:

```text
.stage5_logs/scenario_5_1_5_2/
.stage5_logs/scenario_5_3/
```

**Bài 5.1 - Trace request flow:**

Khi chạy `test_client.py`, client kết nối được tới Customer Agent:

```text
Connected to agent: Customer Agent v1.0.0
```

Nhưng request bị dừng ngay tại bước Customer Agent gọi LLM vì OpenRouter trả lỗi:

```text
Error code: 402 - Insufficient credits
```

Vì vậy flow thực tế quan sát được là:

```text
test_client
  -> Customer Agent
  -> OpenRouter LLM call
  -> failed: 402 Insufficient credits
```

Flow đầy đủ kỳ vọng khi tài khoản có credit:

```text
test_client
  -> Customer Agent :10100
  -> Registry discover("legal_question") :10000
  -> Law Agent :10101
  -> Registry discover("tax_question") :10000
  -> Tax Agent :10102
  -> Registry discover("compliance_question") :10000
  -> Compliance Agent :10103
  -> Law Agent aggregate
  -> Customer Agent
  -> test_client
```

Ghi chú: trong lần chạy này chưa thấy `trace_id` đầy đủ đi xuyên qua các agent vì request fail trước khi Law Agent/Tax/Compliance được gọi. A2A response có `context_id` và `task_id`, ví dụ `context_id='8f8db2e1-c5ce-421c-9932-6c5ac17a29cc'`.

**Bài 5.2 - Test dynamic discovery:**

Đã dừng Tax Agent rồi chạy lại `test_client.py`.

Kết quả Registry sau khi dừng Tax Agent vẫn còn list `tax-agent`. Điều này cho thấy Registry hiện là in-memory registry đơn giản, có đăng ký agent nhưng chưa có health check/deregister tự động.

Request vẫn fail ở Customer Agent vì lỗi OpenRouter `402`, nên chưa đi tới bước Law Agent discover/call Tax Agent. Theo code trong `law_agent/graph.py`, nếu đã đi tới `call_tax` mà Tax Agent thật sự không reachable, lỗi sẽ được catch và trả về dạng:

```text
[Tax analysis unavailable: ...]
```

**Bài 5.3 - Modify Tax Agent behavior:**

Đã sửa `tax_agent/graph.py`, thêm yêu cầu vào `TAX_SYSTEM_PROMPT`:

```text
Keep the answer concise: use at most 3 bullet points and stay under 120 words.
```

Sau khi restart hệ thống, Registry xác nhận `tax-agent` đã đăng ký lại thành công:

```text
tax-agent -> http://localhost:10102
```

Tuy nhiên, test end-to-end sau khi sửa prompt vẫn fail ở Customer Agent vì OpenRouter `402 Insufficient credits`, nên chưa verify được output Tax Agent đã ngắn hơn trong lần chạy này. Cần nạp credit OpenRouter hoặc dùng API key/model còn quota để kiểm tra kết quả trả lời sau khi sửa prompt.

---

## Phần 6: Tổng Kết & Mở Rộng (10 phút)

### So Sánh 5 Stages

| Stage | Pattern | Use Case | Complexity |
|---|---|---|---|
| 1 | Direct LLM | Câu hỏi đơn giản, không cần tools | ⭐ |
| 2 | LLM + Tools | Cần tra cứu data hoặc tính toán | ⭐⭐ |
| 3 | ReAct Agent | Tự động orchestration, multi-step | ⭐⭐⭐ |
| 4 | Multi-Agent | Nhiều domains, parallel processing | ⭐⭐⭐⭐ |
| 5 | Distributed A2A | Production, scalable, fault-tolerant | ⭐⭐⭐⭐⭐ |

### Câu Hỏi Ôn Tập

1. Khi nào nên dùng single agent thay vì multi-agent?
2. Ưu điểm của A2A protocol so với gRPC hoặc REST thông thường?
3. Làm thế nào để prevent infinite delegation loops trong A2A?
4. Tại sao cần Registry service? Có thể hardcode URLs không?

**Trả lời:**

**1. Khi nào nên dùng single agent thay vì multi-agent?**

Nên dùng single agent khi bài toán đơn giản, phạm vi hẹp, không cần nhiều chuyên gia riêng biệt và không cần xử lý song song.

Ví dụ:
- Chatbot pháp lý cơ bản
- Agent tra cứu tài liệu nội bộ
- Workflow chỉ cần gọi một vài tools
- Prototype hoặc demo nhỏ

Single agent dễ debug, dễ triển khai và ít overhead hơn multi-agent.

**2. Ưu điểm của A2A protocol so với gRPC hoặc REST thông thường?**

A2A phù hợp hơn cho giao tiếp giữa các agents vì nó chuẩn hóa cách agent mô tả năng lực, nhận task, trả message/artifact và xử lý trạng thái task.

Ưu điểm chính:
- Có Agent Card để agent tự mô tả capability
- Hỗ trợ task, message, artifact theo mô hình agent-native
- Dễ làm dynamic discovery qua Registry
- Agent có thể được triển khai độc lập ở các service khác nhau
- Phù hợp với hệ thống nhiều agents, mỗi agent có vai trò riêng

REST/gRPC vẫn dùng được, nhưng thường phải tự thiết kế format request/response, discovery, task status và metadata.

**3. Làm thế nào để prevent infinite delegation loops trong A2A?**

Có thể dùng các cách sau:
- Thêm `MAX_DELEGATION_DEPTH`, ví dụ giới hạn tối đa 3 lần delegate
- Truyền `depth` trong metadata qua mỗi A2A call
- Nếu `depth >= MAX_DELEGATION_DEPTH` thì agent không được delegate tiếp
- Dùng `trace_id` hoặc `context_id` để phát hiện request lặp
- Thiết kế routing rõ ràng, ví dụ Customer -> Law -> Specialist, không cho Specialist gọi ngược Customer/Law tùy tiện

Trong repo này có pattern `MAX_DELEGATION_DEPTH = 3` để tránh vòng lặp vô hạn.

**4. Tại sao cần Registry service? Có thể hardcode URLs không?**

Registry service giúp các agents tìm nhau động theo capability/task, ví dụ:

```text
discover("tax_question") -> http://localhost:10102
```

Lợi ích:
- Không cần hardcode endpoint trong code
- Dễ thay đổi port, host hoặc scale service
- Agent có thể tự register khi startup
- Hệ thống linh hoạt hơn khi thêm/xóa agent
- Phù hợp production hơn, nhất là khi deploy nhiều service

Có thể hardcode URLs trong demo nhỏ, nhưng không nên trong hệ thống thực tế vì khó bảo trì, khó scale và dễ lỗi khi đổi hạ tầng.

### Bài Tập Nâng Cao (Tự Học)

**Challenge 1:** Thêm memory/conversation history

Implement conversation memory để agent nhớ các câu hỏi trước đó.

**Challenge 2:** Add authentication

Thêm API key authentication cho các A2A endpoints.

**Challenge 3:** Implement retry logic

Khi một agent fail, tự động retry với exponential backoff.

**Challenge 4:** Monitoring & Observability

Tích hợp LangSmith hoặc Prometheus để monitor agent performance.

---

## Tài Liệu Tham Khảo

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [A2A Protocol Spec](https://github.com/google/A2A)
- [OpenRouter API](https://openrouter.ai/docs)
- Architecture diagrams: `docs/*.svg`

## Hỗ Trợ

Nếu gặp vấn đề:
1. Check `.env` file có đúng API key không
2. Đảm bảo tất cả ports (10000-10103) không bị chiếm
3. Xem logs trong terminal để debug
4. Đọc error messages cẩn thận — thường có hint rõ ràng

---

## **Bài Tập Cộng Điểm:**
Sau khi chạy full Stage 5 (test_client.py) trả lời 2 câu hỏi:
- Latency (Tổng thời gian trả lời 1 câu hỏi của hệ thống) là bao nhiêu giây?
- Đề xuất phương án giảm latency và demo + show thời gian xử lý đã giảm được khi apply phương án?

### Kết Quả Bài Tập Cộng Điểm

**Thời điểm chạy:** 2026-06-09  
**Câu hỏi test:** `If a company breaks a contract and avoids taxes, what are the legal and regulatory consequences?`  
**Log:** `.stage5_logs/bonus_latency/`

Lưu ý: tại thời điểm test, OpenRouter trả lỗi `402 Insufficient credits`, nên hệ thống chưa tạo được câu trả lời legal hoàn chỉnh. Vì vậy số đo dưới đây là latency từ lúc `test_client.py` gửi request đến lúc hệ thống trả lỗi. Kết quả vẫn hữu ích để so sánh overhead của tầng Customer Agent trước/sau tối ưu.

**Baseline - Customer Agent dùng ReAct/LLM để quyết định delegate:**

```text
Scenario: baseline_customer_react
Result: failed at Customer Agent LLM call
Error: 402 Insufficient credits
Elapsed seconds: 4.55
```

**Phương án giảm latency:**

Customer Agent hiện dùng `create_react_agent` và một LLM call chỉ để quyết định gọi `delegate_to_legal_agent`. Trong prompt đã quy định:

```text
Always use the `delegate_to_legal_agent` tool for any substantive legal question.
```

Vì vậy có thể bỏ LLM hop ở Customer Agent và delegate trực tiếp sang Law Agent. Cách này giảm:
- 1 lần gọi LLM ở Customer Agent
- 1 vòng ReAct tool-call loop
- token prompt/response ở tầng Customer Agent

Đã thêm chế độ tối ưu bằng biến môi trường:

```text
CUSTOMER_DIRECT_DELEGATE=1
```

Khi bật biến này, `customer_agent/agent_executor.py` sẽ:

```text
Customer Agent
  -> Registry discover("legal_question")
  -> Law Agent qua A2A
```

thay vì:

```text
Customer Agent
  -> Customer LLM/ReAct
  -> delegate_to_legal_agent tool
  -> Registry discover("legal_question")
  -> Law Agent qua A2A
```

**Kết quả sau tối ưu:**

```text
Scenario: optimized_direct_delegate_fixed_parser
Result: reached Law Agent, then failed at Law Agent LLM call
Error: 402 Insufficient credits
Elapsed seconds: 3.58
```

**So sánh:**

| Case | Latency |
|---|---:|
| Baseline | 4.55 giây |
| Sau tối ưu | 3.58 giây |
| Giảm được | 0.97 giây |
| Tỷ lệ giảm | khoảng 21.3% |

**Kết luận:**

Phương án direct delegation giúp giảm latency vì loại bỏ một LLM call ở Customer Agent. Khi tài khoản OpenRouter có đủ credit để chạy full response, kỳ vọng mức giảm còn rõ hơn vì baseline phải chờ Customer LLM hoàn tất tool decision trước khi Law Agent bắt đầu xử lý.

Các hướng tối ưu tiếp theo:
- Cache kết quả Registry discovery để giảm HTTP call tới `/discover`
- Giảm `max_tokens` cho các agent trung gian
- Dùng model nhỏ/rẻ hơn cho routing, model mạnh hơn cho bước aggregate cuối
- Chạy Tax Agent và Compliance Agent song song như hiện tại, tránh thêm dependency tuần tự
- Thêm timeout/retry có kiểm soát để tránh chờ lâu khi một specialist agent lỗi

**Chúc các bạn học tốt! 🚀**

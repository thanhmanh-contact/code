# GIẢI PHÁP GIẢI CÂU HỎI TRẮC NGHIỆM ĐA NGÀNH (MCQ SOLVER)

Dự án này triển khai một hệ thống tự động giải quyết các câu hỏi trắc nghiệm đa ngành (Lịch sử, Luật, Toán, Lý, Hóa, Kinh tế...) bằng tiếng Việt. Hệ thống sử dụng mô hình ngôn ngữ lớn tối ưu **Qwen2.5-3B-Instruct** kết hợp với kiến trúc RAG (Retrieval-Augmented Generation) chạy hoàn toàn ngoại tuyến (Offline) trong môi trường Docker, đảm bảo tính chính xác cao và tối ưu thời gian phản hồi (Latency).

---

## 1. Pipeline Flow (Luồng xử lý của hệ thống)

Hệ thống hoạt động theo một quy trình khép kín từ khâu tiếp nhận câu hỏi dữ liệu thô cho đến khi xuất kết quả cuối cùng theo chuẩn của BTC. 

### Sơ đồ luồng xử lý (Flowchart)
```text
[private_test.json] ──> (Trích xuất Câu hỏi & Lựa chọn)
                               │
                               ▼
                    (Cơ chế Phân loại Câu hỏi)
                     ├── Toán/Định lượng  ──> Prompt Ngắn gọn, súc tích
                     └── Xã hội/Đọc hiểu  ──> Prompt Phân tích sâu, loại trừ
                               │
                               ▼
                  (Truy vấn Vector DB - RAG) ──> Lấy ngữ cảnh bổ trợ (nếu có)
                               │
                               ▼
                  (Đóng gói Cấu trúc Prompt) ──> System Prompt + Context + MCQ
                               │
                               ▼
                 [Mô hình Qwen2.5-3B-Instruct] ──> Sinh chuỗi suy luận (Reasoning)
                               │
                               ▼
                  (Bộ lọc Trích xuất Regex) ──> Bẫy từ khóa Tiếng Việt/Anh
                               │
                               ▼
         [submission.csv]  &  [submission_time.csv] (Đo thời gian từng câu)

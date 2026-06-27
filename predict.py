import json
import os
import re
import gc
import time
from typing import List, Dict, Tuple
import torch
import pandas as pd
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

# ── 1. CẤU HÌNH ĐƯỜNG DẪN THEO CHUẨN DOCKER BTC ────────────────
MODEL_NAME      = "./models/Qwen2.5-3B-Instruct-bnb-4bit" 
DATA_PATH       = "/code/private_test.json"  
if os.path.exists("/code/output_test"):
    SUB_PATH    = "/code/output_test/submission.csv"
    TIME_PATH   = "/code/output_test/submission_time.csv"
else:
    SUB_PATH    = "./submission.csv"
    TIME_PATH   = "./submission_time.csv"

MAX_SEQ_LEN     = 4096
MAX_NEW_TOKENS  = 512
GEN_TEMPERATURE = 0.0
SEED            = 42

torch.manual_seed(SEED)

# ── 2. TỐI ƯU SYSTEM PROMPT TIẾNG VIỆT ĐA NGÀNH NÂNG CAO ───────
SYSTEM_PROMPT = """Bạn là một chuyên gia AI xuất sắc, sở hữu tri thức chuyên sâu và khả năng lập luận logic cao cấp để giải quyết các câu hỏi trắc nghiệm (MCQ) bằng tiếng Việt thuộc nhiều lĩnh vực:
- Lịch sử, Chính trị, Triết học & Địa lý
- Pháp luật & Giáo dục công dân
- Đọc hiểu văn bản & Kiến thức tổng hợp
- Khoa học tự nhiên (Vật lý, Hóa học, Sinh học, Môi trường)
- Kinh tế học (Vĩ mô, Vi mô) & Tài chính - Kế toán
- Toán học & Thống kê định lượng

HƯỚNG DẪN TƯ DUY VÀ ĐIỀU TIẾT ĐỘ DÀI:
1. ĐỐI VỚI CÂU TOÁN / TÍNH TOÁN / ĐỊNH LƯỢNG: Hãy đi thẳng vào các biến số, công thức và thực hiện các bước tính toán một cách cực kỳ NGẮN GỌN, súc tích. Không giải thích lý thuyết dài dòng.
2. ĐỐI VỚI CÂU XÃ HỘI / LUẬT / ĐỌC HIỂU: Hãy phân tích kỹ lưỡng bối cảnh, ngữ cảnh và loại trừ các phương án nhiễu một cách chi tiết (VIẾT DÀI VÀ SÂU).
3. Luôn đánh giá khách quan tất cả các lựa chọn trước khi đưa ra kết luận cuối cùng.

Bạn PHẢI phản hồi chính xác theo cấu trúc nghiêm ngặt sau đây:
### Reasoning
<phần lập luận từng bước bằng tiếng Việt dựa trên quy định độ dài ở trên, tối đa 150 từ>
### Answer
<chỉ ghi duy nhất một chữ cái viết hoa đại diện cho đáp án đúng>"""

def choices_to_letter(idx: int) -> str:
    if idx < 26:
        return chr(65 + idx)
    return chr(65 + idx // 26 - 1) + chr(65 + idx % 26)

def get_valid_letters(n_choices: int) -> List[str]:
    return [choices_to_letter(i) for i in range(n_choices)]

def build_prompt(question: str, choices: List[str]) -> str:
    labeled = "\n".join(f"{choices_to_letter(i)}. {c.strip()}" for i, c in enumerate(choices))
    valid_letters = ", ".join(choices_to_letter(i) for i in range(len(choices)))
    return (
        f"Câu hỏi:\n{question.strip()}\n\n"
        f"Các lựa chọn:\n{labeled}\n\n"
        f"Chọn đáp án đúng nhất ({valid_letters}):"
    )

# ── 3. BỔ SUNG BẪY REGEX TIẾNG VIỆT (TRÁNH MẤT ĐIỂM OAN) ───────
def extract_answer(model_output: str, n_choices: int) -> str:
    valid_set = set(get_valid_letters(n_choices))
    text = model_output.strip()

    # Loại bỏ phần chặn suy nghĩ nếu có
    think_end = text.rfind("</think>")
    if think_end != -1:
        text = text[think_end + len("</think>"):].strip()

    # 1. Bẫy chuẩn format BTC
    m = re.search(r"###\s*Answer[:\s]*\n?\s*([A-Z]{1,2})", text, re.IGNORECASE)
    if m and m.group(1).upper() in valid_set:
        return m.group(1).upper()

    # 2. Bẫy từ khóa Tiếng Việt (Mới bổ sung)
    m = re.search(r"(?:đáp\s*án|chọn|phương\s*án)\s*(?:đúng\s*là|chính\s*xác\s*là|:)?\s*\(?([A-Z]{1,2})\)?", text, re.IGNORECASE)
    if m and m.group(1).upper() in valid_set:
        return m.group(1).upper()

    # 3. Bẫy từ khóa Tiếng Anh phòng hờ
    m = re.search(r"(?:the\s+)?(?:correct\s+)?answer\s*(?:is|:)\s*\(?([A-Z]{1,2})\)?", text, re.IGNORECASE)
    if m and m.group(1).upper() in valid_set:
        return m.group(1).upper()

    m = re.search(r"(?:option|choice|select)\s+([A-Z]{1,2})\b", text, re.IGNORECASE)
    if m and m.group(1).upper() in valid_set:
        return m.group(1).upper()

    # 4. Quét 300 ký tự cuối cùng
    tail = text[-300:]
    hits = [c for c in re.findall(r"\b([A-Z]{1,2})\b", tail) if c.upper() in valid_set]
    if hits:
        return hits[-1].upper()

    # 5. Quét toàn văn bản
    hits = [c for c in re.findall(r"\b([A-Z]{1,2})\b", text) if c.upper() in valid_set]
    if hits:
        return hits[-1].upper()

    return "A"

# ── 4. TẢI MÔ HÌNH (OFFLINE READY) ────────────────────────────
print(f"Loading Model from {MODEL_NAME}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = MODEL_NAME,
    max_seq_length = MAX_SEQ_LEN,
    dtype          = None,
    load_in_4bit   = True,
)
tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
tokenizer.padding_side = "left"
FastLanguageModel.for_inference(model)

# ── 5. LOAD DATA TEST ──────────────────────────────────────────
def load_test_data(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("data", "items", "questions", "test"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
        else:
            data = list(data.values())
    return data

# ── 6. VÒNG LẶP HỒI QUY ĐO THỜI GIAN CHUẨN ─────────────────────
def main():
    if not os.path.exists(DATA_PATH):
        print(f"Không tìm thấy file test tại {DATA_PATH}")
        return

    test_items = load_test_data(DATA_PATH)
    print(f"Bắt đầu chấm Pipeline với {len(test_items)} câu hỏi.")

    submission_results = []
    time_results = []

    for idx, item in enumerate(test_items):
        qid = str(item.get("qid", idx))
        question = item["question"]
        choices = item["choices"]
        n_choices = len(choices)

        prompt_text = build_prompt(question, choices)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt_text}
        ]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([input_text], return_tensors="pt").to(model.device)

        # Ép GPU giải quyết hết các tác vụ tồn đọng trước khi bấm giờ
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens     = MAX_NEW_TOKENS,
                do_sample          = False,
                temperature        = None,
                repetition_penalty = 1.1,
                pad_token_id       = tokenizer.eos_token_id,
                use_cache = True,
            )
        
        # Ép GPU phải chạy xong hoàn toàn câu này rồi mới dừng đồng hồ
        torch.cuda.synchronize()
        end_time = time.time()
        
        time_infer_sample = end_time - start_time
        # ───────────────────────────────────────────

        # Trích xuất dữ liệu
        in_len = inputs["input_ids"].shape[1]
        new_ids = outputs[0][in_len:]
        raw_output = tokenizer.decode(new_ids, skip_special_tokens=True)
        final_answer = extract_answer(raw_output, n_choices)

        submission_results.append({"qid": qid, "answer": final_answer})
        time_results.append({"qid": qid, "answer": final_answer, "time": time_infer_sample})

    # ── 7. XUẤT FILE ĐỒNG BỘ ──────────────────────────────────
    df_sub = pd.DataFrame(submission_results)
    df_sub[["qid", "answer"]].to_csv(SUB_PATH, index=False)
    print(f" Saved {SUB_PATH}")

    df_time = pd.DataFrame(time_results)
    df_time[["qid", "answer", "time"]].to_csv(TIME_PATH, index=False)
    print(f" Saved {TIME_PATH}")

if __name__ == "__main__":
    main()
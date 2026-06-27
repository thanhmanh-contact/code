# Sử dụng Base Image CUDA 12.8 theo yêu cầu sửa đổi cho card RTX 5060Ti
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /code

# Cài đặt Python và các công cụ hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Cài đặt công cụ uv để tăng tốc độ cài đặt và quản lý gói
RUN pip3 install --no-cache-dir --upgrade pip uv

COPY requirements.txt .

# Lệnh cài đặt tiêu chuẩn, sạch sẽ nhất
RUN uv pip install --system --no-cache-dir -r requirements.txt --torch-backend=cu128
# Copy toàn bộ code và weights mô hình cục bộ vào thư mục /code
COPY . .

RUN chmod +x inference.sh

# Chạy script bash khởi động theo đúng định dạng template của BTC
CMD ["bash", "inference.sh"]
TRAKE V4: AIC Multimodal Video Retrieval System
Hệ thống truy xuất video đa phương thức được thiết kế để xử lý các truy vấn ngôn ngữ tự nhiên dạng chuỗi sự kiện theo trình tự thời gian (Chronological Events). Dự án sử dụng mã nguồn mở phục vụ cho AI Challenge (AIC) 2026.

Kiến trúc Hệ thống
TRAKE V4 giải quyết bài toán truy xuất thông qua pipeline 3 bước:
Multimodal Embedding & Retrieval: Sử dụng mô hình SigLIP-2 (google/siglip2-so400m-patch16-384) để nhúng văn bản và hình ảnh thành vector 1152 chiều. Dữ liệu được tìm kiếm song song qua thư viện FAISS (In-memory IndexFlatIP) trên cả 2 luồng: Visual (ảnh Keyframes) và Text (Transcripts/Summaries).  
Candidate Fusion: Kết hợp điểm số đa phương thức bằng trọng số tuyến tính (0.6 * Visual + 0.4 * Text), sau đó tính toán độ phủ (coverage) sự kiện để lọc ra Top N video tiềm năng nhất.  
DANTE Temporal DP: Thuật toán quy hoạch động (Dynamic Programming) đóng vai trò khớp nối thời gian. DANTE áp dụng khoảng cách tối thiểu (min_frame_gap = 2) và hàm phạt gap_penalty (lambda_weight = 0.0002, ideal_gap = 15) để ráp nối các khung hình không gian rời rạc thành một chuỗi sự kiện liền mạch.  

Cấu trúc Thư mục
temporal-retrieval-aic/
│   .gitignore
│   requirements.txt          
│   main.py                   
│
├── configs/                  
├── data/                     # Đặt dataset, metadata.csv và merged_embeddings.npy tại đây
│       README.md
├── docs/                     
├── examples/                 
└── src/                      # Mã nguồn phân tách (config, data_loader, retrieval, dante...)

Hướng dẫn Cài đặt & Sử dụng
1. Cài đặt môi trường
Yêu cầu Python 3.10+ và có hỗ trợ GPU/CUDA.

git clone https://github.com/tlmduc7796/temporal-retrieval-aic.git
cd temporal-retrieval-aic
pip install -r requirements.txt

2. Chuẩn bị Dữ liệu
Tải dataset (Keyframes, Transcripts) và các file embedding (merged_embeddings.npy, merged_metadata.csv) đặt vào thư mục data/. Đảm bảo cấu hình đúng đường dẫn tuyệt đối bên trong src/config.py

3. Khởi chạy Hệ thống
Cập nhật mảng USER_QUERY trong file main.py để thay đổi chuỗi sự kiện truy vấn và thực thi:
python main.py

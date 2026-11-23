# PlantVillage Disease Classification Using Deep Learning

Đồ án cuối kỳ môn Deep Learning nghiên cứu khả năng nhận diện loài cây và bệnh lá từ ảnh bằng PyTorch. Dự án so sánh mạng CNN tự xây dựng, ResNet-18 đóng băng backbone và ResNet-18 fine-tuning, đồng thời đánh giá tác động của Data Augmentation.

## 1. Giới thiệu đề tài

Việc quan sát lá cây thủ công phụ thuộc nhiều vào kinh nghiệm, tốn thời gian và dễ sai sót khi số lượng bệnh giống nhau hoặc xuất hiện trong điều kiện chụp khác nhau. Deep Learning, đặc biệt Convolutional Neural Network, có khả năng học đặc trưng hình ảnh trực tiếp và hỗ trợ nhận diện tự động.

Dự án sử dụng ảnh lá thực tế từ bộ dữ liệu PlantVillage. Pipeline thực hiện khảo sát dữ liệu, chia tập bằng chiến lược phân tầng, huấn luyện nhiều mô hình, theo dõi thí nghiệm, đánh giá trên tập test và phân tích lỗi.

Dự án được triển khai bằng Python và PyTorch. Các notebook là điểm nhập chính để tái lập quy trình; thư mục `src/` chứa các module dùng lại cho dữ liệu, mô hình và huấn luyện.

## 2. Mục tiêu

- Xây dựng mô hình nhận diện loài cây và tình trạng bệnh từ ảnh lá.
- So sánh SimpleCNN baseline với ResNet-18 pretrained.
- Tách ảnh hưởng của backbone đóng băng và fine-tuning toàn phần.
- Đánh giá tác động của Data Augmentation.
- Sử dụng Macro-F1 làm metric chính cho bài toán nhiều lớp mất cân bằng.
- Giữ seed, split, preprocessing và ngân sách huấn luyện nhất quán giữa các thí nghiệm.
- Ghi đầy đủ metric, checkpoint, lỗi và traceback để tái lập và debug.
- Phân tích ma trận nhầm lẫn, sai sót theo lớp và khả năng suy luận của mô hình.

## 3. Câu hỏi nghiên cứu

### RQ1 - Kiến trúc và học chuyển giao

Fine-tuning toàn bộ ResNet-18 có cải thiện Macro-F1 trên tập validation và test so với đóng băng backbone hay huấn luyện SimpleCNN từ đầu hay không?

| Thực nghiệm | Kiến trúc          | Backbone                | Augmentation |
| ------------- | -------------------- | ----------------------- | ------------ |
| E0            | SimpleCNN            | Huấn luyện từ đầu  | Không       |
| E1            | ResNet-18 pretrained | Đóng băng            | Không       |
| E2            | ResNet-18 pretrained | Fine-tuning toàn phần | Không       |
| E3            | ResNet-18 pretrained | Fine-tuning toàn phần | Có          |

Nội dung thực hiện:

- Chia dữ liệu bằng stratified split seed 42.
- Giữ cùng input, batch size, seed và ngân sách epoch.
- Chọn checkpoint theo validation Macro-F1.
- Chỉ đánh giá test sau khi khóa cấu hình và checkpoint tốt nhất.
- So sánh accuracy, loss, Macro-F1 và ma trận nhầm lẫn.

### RQ2 - Data Augmentation

Data Augmentation có cải thiện khả năng tổng quát hóa và giảm nhầm lẫn giữa các lớp bệnh tương tự hay không?

Nội dung thực hiện:

- So sánh E2 với E3 bằng cùng kiến trúc và learning rate.
- Chỉ giữ augmentation ở phía E3.
- Theo dõi learning curve, validation Macro-F1 và độ ổn định giữa các seed.
- Phân tích per-class F1, recall và các cặp lớp dễ nhầm.

### RQ3 - Phân tích lỗi và khả năng suy luận

Mô hình dễ nhầm giữa những cặp bệnh nào, sai sót tập trung ở loại cây hay tình trạng nào, và mô hình có phụ thuộc vào phông nền hay không?

Nội dung thực hiện:

- Phân tích confusion matrix.
- So sánh sai theo loài cây và theo tình trạng khỏe/bệnh.
- Kiểm tra ảnh mẫu, ảnh có xác suất thấp và ảnh bị phân loại sai.
- Sử dụng Grad-CAM trong phiên bản sau để kiểm tra vùng mô hình đang quan sát.

## 4. Dataset

### 4.1. Nguồn dữ liệu

Dự án sử dụng [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset), chỉ dùng nhóm ảnh `raw/color` để tránh đưa cùng ảnh nguồn vào nhiều chế độ dữ liệu.

Thư mục dữ liệu ảnh nằm ngoài Git vì dung lượng lớn:

```text
PlantVillage-Dataset-master/
└── PlantVillage-Dataset-master/
    └── raw/
        └── color/
```

### 4.2. Cấu trúc nhãn

Tên lớp có dạng:

```text
Loai_cay___Tinh_trang_benh
```

Ví dụ: `Tomato___Tomato_Yellow_Leaf_Curl_Virus`, `Apple___healthy`.

Tình trạng được tách thành hai nhóm phục vụ phân tích:

- `Khỏe`.
- `Bệnh`.

### 4.3. Quy mô và kiểm tra dữ liệu

EDA đã xác nhận:

- 54.305 ảnh hợp lệ.
- 38 lớp phân loại.
- Toàn bộ ảnh đều có kích thước 256×256.
- 54.304 ảnh ở mode RGB và 1 ảnh ở mode RGBA.
- Không phát hiện ảnh hỏng trong nhóm dữ liệu đang kiểm tra.
- Có ảnh trùng nội dung sau chuẩn hóa 64×64 RGB.

Trước huấn luyện, dữ liệu trùng cần được gom hoặc loại để tránh rò rỉ giữa train, validation và test.

### 4.4. Split dữ liệu

Mô phỏng theo seed 42:

| Split      | Tỷ lệ | Số ảnh |
| ---------- | ------: | -------: |
| Train      |     70% |   38.013 |
| Validation |     15% |    8.146 |
| Test       |     15% |    8.146 |

Split được tạo bằng stratified sampling theo lớp. Test chỉ dùng một lần ở notebook đánh giá cuối.

## 5. Phương pháp thực hiện

### 5.1. Tiền xử lý

- Resize ảnh về kích thước mô hình.
- Chuyển mọi ảnh sang RGB.
- Chuẩn hóa tensor theo thống kê ImageNet.
- Xem xét loại trùng ảnh trước split cuối cùng.

### 5.2. Mô hình

- **SimpleCNN:** baseline CNN huấn luyện từ đầu.
- **ResNet-18 frozen:** pretrained ImageNet; chỉ huấn luyện classifier cuối.
- **ResNet-18 fine-tuning:** mở toàn bộ tham số để tinh chỉnh.
- **ResNet-18 fine-tuning + augmentation:** bổ sung augmentation khi huấn luyện.

### 5.3. Augmentation

Các biến đổi mặc định bao gồm:

- Resize.
- Lật ngang ngẫu nhiên.
- Xoay ngẫu nhiên trong khoảng vừa phải.
- Thay đổi brightness, contrast và saturation.

### 5.4. Metric đánh giá

Metric chính:

- Macro-F1.
- Accuracy.
- Validation loss.
- Per-class F1 và recall.
- Confusion matrix.

Metric được tính trên train và validation mỗi epoch. Tập test được giữ cho đánh giá cuối sau khi chọn cấu hình.

### 5.5. Giao thức thực nghiệm

- Giữ seed cố định cho mỗi lần chạy.
- Chạy 4 thực nghiệm với 5 seed: 42, 43, 44, 45, 46.
- Chọn checkpoint theo validation Macro-F1.
- Ghi loss và metric từng epoch.
- Ghi log thành công, thất bại, lỗi và traceback.
- Không ghi đè checkpoint hoặc kết quả của run cũ.

## 6. Hệ thống logging

Dự án lưu nhật ký ở hai nơi:

- `outputs/experiment_log.jsonl`: log local dạng JSON Lines.
- W&B: run, metric, cấu hình và traceback.

Notebook `02_train_experiments.ipynb` tạo một run cho mỗi seed và mỗi lần retry. Khi run thất bại:

- Lỗi được ghi vào JSONL.
- Traceback được lưu trong W&B summary.
- Run được `finish()` thay vì để dangling.
- Hệ thống tự retry tối đa 3 lần.

### Cài đặt và đăng nhập W&B

```powershell
pip install -r requirements.txt
wandb login
```

Chạy notebook thực nghiệm bình thường sẽ đồng bộ run lên W&B online. Để chạy offline:

```powershell
$env:WANDB_MODE='offline'
```

Đồng bộ run offline sau:

```powershell
wandb sync <đường-dẫn-thư-mục-wandb-offline>
```

## 7. Cấu trúc repository

```text
plant_disease_project/
├── notebooks/
│   ├── 00_eda.ipynb
│   ├── 01_prepare_data.ipynb
│   ├── 02_train_experiments.ipynb
│   ├── 03_evaluate_and_error_analysis.ipynb
│   ├── 04_inference_demo.ipynb
│   └── 05_analyze_logs.ipynb
├── report/
│   ├── bao_cao.md
│   ├── baocao.docx
│   ├── references.bib
│   └── plantvillage.pptx
├── src/
│   ├── data.py
│   ├── model.py
│   ├── prepare_splits.py
│   └── train.py
├── data/                 # metadata và split đã sinh
├── checkpoints/          # checkpoint mô hình
├── outputs/              # biểu đồ, history và audit log
└── README.md
```

Dataset, `wandb/`, `checkpoints/` và `outputs/` không nên đưa lên GitHub khi dung lượng lớn.

## 8. Cài đặt và chạy

### Yêu cầu

- Python 3.10 trở lên.
- PyTorch và torchvision.
- CPU hoặc GPU CUDA.
- Jupyter Notebook/JupyterLab.

### Cài dependency

```powershell
cd plant_disease_project
python -m pip install -r requirements.txt
jupyter lab
```

### Tạo split dữ liệu

```powershell
python -m src.prepare_splits --data-root ..\PlantVillage-Dataset-master\PlantVillage-Dataset-master\raw\color --output data\metadata.csv --seed 42
```

### Chạy notebook theo thứ tự

1. `00_eda.ipynb`: Khảo sát dữ liệu.
2. `01_prepare_data.ipynb`: Tạo split.
3. `02_train_experiments.ipynb`: Huấn luyện và W&B logging.
4. `03_evaluate_and_error_analysis.ipynb`: Đánh giá test và phân tích lỗi.
5. `04_inference_demo.ipynb`: Chạy suy luận trên ảnh mới.
6. `05_analyze_logs.ipynb`: Tổng hợp run thành công/thất bại.

## 9. Công nghệ sử dụng

| Công nghệ         | Mục đích                                  |
| ------------------- | -------------------------------------------- |
| Python              | Ngôn ngữ lập trình chính                |
| PyTorch             | Huấn luyện và suy luận                   |
| torchvision         | ResNet-18, pretrained weights và transforms |
| NumPy, Pandas       | Xử lý và tổng hợp dữ liệu             |
| scikit-learn        | Metric và train/validation/test split       |
| Matplotlib, Seaborn | Trực quan hóa                              |
| Pillow              | Đọc và xử lý ảnh                       |
| tqdm                | Theo dõi tiến trình                       |
| W&B                 | Quản lý và phân tích thí nghiệm       |
| Jupyter             | Thực hiện EDA và thí nghiệm             |
| Git / GitHub        | Quản lý mã nguồn                         |

## 10. Trạng thái dự án

**Trạng thái: Đang thực hiện.**

Đã hoàn thành:

- EDA cơ bản và phân tích sâu.
- Discovery ảnh và mapping lớp.
- Stratified train/validation/test split.
- SimpleCNN và ResNet-18 builder.
- Training loop, metric, checkpoint và JSONL logger.
- Tích hợp W&B cho run thành công/thất bại.
- Cơ chế retry tối đa 3 lần.
- Smoke test thành công và thất bại.

Đang thực hiện:

- Chạy đầy đủ 20 run thực nghiệm.
- Loại trùng ảnh trước split cuối cùng.
- Hoàn thiện đánh giá test và confusion matrix.
- Phân tích sai phân loại và Grad-CAM.
- Cập nhật biểu đồ, bảng kết quả và báo cáo Word.

Chưa bắt đầu đầy đủ:

- Phân tích Grad-CAM cuối cùng.
- Kết luận cuối cùng cho từng RQ.
- Tổng hợp bảng mean ± std theo seed.
- Demo suy luận hoàn chỉnh.

## 11. Đầu ra mong đợi

- Source code huấn luyện và đánh giá.
- Bảng so sánh mô hình và seed.
- Learning curve và confusion matrix.
- Phân tích per-class và ảnh nhầm lẫn.
- Grad-CAM cho các trường hợp điển hình.
- Checkpoint mô hình tốt nhất.
- Nhật ký W&B và JSONL audit.
- Notebook suy luận.
- Báo cáo nghiên cứu và slide.

## 12. Ghi chú hỗ trợ AI

AI hỗ trợ tham khảo mã, tài liệu, quy trình logging và phân tích. Tất cả kết quả để báo cáo đều được chạy và kiểm chứng từ thí nghiệm thật. Em xin cam đoan không lấy kết quả mô hình từ văn bản do AI sinh ra.

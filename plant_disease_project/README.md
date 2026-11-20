# Phân loại bệnh lá cây bằng Deep Learning

Đồ án cuối kỳ môn Deep Learning: nhận diện đồng thời loài cây và bệnh trên ảnh lá
PlantVillage bằng PyTorch. Notebook là nơi thực hiện toàn bộ quy trình; thư mục
`src/` chỉ chứa các hàm dùng lại và tiện ích dòng lệnh.

## Câu hỏi nghiên cứu

- **RQ1:** Fine-tuning toàn bộ ResNet-18 có cải thiện Macro-F1 trên test so với
  chỉ huấn luyện classifier trên backbone đóng băng không?
- **RQ2:** Augmentation có cải thiện khả năng tổng quát hóa và làm giảm lỗi giữa
  các lớp bệnh tương tự không?

| Thực nghiệm | Trả lời |
|---|---|
| E0: SimpleCNN baseline | Mốc tham chiếu không dùng pretrained model |
| E1: ResNet-18, backbone frozen | RQ1 |
| E2: ResNet-18, fine-tuning | RQ1 |
| E3: ResNet-18 fine-tuning + augmentation | RQ2 |

Mọi thực nghiệm giữ nguyên seed, split, metric và ngân sách epoch; test chỉ được
dùng một lần ở notebook đánh giá cuối.

## Cấu trúc

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
│   └── references.bib
├── src/
│   ├── data.py
│   ├── model.py
│   ├── prepare_splits.py
│   └── train.py
├── data/                 # metadata và split sinh ra, không commit ảnh gốc
├── checkpoints/          # model checkpoints
└── outputs/              # biểu đồ, bảng và confusion matrix
```

## Dữ liệu

Tải PlantVillage từ [repository chính thức](https://github.com/spMohanty/PlantVillage-Dataset)
và trỏ `DATA_ROOT` tới thư mục `PlantVillage-Dataset-master/raw/color`.
Chỉ dùng một chế độ ảnh (`color`) để tránh đưa cùng ảnh vào nhiều split.

## Chạy

```powershell
cd C:\Users\kelac\Downloads\HOCTAP\deeplearning\cuoiky\DUAN\plant_disease_project
pip install -r requirements.txt
jupyter lab
```

Chạy notebook theo thứ tự từ `00` đến `04`. Nếu cần tạo split từ terminal:

```powershell
python -m src.prepare_splits --data-root ..\PlantVillage-Dataset-master\PlantVillage-Dataset-master\raw\color --output data\metadata.csv
```

Notebook sẽ ghi lại seed, cấu hình, phiên bản thư viện, kết quả metric và đường
dẫn checkpoint để tái lập.

## Báo cáo

Bản báo cáo theo cấu trúc riêng của đề tài nằm tại
[report/bao_cao.md](report/bao_cao.md), kèm danh mục BibTeX tại
[report/references.bib](report/references.bib). Các bảng kết quả có ô đánh dấu
để điền trực tiếp sau khi chạy thực nghiệm, tránh ghi số liệu chưa được kiểm
chứng.

Bản Word để nộp/chỉnh sửa nằm tại
[report/bao_cao_plantvillage.docx](report/bao_cao_plantvillage.docx). Trong Word,
nhấn chuột phải vào mục lục và chọn **Update Field** để cập nhật số trang.
Notebook `05_analyze_logs.ipynb` dùng để tổng hợp 20 run từ
`outputs/experiment_log.jsonl`.
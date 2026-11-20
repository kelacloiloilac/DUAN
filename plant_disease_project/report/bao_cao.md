# PHÂN LOẠI BỆNH LÁ CÂY BẰNG DEEP LEARNING

**Đồ án cuối kỳ môn Deep Learning**  
**Nhóm:** `[Tên nhóm]`  
**Thành viên:** `[Họ tên - MSSV - phần trăm đóng góp]`  
**Giảng viên:** `[Tên giảng viên]`  
**Thời gian:** `[Tháng/năm]`

> **Quy ước cập nhật:** Các ô `[ĐIỀN SAU KHI CHẠY NOTEBOOK]` là số liệu phải lấy
> trực tiếp từ `02_train_experiments.ipynb` và
> `03_evaluate_and_error_analysis.ipynb`; không tự điền số liệu giả định.

## TÓM TẮT

Đề tài xây dựng hệ thống phân loại ảnh lá cây thành các lớp loài cây và trạng
thái bệnh/khỏe bằng PyTorch trên PlantVillage. Nghiên cứu tập trung vào hai câu
hỏi: (1) fine-tuning ResNet-18 có tốt hơn cách đóng băng backbone và chỉ huấn
luyện classifier hay không; (2) augmentation có cải thiện khả năng tổng quát
hóa hay không. Thực nghiệm sử dụng một split cố định gồm train, validation và
test; chọn mô hình bằng Macro-F1 trên validation và chỉ báo cáo test sau khi
khóa cấu hình. Ngoài ResNet-18, SimpleCNN được dùng làm baseline từ đầu.

**Từ khóa:** PlantVillage, bệnh lá cây, phân loại ảnh, PyTorch, ResNet-18,
transfer learning, augmentation, Macro-F1.

## MỤC LỤC

1. [Giới thiệu](#1-giới-thiệu)
2. [Công trình liên quan](#2-công-trình-liên-quan)
3. [Dữ liệu và bài toán](#3-dữ-liệu-và-bài-toán)
4. [Phương pháp đề xuất](#4-phương-pháp-đề-xuất)
5. [Thiết kế thực nghiệm](#5-thiết-kế-thực-nghiệm)
6. [Kết quả](#6-kết-quả)
7. [Phân tích lỗi và thảo luận](#7-phân-tích-lỗi-và-thảo-luận)
8. [Kết luận và hướng phát triển](#8-kết-luận-và-hướng-phát-triển)
9. [Tài liệu tham khảo](#9-tài-liệu-tham-khảo)
10. [Phụ lục](#10-phụ-lục)

## 1. GIỚI THIỆU

### 1.1. Bối cảnh

Bệnh cây làm giảm năng suất và thường khó nhận biết sớm khi thiếu chuyên gia.
Ảnh lá cung cấp dấu hiệu trực quan để hỗ trợ sàng lọc. Tuy nhiên, mô hình đạt
điểm cao trên ảnh chụp trong điều kiện kiểm soát chưa chắc hoạt động tốt trên
ảnh thực tế. Vì vậy, báo cáo không chỉ quan tâm đến điểm số mà còn ghi nhận
split dữ liệu, khả năng tái lập và các lớp dễ nhầm lẫn.

### 1.2. Bài toán

**Input:** ảnh màu của một lá cây.  
**Output:** một nhãn lớp kết hợp loài cây và trạng thái bệnh, ví dụ
`Tomato___Late_blight`.  
**Loại bài toán:** phân loại ảnh đa lớp, một nhãn cho mỗi ảnh.

Phiên bản dữ liệu cục bộ được kiểm kê bởi notebook có **54.305 ảnh và 38 lớp**.
Con số này được dùng trong báo cáo vì có thể tái kiểm tra từ thư mục dữ liệu
thực tế; nguồn gốc PlantVillage trong công trình ban đầu được mô tả ở
Hughes và Salathé [1].

### 1.3. Mục tiêu

- Xây dựng pipeline PyTorch có thể tái lập từ khám phá dữ liệu đến inference.
- Thiết lập baseline SimpleCNN và mô hình pretrained ResNet-18.
- Định lượng ảnh hưởng của frozen backbone, fine-tuning và augmentation.
- Phân tích lớp bệnh dễ nhầm lẫn bằng classification report và confusion matrix.

### 1.4. Câu hỏi nghiên cứu và giả thuyết

**RQ1.** Với cùng split, seed, metric và ngân sách huấn luyện, fine-tuning
ResNet-18 có cải thiện Macro-F1 trên test so với ResNet-18 đóng băng backbone
không?

- **H1:** Fine-tuning giúp đặc trưng thích nghi với ảnh lá và đạt Macro-F1 cao hơn.
- **E1:** ResNet-18 frozen.
- **E2:** ResNet-18 fine-tuning.
- **Đối chiếu:** so sánh `val_macro_f1` để chọn cấu hình, sau đó so sánh Macro-F1
  trên test.

**RQ2.** Augmentation hình học và màu có cải thiện khả năng tổng quát hóa so
với pipeline không augmentation không?

- **H2:** Augmentation làm tăng Macro-F1 validation/test hoặc giảm lỗi ở các
  lớp có biểu hiện gần nhau.
- **E2:** ResNet-18 fine-tuning không augmentation.
- **E3:** ResNet-18 fine-tuning có augmentation.

### 1.5. Phạm vi và giới hạn

Đề tài dùng ảnh màu PlantVillage và phân loại theo nhãn thư mục. Đây là dữ liệu
được thu thập phần lớn trong điều kiện kiểm soát, không đại diện đầy đủ cho
ánh sáng, nền, góc chụp và nhiều lá trong ảnh thực tế. Mô hình là công cụ minh
họa học thuật, không phải công cụ chẩn đoán nông nghiệp.

## 2. CÔNG TRÌNH LIÊN QUAN

### 2.1. Dataset PlantVillage

Hughes và Salathé [1] giới thiệu kho hơn 50.000 ảnh lá khỏe và nhiễm bệnh, được
chọn lọc bởi chuyên gia, nhằm hỗ trợ phát triển chẩn đoán bệnh cây bằng thiết bị
di động. Đây là nền tảng dữ liệu cho đề tài và cũng là lý do phải phân biệt rõ
giữa kết quả trong điều kiện kiểm soát và khả năng triển khai ngoài thực địa.

### 2.2. Deep learning cho nhận diện bệnh cây

Mohanty, Hughes và Salathé [2] huấn luyện CNN trên 54.306 ảnh, 14 loài cây và
26 bệnh hoặc trạng thái khỏe. Công trình báo cáo 99,35% trên test giữ lại trong
điều kiện tương tự dữ liệu huấn luyện, nhưng chỉ 31,4% trên ảnh từ nguồn trực
tuyến khác điều kiện. Kết quả này là động lực để đề tài báo cáo hạn chế miền dữ
liệu, không diễn giải điểm test PlantVillage như độ chính xác ngoài thực địa.

Ferentinos [3] nghiên cứu các mô hình deep learning cho phát hiện và chẩn đoán
bệnh cây, cho thấy CNN có tiềm năng lớn trên ảnh lá. Công trình được dùng làm
tham chiếu cho cách đặt bài toán và lựa chọn metric, nhưng đề tài hiện tại
không tuyên bố tái tạo nguyên xi kiến trúc hoặc split của công trình đó.

### 2.3. Fine-tuning và kiến trúc CNN

He và cộng sự [4] đề xuất residual learning, là cơ sở của ResNet. Skip
connection giúp tối ưu mạng sâu ổn định hơn; đề tài sử dụng ResNet-18 vì cân
bằng giữa năng lực biểu diễn và tài nguyên GPU.

Tan và Le [5] đề xuất EfficientNet và compound scaling, cân bằng độ sâu, độ
rộng và độ phân giải. EfficientNet-B0 là hướng mở rộng hợp lý nếu cần đối
chứng mô hình nhẹ hơn hoặc hiệu quả tham số, nhưng thực nghiệm tối thiểu của
đề tài ưu tiên ResNet-18 để giữ phạm vi vừa sức.

Too và cộng sự [6] thực hiện nghiên cứu so sánh fine-tuning các mô hình deep
learning cho nhận diện bệnh cây. Nghiên cứu này liên quan trực tiếp đến RQ1:
không chỉ gọi một checkpoint pretrained để suy luận mà phải đo ảnh hưởng của
cách thích nghi mô hình trên cùng điều kiện đánh giá.

### 2.4. Khoảng trống và vị trí của đề tài

Các nghiên cứu trước cho thấy PlantVillage phù hợp để kiểm chứng mô hình nhưng
điểm số trong miền kiểm soát có thể không phản ánh miền thực tế. Đề tài này
đóng góp ở quy mô đồ án bằng một thiết kế có kiểm soát: baseline từ đầu, frozen
backbone, fine-tuning và augmentation; tất cả dùng cùng metadata split, seed,
metric và quy trình chọn checkpoint. Nhờ vậy, kết luận tập trung vào tác động
của từng lựa chọn thực nghiệm thay vì chỉ nêu một accuracy đơn lẻ.

## 3. DỮ LIỆU VÀ BÀI TOÁN

### 3.1. Nguồn và tổ chức dữ liệu

Nguồn dữ liệu: [PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset),
chế độ ảnh `raw/color`. Mỗi thư mục con là một lớp; tên lớp có dạng
`Plant___Disease`. Notebook `00_eda.ipynb` kiểm tra số lượng ảnh, số lớp và
phân bố lớp.

| Thành phần | Giá trị |
|---|---:|
| Số ảnh | 54.305 |
| Số lớp | 38 |
| Train | 38.013 |
| Validation | 8.146 |
| Test | 8.146 |
| Tỷ lệ | 70% / 15% / 15% |

### 3.2. Tiền xử lý

Ảnh được chuyển sang RGB, resize về 224 x 224 và chuẩn hóa theo mean/std
ImageNet. Nhánh train có augmentation khi thực nghiệm E3: lật ngang, xoay
ngẫu nhiên và thay đổi nhẹ brightness/contrast/saturation. Validation và test
chỉ resize và normalize.

### 3.3. Chống rò rỉ dữ liệu

Split được tạo một lần bằng stratification theo nhãn với seed 42. Mỗi đường
dẫn ảnh chỉ xuất hiện trong một split; validation dùng để chọn checkpoint và
test chỉ mở ở notebook đánh giá cuối. Không dùng ảnh grayscale hoặc segmented
cùng với ảnh màu trong một thực nghiệm.

## 4. PHƯƠNG PHÁP ĐỀ XUẤT

### 4.1. Baseline SimpleCNN

SimpleCNN gồm ba khối convolution + BatchNorm + ReLU + max pooling, adaptive
average pooling và một linear classifier. Mô hình được huấn luyện từ đầu, tạo
mốc tham chiếu để biết pretrained ResNet đóng góp bao nhiêu.

### 4.2. ResNet-18

ResNet-18 khởi tạo từ trọng số ImageNet và thay lớp `fc` bằng linear layer có
`num_classes = 38`. Có hai chế độ:

1. **Frozen:** đóng băng toàn bộ backbone, chỉ học classifier mới.
2. **Fine-tuning:** cho phép cập nhật toàn bộ tham số với learning rate nhỏ hơn.

### 4.3. Hàm mất mát và tối ưu

Sử dụng cross-entropy cho phân loại đa lớp và AdamW với weight decay `1e-4`.
Mô hình tốt nhất được chọn theo validation Macro-F1 và lưu checkpoint gồm
`model_state`, epoch và điểm validation. Các giá trị chính thức lấy từ notebook
và phải được ghi lại trong bảng cấu hình.

## 5. THIẾT KẾ THỰC NGHIỆM

### 5.1. Các thực nghiệm

| Mã | Mô hình | Pretrained | Backbone | Augmentation | Mục đích |
|---|---|---|---|---|---|
| E0 | SimpleCNN | Không | Huấn luyện từ đầu | Không | Baseline |
| E1 | ResNet-18 | ImageNet | Frozen | Không | RQ1 |
| E2 | ResNet-18 | ImageNet | Fine-tuning | Không | RQ1/RQ2 đối chứng |
| E3 | ResNet-18 | ImageNet | Fine-tuning | Có | RQ2 |

### 5.2. Thiết lập tái lập

| Tham số | Giá trị |
|---|---|
| Seed | 42 |
| Kích thước ảnh | 224 x 224 |
| Batch size | 32 |
| Số epoch | 10 |
| Optimizer | AdamW |
| Learning rate | E0/E1: `1e-3`; E2/E3: `1e-4` |
| Weight decay | `1e-4` |
| Thiết bị | `[ĐIỀN SAU KHI CHẠY NOTEBOOK]` |
| Phiên bản PyTorch | `[ĐIỀN SAU KHI CHẠY NOTEBOOK]` |

### 5.3. Metric

Accuracy được báo cáo để dễ diễn giải, nhưng metric chính là Macro-F1:

\[
Macro\text{-}F1 = \frac{1}{K}\sum_{k=1}^{K}F1_k
\]

Macro-F1 cho mỗi lớp trọng số ngang nhau, phù hợp khi số lượng ảnh giữa các
lớp không hoàn toàn đồng đều. Confusion matrix dùng để phân tích lỗi từng lớp.

## 6. KẾT QUẢ

### 6.1. Kết quả validation

| Thực nghiệm | Best epoch | Val accuracy | Val Macro-F1 |
|---|---:|---:|---:|
| E0 SimpleCNN | `[điền]` | `[điền]` | `[điền]` |
| E1 ResNet frozen | `[điền]` | `[điền]` | `[điền]` |
| E2 ResNet fine-tuning | `[điền]` | `[điền]` | `[điền]` |
| E3 ResNet + augmentation | `[điền]` | `[điền]` | `[điền]` |

Hình cần chèn: `outputs/validation_f1_comparison.png`.

### 6.2. Kết quả test

| Thực nghiệm | Test accuracy | Test Macro-F1 | Ghi chú |
|---|---:|---:|---|
| E0 SimpleCNN | `[điền]` | `[điền]` | Baseline |
| E1 ResNet frozen | `[điền]` | `[điền]` | So sánh RQ1 |
| E2 ResNet fine-tuning | `[điền]` | `[điền]` | So sánh RQ1/RQ2 |
| E3 ResNet + augmentation | `[điền]` | `[điền]` | So sánh RQ2 |

### 6.3. Ablation studies

Đối với RQ1, mức thay đổi được tính bằng:

\[
\Delta F1_{RQ1} = MacroF1(E2) - MacroF1(E1)
\]

Đối với RQ2:

\[
\Delta F1_{RQ2} = MacroF1(E3) - MacroF1(E2)
\]

| Câu hỏi | So sánh | Kết quả | Kết luận |
|---|---|---:|---|
| RQ1 | E2 - E1 | `[điền]` | `[ủng hộ/không ủng hộ H1]` |
| RQ2 | E3 - E2 | `[điền]` | `[ủng hộ/không ủng hộ H2]` |

## 7. PHÂN TÍCH LỖI VÀ THẢO LUẬN

### 7.1. Confusion matrix

Chèn `outputs/confusion_matrix_test.png`. Tập trung vào ba cặp có số nhầm lẫn
cao nhất:

1. `[lớp A]` ↔ `[lớp B]`: `[giải thích đặc điểm hình ảnh tương tự]`.
2. `[lớp C]` ↔ `[lớp D]`: `[giải thích]`.
3. `[lớp E]` ↔ `[lớp F]`: `[giải thích]`.

### 7.2. Ví dụ dự đoán sai

Chọn ít nhất 6 ảnh sai từ test, ghi nhãn thật, nhãn dự đoán, xác suất cao nhất
và nhận xét về nền, góc chụp, mức độ bệnh hoặc biểu hiện tương tự. Không chọn
ảnh từ train/validation để minh họa lỗi test.

### 7.3. Trả lời câu hỏi nghiên cứu

- **RQ1:** `[Kết luận dựa trên Macro-F1 E1/E2 và độ ổn định đường validation.]`
- **RQ2:** `[Kết luận dựa trên Macro-F1 E2/E3 và các lớp bị ảnh hưởng.]`

Kết quả chỉ cho phép kết luận trong miền dữ liệu PlantVillage và thiết lập
thực nghiệm này; không suy rộng trực tiếp thành độ chính xác ngoài đồng ruộng.

### 7.4. Hạn chế

- Ảnh PlantVillage có điều kiện chụp tương đối kiểm soát.
- Chưa có external test set đủ đại diện cho nhiều vùng, mùa vụ và thiết bị.
- Chưa dùng calibration, test-time augmentation hoặc explainability nâng cao.
- Thời gian/nguồn lực giới hạn số kiến trúc và số lần lặp seed.

## 8. KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

### 8.1. Kết luận

Đề tài xây dựng pipeline phân loại bệnh lá cây bằng PyTorch với baseline
SimpleCNN và ResNet-18 pretrained. Thiết kế E1–E3 giúp tách riêng tác động của
fine-tuning và augmentation, trong khi Macro-F1 và confusion matrix giúp đánh
giá công bằng hơn accuracy đơn thuần. Kết luận định lượng cuối cùng được cập
nhật sau khi chạy đầy đủ notebook và điền bảng kết quả ở Chương 6.

### 8.2. Hướng phát triển

- Đánh giá trên PlantDoc hoặc ảnh hiện trường độc lập.
- So sánh thêm EfficientNet-B0 theo ngân sách tham số và thời gian.
- Dùng repeated seeds và khoảng tin cậy.
- Bổ sung Grad-CAM để kiểm tra mô hình tập trung vào vùng tổn thương.
- Đóng gói inference bằng Streamlit/Gradio với cảnh báo mô hình chỉ hỗ trợ
  nghiên cứu.

## 9. TÀI LIỆU THAM KHẢO

[1] D. P. Hughes and M. Salathé, “An open access repository of images on plant
health to enable the development of mobile disease diagnostics,” arXiv:1511.08060,
2015/2016. https://arxiv.org/abs/1511.08060

[2] S. P. Mohanty, D. P. Hughes, and M. Salathé, “Using deep learning for
image-based plant disease detection,” arXiv:1604.03169, 2016.
https://arxiv.org/abs/1604.03169

[3] K. P. Ferentinos, “Deep learning models for plant disease detection and
diagnosis,” *Computers and Electronics in Agriculture*, vol. 145, pp. 311–318,
2018. https://doi.org/10.1016/j.compag.2018.01.009

[4] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image
recognition,” arXiv:1512.03385, 2015. https://arxiv.org/abs/1512.03385

[5] M. Tan and Q. V. Le, “EfficientNet: Rethinking model scaling for convolutional
neural networks,” *Proceedings of ICML*, 2019, arXiv:1905.11946.
https://arxiv.org/abs/1905.11946

[6] E. D. Too, L. Yujian, S. Njuki, and L. Yingchun, “A comparative study of
fine-tuning deep learning models for plant disease identification,” *Computers
and Electronics in Agriculture*, vol. 161, pp. 272–279, 2019.
https://doi.org/10.1016/j.compag.2018.03.032

## 10. PHỤ LỤC

### Phụ lục A. Liên kết với notebook

| Nội dung báo cáo | Notebook |
|---|---|
| EDA và phân bố lớp | `notebooks/00_eda.ipynb` |
| Split và chống leakage | `notebooks/01_prepare_data.ipynb` |
| Huấn luyện E0–E3 | `notebooks/02_train_experiments.ipynb` |
| Test và confusion matrix | `notebooks/03_evaluate_and_error_analysis.ipynb` |
| Inference | `notebooks/04_inference_demo.ipynb` |

### Phụ lục B. Tái lập

```powershell
cd C:\Users\kelac\Downloads\HOCTAP\deeplearning\cuoiky\DUAN\plant_disease_project
pip install -r requirements.txt
jupyter lab
```

### Phụ lục C. Ghi nhận hỗ trợ

Ghi rõ các nguồn mã nguồn, pretrained checkpoint, dataset, thư viện và phần hỗ
trợ AI đã sử dụng theo quy định môn học.

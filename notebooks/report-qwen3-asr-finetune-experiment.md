# Báo cáo: Fine-tune Qwen3-ASR-0.6B nhận dạng tiếng Việt trong môi trường ồn (v4)

## 1. Tóm tắt
- Mô hình sau fine-tune (LoRA + loss nhất quán sạch–ồn) **tốt hơn zero-shot ở cả 9 điều kiện** của tập test FLEURS. Khác biệt có ý nghĩa thống kê ở **8/9** điều kiện (paired bootstrap, 95%).
- **WER trung bình 9 điều kiện giảm từ 24,67% xuống 17,61%**, tức giảm 29% tương đối. Cải thiện lớn nhất ở babble mạnh: 0 dB từ 109,59 xuống 63,40; 5 dB từ 30,65 xuống 19,31. Audio sạch và traffic giảm 0,4–1,0 điểm.
- Lần chạy trước (v3) kém zero-shot ở audio sạch và traffic. v4 đã khắc phục điều này, chủ yếu nhờ lịch train 3 epoch và việc chọn checkpoint trên toàn bộ tập dev.

## 2. Thiết lập thí nghiệm

### Dữ liệu (`asr-noisy-vi`, phần FLEURS tiếng Việt)
| Tập | Kích thước | Ghi chú |
|---|---|---|
| Train | 2990 câu, 9,0 giờ | Trộn ồn online theo `train_schedule.csv`: 25% câu sạch, còn lại traffic hoặc babble với SNR ngẫu nhiên đều trong 0–20 dB; mỗi epoch nhận tiếng ồn mới |
| Dev | 359 file | Mỗi câu một điều kiện: khoảng 25% sạch, còn lại traffic hoặc babble ở 20/10/5/0 dB |
| Test | 850 câu × 9 điều kiện = 7650 file, 26,4 giờ | Sạch, cộng traffic và babble ở 20 / 10 / 5 / 0 dB |

- **Traffic**: clip từ FSD50K.
- **Babble**: 3–5 giọng trộn lại, lấy từ MUSAN `librivox` (chủ yếu là tiếng Anh).
- Clip tiếng ồn của tập train và tập test không trùng nhau.

### Mô hình và cách đánh giá
- **Mô hình**: `Qwen/Qwen3-ASR-0.6B-hf`, gồm speech encoder 18 lớp và LM Qwen3 28 lớp, tổng khoảng 0,8 tỉ tham số.
- **Môi trường**: 1 GPU T4, PyTorch 2.11, transformers 5.20.2.
- **Decode**: greedy, ép ngôn ngữ tiếng Việt, tối đa 400 token.
- **Chấm điểm**: WER và CER tính trên toàn tập, sau khi chuẩn hóa cả nhãn lẫn output (chữ thường, bỏ dấu câu, đọc số thành chữ, thống nhất vị trí dấu thanh). Có khoảng tin cậy 95% bằng bootstrap, và kiểm định paired bootstrap giữa hai mô hình.

## 3. Phương pháp fine-tune
- **LoRA** trên mọi projection của attention và MLP:
  - speech encoder: r = 32, α = 64;
  - LM: r = 8, α = 16;
  - tổng cộng 14,3 triệu tham số được train, chiếm 1,8% mô hình.
- **Loss nhất quán sạch–ồn**:
  `loss = CE + 1,0 × (1 − cos(encoder có LoRA trên audio ồn, encoder gốc trên audio sạch của cùng câu))`.
  Loss được tính theo từng khung trên đầu ra của encoder + projector. Encoder gốc chạy không cần gradient.
- **Train**:
  - AdamW, warmup 10%, sau đó LR giảm tuyến tính về 0 trong 3 epoch;
  - 16 câu mỗi lần cập nhật (batch 4 × gradient accumulation 4);
  - fp16 autocast, gradient checkpointing, batch gom theo độ dài;
  - bộ nhớ GPU đỉnh 5,5 GB.
- **Chọn mô hình**:
  - 4 ứng viên, mỗi ứng viên train đủ 3 epoch và giữ checkpoint có dev WER thấp nhất;
  - chọn ứng viên có dev WER thấp nhất;
  - tập test chỉ được dùng một lần, cho ứng viên đã được chọn.

## 4. Chọn mô hình trên tập dev (WER %, zero-shot = 23,22)

| Ứng viên | Tham số train | Epoch 1 | Epoch 2 | Epoch 3 | Tốt nhất |
|---|---|---|---|---|---|
| **encoder r32 / LM r8 + consistency, LR 1e-4** | 14,3M | 15,47 | 15,61 | **15,22** | **15,22** (được chọn) |
| encoder r32 / LM r8 + consistency, LR 5e-5 | 14,3M | 15,68 | 15,33 | 15,59 | 15,33 |
| chỉ encoder r32 + consistency, LR 1e-4 | 9,3M | 19,76 | 18,20 | 15,59 | 15,59 |
| encoder r32 / LM r8 + consistency, LR 2e-4 | 14,3M | 16,56 | 15,97 | 15,63 | 15,63 |

Bốn ứng viên chỉ chênh nhau 0,41 điểm. Trên 359 file dev, mức chênh này nằm trong nhiễu, nên có thể coi các công thức là ngang nhau. Riêng ứng viên "chỉ encoder" vẫn còn đang giảm ở epoch 3.

## 5. Kết quả trên tập test (850 câu mỗi điều kiện)

### WER (%)
| Điều kiện | Zero-shot | Fine-tune | Chênh lệch | Tương đối | Khoảng tin cậy 95% của chênh lệch | Có ý nghĩa |
|---|---|---|---|---|---|---|
| clean | 8,33 | **7,90** | −0,43 | −5,2% | [−0,77; −0,06] | ✅ |
| traffic 20 dB | 8,81 | **8,13** | −0,68 | −7,7% | [−1,08; −0,27] | ✅ |
| traffic 10 dB | 10,09 | **9,51** | −0,59 | −5,8% | [−1,01; −0,15] | ✅ |
| traffic 5 dB | 12,87 | **12,04** | −0,83 | −6,4% | [−1,30; −0,32] | ✅ |
| traffic 0 dB | 19,73 | **18,78** | −0,95 | −4,8% | [−1,58; −0,31] | ✅ |
| babble 20 dB | 8,66 | **8,31** | −0,35 | −4,0% | [−0,73; +0,08] | ❌ |
| babble 10 dB | 13,28 | **11,09** | −2,19 | −16,5% | [−4,49; −0,79] | ✅ |
| babble 5 dB | 30,65 | **19,31** | −11,34 | −37,0% | [−16,71; −7,41] | ✅ |
| babble 0 dB | 109,59 | **63,40** | −46,19 | −42,1% | [−58,69; −34,40] | ✅ |
| **Trung bình 9 điều kiện** | 24,67 | **17,61** | −7,06 | −28,6% | | |
| Trung bình, không tính babble 0 dB | 14,05 | **11,88** | −2,17 | −15,4% | | |

### CER (%)
| Điều kiện | clean | traffic 20 / 10 / 5 / 0 dB | babble 20 / 10 / 5 / 0 dB |
|---|---|---|---|
| Zero-shot | 4,39 | 4,73 / 5,54 / 7,49 / 12,40 | 4,62 / 8,09 / 21,40 / 87,82 |
| Fine-tune | 4,59 | 4,72 / 5,69 / 7,54 / 12,35 | 4,84 / 6,81 / 12,89 / 47,51 |

## 6. So sánh với lần chạy trước (v3)
| WER (%) | clean | traffic 20 / 10 / 5 / 0 dB | babble 20 / 10 / 5 / 0 dB |
|---|---|---|---|
| Zero-shot | 8,33 | 8,81 / 10,09 / 12,87 / 19,73 | 8,66 / 13,28 / 30,65 / 109,59 |
| v3 | 8,99 | 9,41 / 11,10 / 14,36 / 23,29 | 9,51 / 12,88 / 22,50 / 95,18 |
| **v4** | **7,90** | **8,13 / 9,51 / 12,04 / 18,78** | **8,31 / 11,09 / 19,31 / 63,40** |

v3 dùng công thức giống ứng viên thứ tư ở trên (LR 2e-4), nhưng có ba điểm khác:
- lịch LR dài 10 epoch, có early stopping;
- chọn checkpoint trên chỉ 100 file dev;
- checkpoint tốt nhất ở epoch 2, khi LR vẫn còn 89% mức cao nhất.

Ở v3, train loss giảm từ 0,29 xuống 0,08 trong khi dev WER đứng yên: mô hình bị overfit. Ở v4, cùng công thức đó, khi train theo lịch 3 epoch và chọn trên toàn bộ dev, đạt 15,63, gần bằng ứng viên tốt nhất. Như vậy, **lịch train và cách chọn checkpoint mới là yếu tố quyết định, chứ không phải learning rate**.

## 7. Thảo luận
- **Cải thiện tập trung ở babble mạnh.** Mô hình zero-shot hay "bịa" hoặc lặp câu khi bị giọng nói khác chen vào: ở babble 0 dB, lỗi chèn chiếm 35,2 trong tổng số 109,6 điểm WER, và có câu WER trên 1000%. Fine-tune giảm mạnh hiện tượng này nhưng chưa loại bỏ hẳn; trong các câu sai nhiều nhất của mô hình fine-tune, vẫn có câu ở babble 0 dB có WER 1123%.
- **Có một "sàn" lỗi khoảng 8% không liên quan tới tiếng ồn.** Audio sạch 7,90, traffic 20 dB 8,13, babble 20 dB 8,31, gần như bằng nhau. Các ví dụ từ bước kiểm tra zero-shot cho thấy loại lỗi này:
  - tên riêng nước ngoài: "springboks" thành "Spring Boot", "t rex" thành "tigre";
  - nhầm "tr"/"ch": "trung hoa" thành "chung hoa";
  - quy ước chữ viết trong nhãn: "hjr ba", "tcn".

  Fine-tune để chống ồn không nhắm vào loại lỗi này.
- **Phân bố lỗi còn lại của mô hình fine-tune** (tỉ lệ theo WER, vì mọi điều kiện dùng chung 850 câu):

  | Điều kiện | Tỉ lệ trong tổng số lỗi |
  |---|---|
  | babble 0 dB | 40% |
  | babble 5 dB | 12% |
  | traffic 0 dB | 12% |
  | 6 điều kiện còn lại | 5–8% mỗi điều kiện |

- **CER ở audio sạch và traffic gần như không đổi** (sạch 4,39 → 4,59; traffic 0 dB 12,40 → 12,35). Cải thiện ở hai nhóm điều kiện này chủ yếu ở mức từ, và còn nhỏ.

## 8. Hạn chế
- Mới train và test trên FLEURS (9 giờ train). Chưa đánh giá trên VIVOS hay VietMed.
- Cả 4 ứng viên đều dùng consistency loss, nên **v4 chưa chứng minh được tác dụng riêng của nó**. Bằng chứng hiện có chỉ là sweep của v3 trên 100 file dev, quá yếu.
- Tập dev có 359 file, nên không phân biệt được các ứng viên chênh nhau dưới khoảng 0,5 điểm.
- CER chưa có khoảng tin cậy.
- Babble được tạo từ tiếng nói tiếng Anh, chưa phản ánh môi trường có tiếng nói tiếng Việt ở nền.
- Nếu so sánh với một mô hình khác (ví dụ PhoWhisper), cần dùng cùng quy trình (số epoch, cách chọn checkpoint trên dev). Consistency loss là một phương pháp mới, nên báo cáo thành một dòng riêng.

## 9. Hướng tiếp theo
1. **Phân tích lỗi**: phân loại các lỗi còn lại (tên riêng, số và viết tắt, lặp câu, nhầm tr/ch) theo từng điều kiện.
2. **Chặn vòng lặp khi decode** (`CAP_TOKENS_BY_DURATION`): không cần train lại, nhắm vào babble 0–5 dB, nơi chiếm khoảng 50% số lỗi. Đây là thay đổi luật decode, nên phải áp dụng giống nhau cho mọi mô hình được so sánh.
3. **Ablation không dùng consistency** theo đúng quy trình v4, thêm khoảng 45 phút.
4. **Thêm dữ liệu train** (VIVOS, VietMed).
5. **Thử Qwen3-ASR-1.7B**: ước lượng khoảng 6–7 giờ trên T4 cho 1 ứng viên, cần nạp base ở float16.

## 10. Thông tin tái lập
- **Notebook**: `multimedia-qwen-evaluate-v4.ipynb`. Bản đã chạy kèm output: `qwen-ars-v4.ipynb`.
- **Code**: repo `asr-noisy-vi` tại commit `83fad87`, seed 42.
- **Thời gian chạy trên T4**:

  | Bước | Thời gian |
  |---|---|
  | Zero-shot test | 60,6 phút |
  | Train 4 ứng viên | 161,6 phút |
  | Test sau fine-tune | 35,6 phút |
  | **Tổng** | **khoảng 4,3 giờ** |

- **File kết quả**:
  - `predictions_qwen3-asr-0.6b_zero-shot.csv`
  - `predictions_qwen3-asr-0.6b_ft-noisy-enc32-lm8-cons-lr1e-4.csv`
  - `lora_qwen3-asr-0.6b_ft-noisy-enc32-lm8-cons-lr1e-4/`
  - `history_*.csv`, `candidates_qwen3-asr-0.6b_ft-noisy.csv`, `run_qwen3-asr-0.6b_ft-noisy.json`

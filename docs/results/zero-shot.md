# Kết quả zero-shot

Kết quả của notebook `02_zero_shot`: PhoWhisper-small và Qwen3-ASR-0.6B, chưa fine-tune,
chạy trên toàn bộ test plan. Tài liệu gồm số liệu, phân tích nguyên nhân, đánh giá và kết
luận dùng cho báo cáo. Những chỗ ghi **Giả thuyết** là cách giải thích chưa được kiểm chứng;
cần kiểm tra trước khi viết thành kết luận.

| Mục | Giá trị |
|---|---|
| Code | commit `15f9667` (`feature/dataprep-evaluation`), bundle không có thay đổi chưa commit |
| Dataset | `asr-noisy-vi` v1, seed 42 (`DATASET.json` được dựng lại trong notebook) |
| Model | `vinai/PhoWhisper-small` @ `a86b604`, `Qwen/Qwen3-ASR-0.6B` @ `5eb1441` |
| Phần cứng | Kaggle, Tesla T4, float16, `transformers` 4.57.6, batch 16 |
| Decode | Greedy, ép ngôn ngữ tiếng Việt; `max_new_tokens` 440 (PhoWhisper), 512 (Qwen3-ASR) |
| Test plan | 2.610 câu × 9 điều kiện = 23.490 file mỗi model |
| Thời gian | PhoWhisper 111,5 phút, Qwen3-ASR 84,1 phút; xong ngày 2026-11-02 |
| Chấm lại | Ngày 2026-11-02, cùng predictions (resume), code `77db14d` + `evaluation/failures.py` chưa commit: thêm `failures_zero-shot.csv` và `fleurs_by_digit.csv` |
| Số liệu gốc | [`data/zero-shot/`](data/zero-shot/), xem [mục 11](#11-file-kết-quả) |

## Mục lục

1. [Tóm tắt](#1-tóm-tắt)
2. [Thiết lập và cách đọc số](#2-thiết-lập-và-cách-đọc-số)
3. [Tổng quan, gộp ba tập](#3-tổng-quan-gộp-ba-tập)
4. [Theo cường độ nhiễu](#4-theo-cường-độ-nhiễu)
5. [Theo loại nhiễu](#5-theo-loại-nhiễu)
6. [Theo tập dữ liệu](#6-theo-tập-dữ-liệu)
7. [So sánh hai model](#7-so-sánh-hai-model)
8. [Các kiểu hỏng ở babble 0 dB](#8-các-kiểu-hỏng-ở-babble-0-db)
9. [Đánh giá](#9-đánh-giá)
10. [Hạn chế và bước tiếp theo](#10-hạn-chế-và-bước-tiếp-theo)
11. [File kết quả](#11-file-kết-quả)
12. [Phụ lục: bảng số đầy đủ](#12-phụ-lục-bảng-số-đầy-đủ)

## 1. Tóm tắt

- **20 dB gần như vô hại**: WER tăng tối đa 0,7 điểm, với cả hai loại nhiễu và cả hai
  model.
- **Ở cùng SNR, babble khó hơn traffic hơn 5 dB**: babble 10 dB ≈ traffic 5 dB, và
  babble 5 dB còn tệ hơn traffic 0 dB.
- **Babble 0 dB làm cả hai model sụp**: WER gấp 6 lần (PhoWhisper) đến 8 lần (Qwen3-ASR)
  bản sạch, trong khi traffic 0 dB chỉ khoảng 2 lần.
- **Lỗi tăng phi tuyến** khi SNR giảm. Với babble, điểm gãy nằm giữa 10 và 5 dB.
- **Theo cơ chế**: traffic làm tăng lỗi thay thế (nghe sai âm). Babble 0 dB thêm ba kiểu
  hỏng: model lặp một cụm từ đến khi hết giới hạn token, model chỉ ra vài từ, và model chép
  giọng nói nền tiếng Anh bằng tiếng Anh.
- **Theo tập dữ liệu**: VIVOS dễ nhất và bền nhất; VietMed khó nhất khi sạch và khi có
  traffic; FLEURS bị babble 0 dB làm hỏng nặng nhất.
- **Hai model**: Qwen3-ASR tốt hơn trên FLEURS và VietMed, và chịu traffic tốt hơn.
  PhoWhisper tốt hơn trên VIVOS khoảng 4 điểm ở mọi điều kiện, và tốt hơn trên FLEURS ở
  babble 0 dB.
- **Điểm yếu của Qwen3-ASR là hỏng cả câu, không phải nghe kém**: ở FLEURS babble 0 dB,
  15,6% số câu của nó bị lặp, bỏ cụt hoặc chép giọng nền, đóng góp 54 điểm WER. Bỏ các câu
  hỏng, khoảng cách với PhoWhisper giảm từ +36,4 xuống +2,3 điểm; ở FLEURS babble 5–10 dB và
  VietMed babble 0 dB, Qwen3-ASR còn tốt hơn.
- **Câu có chữ số** (20% câu FLEURS) có WER sạch cao hơn 5,1 điểm (PhoWhisper) và 2,7 điểm
  (Qwen3-ASR). Khoảng chênh gần như không đổi theo traffic, nên nó đến từ cách đọc và viết
  số, không phải từ nhiễu.
- **Kết quả tái lập được**: lần chạy này cho đúng các số của lần chạy trước ở cùng commit.

## 2. Thiết lập và cách đọc số

Mỗi câu test được chấm ở 9 điều kiện: bản sạch, và SNR 20, 10, 5, 0 dB cho mỗi loại nhiễu.
Một câu dùng **cùng một đoạn nhiễu ở cả bốn mức SNR**, nên khác biệt giữa các mức chỉ đến
từ SNR. Hai model dùng chung test plan. Transcript tham chiếu và output của model được chuẩn
hoá giống nhau bằng `normalize_text` trước khi chấm, nên dấu câu và chữ hoa không bị tính
là lỗi.

### Ba tập test

| Tập | Số câu | Số từ tham chiếu | Từ / câu | Nội dung | Cần lưu ý |
|---|---:|---:|---:|---|---|
| FLEURS `vi_vn` | 850 | 26.123 | 30,7 | Đọc câu Wikipedia, 16 kHz | Tên riêng, từ mượn, khoảng 23% câu có chữ số; câu dài |
| VIVOS | 760 | 7.722 | 10,2 | Đọc trong phòng yên tĩnh, 16 kHz | Câu ngắn, từ vựng thông dụng, thu âm sạch |
| VietMed | 1.000 | 22.649 | 22,6 | Hội thoại y khoa, điện thoại, **8 kHz** | Lấy mẫu con có phân tầng theo điều kiện thu âm; nhiễu được giới hạn băng tần còn 4 kHz cho cùng kênh với tiếng nói |

### Nhiễu

- **Traffic**: clip từ FSD50K (xe cộ, đường phố).
- **Babble**: 3–5 giọng từ MUSAN `speech/librivox`, cộng lại bằng `make_babble`. Phần lớn
  là giọng đọc sách tiếng Anh.

### Hai model

| | PhoWhisper-small | Qwen3-ASR-0.6B |
|---|---|---|
| Kiến trúc | Whisper small: encoder–decoder Transformer, khoảng 244M tham số | Audio encoder nối với decoder là LLM Qwen3, khoảng 0,6B tham số |
| Dữ liệu | Whisper đa ngôn ngữ, fine-tune riêng cho tiếng Việt | Đa ngôn ngữ, quy mô lớn; tiếng Việt là một trong nhiều ngôn ngữ |
| Ép ngôn ngữ | Token ngôn ngữ `vi` ở đầu chuỗi decode | Tham số `language="Vietnamese"` của `qwen-asr` |

### WER

`WER = (S + D + I) / N × 100`, với N là số từ trong transcript tham chiếu, S là số từ bị
nhận sai thành từ khác (thay thế), D là số từ bị bỏ sót (xoá), I là số từ thừa do model
thêm vào (chèn). CER tính tương tự trên ký tự.

WER ở đây là **WER toàn corpus**: cộng lỗi của mọi câu rồi chia cho tổng số từ tham chiếu,
không lấy trung bình WER của từng câu. Câu dài có trọng số lớn hơn, và một câu bịa rất dài
có thể kéo cả điều kiện lên. Lỗi chèn không bị giới hạn nên **WER có thể vượt 100%**.

### SNR

SNR được tính trên toàn bộ câu, gồm cả các khoảng im lặng. Ở cùng SNR, traffic và babble có
cùng năng lượng, nên so hai loại nhiễu ở cùng một mức SNR là so sánh công bằng.

| SNR | Công suất nhiễu / tiếng nói | Biên độ nhiễu / tiếng nói |
|---|---:|---:|
| 20 dB | 1/100 | 1/10 |
| 10 dB | 1/10 | ≈ 0,32 |
| 5 dB | ≈ 1/3 | ≈ 0,56 |
| 0 dB | 1 | 1 |

## 3. Tổng quan, gộp ba tập

WER (%), tính trên toàn corpus của cả ba tập (56.494 từ tham chiếu). **Δ sạch** là số điểm
WER tăng so với bản sạch; **× sạch** là tỉ số so với bản sạch.

| Điều kiện | PhoWhisper | Qwen3-ASR | Δ sạch Pho | Δ sạch Qwen | × sạch Pho | × sạch Qwen |
|---|---:|---:|---:|---:|---:|---:|
| clean | 12,67 | 11,58 | – | – | 1,00 | 1,00 |
| traffic 20 dB | 12,92 | 12,01 | +0,25 | +0,43 | 1,02 | 1,04 |
| traffic 10 dB | 14,73 | 13,24 | +2,06 | +1,66 | 1,16 | 1,14 |
| traffic 5 dB | 17,69 | 15,65 | +5,02 | +4,07 | 1,40 | 1,35 |
| traffic 0 dB | 25,05 | 22,09 | +12,38 | +10,51 | 1,98 | 1,91 |
| babble 20 dB | 13,33 | 12,18 | +0,66 | +0,60 | 1,05 | 1,05 |
| babble 10 dB | 18,27 | 16,78 | +5,60 | +5,20 | 1,44 | 1,45 |
| babble 5 dB | 31,33 | 29,81 | +18,66 | +18,23 | 2,47 | 2,57 |
| babble 0 dB | 75,82 | 93,48 | +63,15 | +81,90 | 5,98 | 8,07 |

![WER theo SNR, từng tập và từng loại nhiễu](figures/wer_vs_snr.png)

*WER theo SNR, kèm khoảng tin cậy bootstrap 95%. Mỗi hàng dùng chung một thang y: hàng trên
là traffic, hàng dưới là babble. Trục x đi từ sạch đến 0 dB, càng sang phải càng nhiễu.*

Gộp ba tập, **Qwen3-ASR thấp hơn PhoWhisper ở 8/9 điều kiện**, chênh 1–3 điểm. Ngoại lệ
duy nhất là babble 0 dB, nơi Qwen3-ASR cao hơn 17,7 điểm. Số gộp che đi khác biệt lớn giữa
các tập, nên phần so sánh hai model dựa trên số theo từng tập ([mục 7](#7-so-sánh-hai-model)).

## 4. Theo cường độ nhiễu

- **20 dB**: tăng +0,25 đến +0,66 điểm. Model đã gặp nhiều audio có nhiễu nền nhẹ khi huấn
  luyện, nên mức này nằm trong vùng chúng quen.
- **10 dB**: bắt đầu thấy rõ, chủ yếu với babble (+5,6 và +5,2), trong khi traffic chỉ
  +1,7 đến +2,1.
- **5 dB**: điểm gãy của babble (+18,7 và +18,2). Traffic ở mức +4 đến +5.
- **0 dB**: traffic vẫn dùng được (WER 22–25%, khoảng 1 từ sai trên 4–5 từ); babble thì
  không (76–93%).

**Vì sao lỗi tăng phi tuyến.** SNR là thang log: từ 10 xuống 5 dB, công suất nhiễu tăng
khoảng 3,2 lần, và từ 5 xuống 0 dB lại tăng thêm 3,2 lần. Khi tiếng nói còn trội hơn nhiễu
trong các dải tần mang formant và phụ âm (khoảng 300–4.000 Hz), encoder vẫn giữ được đặc
trưng âm vị và mô hình ngôn ngữ của decoder sửa được phần còn lại. Khi nhiễu lấn át ở nhiều
dải cùng lúc, đặc trưng bị mất và lỗi tăng vọt. Vì vậy đường WER phẳng ở SNR cao và dốc ở
SNR thấp.

Ở 0 dB, sự sụp của babble còn có một nguyên nhân khác hẳn: model không chỉ nghe sai mà còn
**chép nhầm người nói** hoặc **sinh văn bản không có trong audio**
([mục 8](#8-các-kiểu-hỏng-ở-babble-0-db)). Các kiểu hỏng này gần như không xuất hiện ở
babble 5 dB (dưới 1% số câu) nhưng lên tới 6,8% số câu của Qwen3-ASR ở 0 dB, nên WER nhảy từ
khoảng 30% lên 76–93%.

## 5. Theo loại nhiễu

| Cùng mức khó | PhoWhisper | Qwen3-ASR |
|---|---:|---:|
| babble 10 dB / traffic 5 dB | 18,27 / 17,69 | 16,78 / 15,65 |
| babble 5 dB / traffic 0 dB | 31,33 / 25,05 | 29,81 / 22,09 |

Ở cùng SNR, babble **khó hơn traffic hơn 5 dB**.

**Vì sao babble khó.**

- Babble là tiếng người: nằm cùng dải tần với tiếng nói chính (formant, khoảng
  300–3.400 Hz) và có cùng nhịp điều biến khoảng 4 Hz (nhịp âm tiết). Cách nào giữ được
  tiếng nói chính thì cũng giữ luôn babble.
- Babble gây **che thông tin** (*informational masking*), không chỉ che năng lượng: nó mang
  nội dung ngôn ngữ cạnh tranh. Model được huấn luyện để chép mọi tiếng nói nên không biết
  giọng nào là giọng cần chép. Dữ liệu xác nhận điều này: ở babble 0 dB, Qwen3-ASR chép lại
  câu tiếng Anh của giọng nền trong 41 câu ([mục 8](#8-các-kiểu-hỏng-ở-babble-0-db)).
- 3–5 giọng là ít. Từng giọng vẫn nghe rõ và có thể bị chép theo. Babble rất đông người thì
  gần với nhiễu ổn định hơn.

**Vì sao traffic dễ hơn.**

- Năng lượng dồn ở tần số thấp (động cơ, tiếng ầm) và khá ổn định theo thời gian. Ở cùng
  SNR tổng, dải 1–4 kHz chứa phụ âm bị che ít hơn.
- Nó không giống tiếng người nên model không biến nó thành chữ: lỗi chèn gần như không đổi,
  và không có câu nào bị lặp hay bị chép sang tiếng Anh ở traffic 0 dB.
- Dữ liệu huấn luyện quy mô web (Whisper, Qwen) có rất nhiều nhiễu nền loại này.

![Phân rã lỗi theo loại, gộp ba tập](figures/error_kinds.png)

Phân rã lỗi theo loại (% số từ tham chiếu), gộp ba tập. S: thay thế, D: xoá, I: chèn.

| Điều kiện | S Pho | S Qwen | D Pho | D Qwen | I Pho | I Qwen |
|---|---:|---:|---:|---:|---:|---:|
| clean | 7,49 | 7,53 | 1,96 | 1,98 | 3,21 | 2,08 |
| traffic 20 dB | 7,74 | 7,91 | 2,03 | 1,99 | 3,15 | 2,11 |
| traffic 10 dB | 9,33 | 9,05 | 2,15 | 2,11 | 3,24 | 2,08 |
| traffic 5 dB | 12,00 | 11,15 | 2,42 | 2,41 | 3,26 | 2,10 |
| traffic 0 dB | 18,39 | 16,44 | 3,24 | 3,46 | 3,41 | 2,19 |
| babble 20 dB | 8,08 | 8,10 | 2,06 | 2,05 | 3,18 | 2,03 |
| babble 10 dB | 11,99 | 11,33 | 2,46 | 2,47 | 3,81 | 2,98 |
| babble 5 dB | 22,15 | 20,73 | 3,81 | 4,25 | 5,37 | 4,83 |
| babble 0 dB | 52,85 | 51,83 | 8,16 | 14,50 | 14,81 | 27,16 |

- **Traffic làm tăng lỗi thay thế** (PhoWhisper 7,5 → 18,4). Lỗi xoá tăng nhẹ
  (2,0 → 3,2), lỗi chèn gần như không đổi. Model vẫn nghe ra là có tiếng nói, nhưng nghe sai
  âm.
- **Babble 0 dB làm tăng cả ba loại lỗi**, nhiều nhất là lỗi chèn: 14,8 (PhoWhisper) và
  27,2 (Qwen3-ASR). Lỗi thay thế của hai model gần như bằng nhau (≈ 52); chênh lệch WER đến
  từ lỗi chèn và lỗi xoá, tức là từ các kiểu hỏng ở [mục 8](#8-các-kiểu-hỏng-ở-babble-0-db),
  không phải từ việc nghe sai âm.

## 6. Theo tập dữ liệu

Tỉ số giữa WER ở 0 dB và WER sạch:

| Tập | WER sạch Pho / Qwen | traffic 0 dB Pho / Qwen | babble 0 dB Pho / Qwen |
|---|---:|---:|---:|
| FLEURS | 9,33 / 8,39 | ×2,2 / ×2,3 | ×8,9 / ×14,3 |
| VIVOS | 6,20 / 10,10 | ×2,2 / ×1,7 | ×8,0 / ×5,2 |
| VietMed | 18,73 / 15,77 | ×1,8 / ×1,7 | ×4,1 / ×4,9 |

Phân rã lỗi theo tập (% số từ tham chiếu), S / D / I:

| Tập | Model | clean | traffic 0 dB | babble 0 dB |
|---|---|---|---|---|
| FLEURS | PhoWhisper | 6,9 / 0,7 / 1,8 | 17,1 / 1,4 / 2,4 | 57,2 / 7,0 / 19,3 |
| FLEURS | Qwen3-ASR | 6,9 / 0,6 / 0,9 | 16,2 / 1,8 / 1,2 | 56,3 / 17,8 / 45,8 |
| VIVOS | PhoWhisper | 5,7 / 0,3 / 0,2 | 13,0 / 0,4 / 0,4 | 36,8 / 1,5 / 11,5 |
| VIVOS | Qwen3-ASR | 9,4 / 0,4 / 0,3 | 15,6 / 0,8 / 0,4 | 42,1 / 4,4 / 5,8 |
| VietMed | PhoWhisper | 8,8 / 4,0 / 5,9 | 21,7 / 6,3 / 5,6 | 53,3 / 11,8 / 10,8 |
| VietMed | Qwen3-ASR | 7,7 / 4,1 / 4,0 | 17,0 / 6,3 / 3,9 | 50,0 / 14,2 / 13,0 |

Thứ tự độ khó giữ nguyên với traffic (VietMed > FLEURS > VIVOS), nhưng đổi ở babble 0 dB:
FLEURS trở thành tập tệ nhất.

### VIVOS: dễ nhất và bền nhất

WER sạch 6,2 (PhoWhisper) và 10,1 (Qwen3-ASR); khoảng 50% ở babble 0 dB, thấp nhất trong ba
tập.

- Phòng yên tĩnh, phát âm rõ, từ vựng thông dụng.
- Câu ngắn (10 từ) nên decoder ít có cơ hội "trôi" sang nội dung khác khi audio không rõ:
  không có câu VIVOS nào bị lặp ở babble 0 dB.
- **Khoảng cách 4 điểm giữa hai model gần như toàn là lỗi thay thế** (9,4 so với 5,7 khi
  sạch; lỗi xoá và chèn dưới 0,5 với cả hai). Qwen3-ASR nhận sai một âm tiết, thường ở thanh
  điệu hoặc phụ âm đầu: "sóng" thành "sống", "xác" thành "sách", "trông" thành "trong",
  "giá" thành "giáp". Với câu ngắn và từ đơn lẻ, decoder không có ngữ cảnh để sửa, nên độ
  chính xác về âm học quyết định kết quả.
- **Giả thuyết**: PhoWhisper được fine-tune trên dữ liệu có VIVOS (phần train, khác người nói
  với phần test). Nếu đúng thì đây là lợi thế in-domain về giọng, cách thu âm và cách viết
  transcript, khớp với việc PhoWhisper tốt hơn ở cả 9 điều kiện. Cần đối chiếu danh sách dữ
  liệu huấn luyện trong paper PhoWhisper.

### FLEURS: trung bình khi sạch, tệ nhất ở babble 0 dB

WER sạch 9,3 và 8,4; ở babble 0 dB là 83,5 và 119,9 (gấp 9 và 14 lần bản sạch).

- Câu Wikipedia có tên riêng, từ mượn và số. Ngay trên audio sạch, "Asus Eee" bị nhận thành
  "aegis e" (PhoWhisper) và "ASICI" (Qwen3-ASR).
- **Câu dài làm hỏng nặng hơn**: ở babble 0 dB, 15,6% câu FLEURS của Qwen3-ASR rơi vào một
  trong các kiểu hỏng của [mục 8](#8-các-kiểu-hỏng-ở-babble-0-db), so với 3,7% ở VietMed và
  0,9% ở VIVOS. Khi decoder đã lạc, câu càng dài thì nó càng có nhiều chỗ để lặp hoặc bịa.
  Lỗi chèn của Qwen3-ASR trên FLEURS lên tới 45,8%, lỗi xoá 17,8%.

#### Câu có chữ số

169 câu FLEURS (19,9% số câu, 22,7% số từ) có chữ số trong transcript gốc. WER (%) theo
`has_digit`:

| Điều kiện | Không số, Pho / Qwen | Có số, Pho / Qwen | Chênh (có − không), Pho / Qwen |
|---|---:|---:|---:|
| clean | 8,17 / 7,77 | 13,30 / 10,51 | +5,1 / +2,7 |
| traffic 20 dB | 8,31 / 8,16 | 13,53 / 10,90 | +5,2 / +2,7 |
| traffic 10 dB | 9,74 / 9,43 | 15,46 / 11,69 | +5,7 / +2,3 |
| traffic 5 dB | 12,44 / 11,78 | 17,86 / 13,62 | +5,4 / +1,8 |
| traffic 0 dB | 19,90 / 18,75 | 24,50 / 20,73 | +4,6 / +2,0 |
| babble 5 dB | 25,89 / 28,55 | 32,22 / 26,56 | +6,3 / −2,0 |
| babble 0 dB | 83,56 / 112,22 | 83,16 / 145,95 | −0,4 / +33,7 |

- **Câu có số khó hơn một mức cố định.** Khoảng chênh gần như không đổi theo traffic (+4,6 đến
  +5,7 với PhoWhisper, +1,8 đến +2,7 với Qwen3-ASR), nên nó đến từ cách đọc và viết số, không
  phải từ nhiễu. Bỏ các câu có số, WER sạch của FLEURS là 8,2 và 7,8.
- **Vì sao PhoWhisper chịu thiệt hơn.** Trên bản sạch, PhoWhisper viết số bằng chữ theo cách
  người nói đọc (chỉ 0,7% output có chữ số), còn Qwen3-ASR viết chữ số (22,7% output, gần
  bằng tỉ lệ câu có số). `normalize_text` đổi chữ số của tham chiếu và của Qwen3-ASR thành chữ
  theo cùng một cách, nên chúng khớp nhau. Cách đọc của người nói thì có thể khác: "2011" được
  đọc và PhoWhisper chép là "hai không mười một", trong khi tham chiếu sau chuẩn hoá là "hai
  nghìn không trăm mười một", thành ra vài lỗi thay thế và chèn. Lỗi chèn của PhoWhisper ở
  câu có số là 3,2%, so với 1,4% ở câu không số.
- Một phần lợi thế của Qwen3-ASR trên FLEURS (−0,9 điểm khi sạch) đến từ đây: ở câu không
  số, hai model chỉ chênh 0,4 điểm.
- Ở babble 0 dB, câu có số của Qwen3-ASR có WER 146% (chèn 70,7%). Câu có số dài hơn (35 so
  với 30 từ), hợp với nhận xét câu dài hỏng nhiều hơn ở [mục 8](#8-các-kiểu-hỏng-ở-babble-0-db).
- **Giả thuyết**: bản ghi FLEURS thường có khoảng im lặng ở đầu và cuối. SNR tính trên toàn
  câu, nên trong các đoạn đó chỉ còn babble, và model chép babble thành chữ. Cách kiểm tra:
  tính tỉ lệ im lặng của từng câu và xem tương quan với số lỗi chèn.

### VietMed: khó nhất khi sạch, suy giảm tương đối ít nhất

WER sạch 18,7 và 15,8; ở babble 0 dB là 75,9 và 77,1 (chỉ gấp 4–5 lần bản sạch).

- Khác cả về **kênh** lẫn **miền**. Băng hẹp 8 kHz làm mất các tần số cao giúp phân biệt
  phụ âm như s/x, ch/tr. Lời nói tự nhiên, có ngập ngừng và thuật ngữ y khoa ("thuyên tắc
  huyết khối" bị nhận thành "thuyên tắc với khối").
- **Lỗi xoá và chèn cao ngay khi sạch** (4% và 4–6%, so với dưới 2% ở FLEURS và dưới 0,5%
  ở VIVOS). Đây là dấu hiệu của lời nói tự nhiên: từ đệm, lặp từ, nói chồng. Model bỏ hoặc
  thêm những từ này khác với cách người gán nhãn ghi lại.
- Suy giảm tương đối nhỏ một phần vì WER sạch đã cao.
- **Giả thuyết**: tiếng nói không có gì trên 4 kHz, nên model vốn không dựa vào dải đó cho
  tập này, và nhiễu đã giới hạn băng tần không lấy đi thêm thông tin nào ở đó.

## 7. So sánh hai model

Paired bootstrap trên WER, Qwen3-ASR trừ PhoWhisper, 1.000 lần lấy mẫu lại, khoảng tin cậy
95%. Âm nghĩa là Qwen3-ASR tốt hơn. **Có ý nghĩa** nghĩa là khoảng tin cậy không chứa 0.

| Tập | Điều kiện | Pho | Qwen | Δ | KTC 95% | Có ý nghĩa |
|---|---|---:|---:|---:|---|:-:|
| FLEURS | clean | 9,33 | 8,39 | −0,94 | [−1,48; −0,44] | có |
| FLEURS | traffic 20 dB | 9,49 | 8,78 | −0,71 | [−1,27; −0,20] | có |
| FLEURS | traffic 10 dB | 11,03 | 9,94 | −1,09 | [−1,69; −0,55] | có |
| FLEURS | traffic 5 dB | 13,67 | 12,20 | −1,47 | [−2,07; −0,87] | có |
| FLEURS | traffic 0 dB | 20,94 | 19,20 | −1,75 | [−2,53; −1,00] | có |
| FLEURS | babble 20 dB | 9,91 | 8,65 | −1,26 | [−1,84; −0,75] | có |
| FLEURS | babble 10 dB | 13,73 | 13,81 | +0,08 | [−2,22; 4,37] | không |
| FLEURS | babble 5 dB | 27,32 | 28,10 | +0,78 | [−2,80; 5,91] | không |
| FLEURS | babble 0 dB | 83,47 | 119,86 | +36,39 | [22,89; 51,67] | có |
| VIVOS | clean | 6,20 | 10,10 | +3,90 | [3,19; 4,62] | có |
| VIVOS | traffic 20 dB | 6,32 | 10,40 | +4,08 | [3,31; 4,80] | có |
| VIVOS | traffic 10 dB | 7,32 | 11,12 | +3,81 | [3,05; 4,51] | có |
| VIVOS | traffic 5 dB | 9,19 | 12,92 | +3,73 | [2,88; 4,65] | có |
| VIVOS | traffic 0 dB | 13,88 | 16,72 | +2,84 | [1,82; 3,70] | có |
| VIVOS | babble 20 dB | 6,76 | 10,85 | +4,09 | [3,36; 4,80] | có |
| VIVOS | babble 10 dB | 9,52 | 13,20 | +3,68 | [2,77; 4,55] | có |
| VIVOS | babble 5 dB | 17,70 | 21,29 | +3,59 | [2,33; 4,81] | có |
| VIVOS | babble 0 dB | 49,82 | 52,27 | +2,45 | [0,26; 4,63] | có |
| VietMed | clean | 18,73 | 15,77 | −2,96 | [−3,72; −2,20] | có |
| VietMed | traffic 20 dB | 19,12 | 16,28 | −2,83 | [−3,52; −2,20] | có |
| VietMed | traffic 10 dB | 21,53 | 17,76 | −3,77 | [−4,49; −3,06] | có |
| VietMed | traffic 5 dB | 25,22 | 20,57 | −4,64 | [−5,41; −3,97] | có |
| VietMed | traffic 0 dB | 33,58 | 27,26 | −6,33 | [−7,17; −5,53] | có |
| VietMed | babble 20 dB | 19,52 | 16,70 | −2,82 | [−3,46; −2,16] | có |
| VietMed | babble 10 dB | 26,48 | 21,42 | −5,06 | [−7,10; −3,57] | có |
| VietMed | babble 5 dB | 40,60 | 34,68 | −5,92 | [−8,04; −4,34] | có |
| VietMed | babble 0 dB | 75,86 | 77,12 | +1,25 | [−4,01; 8,83] | không |

**Kết quả.**

- **Qwen3-ASR tốt hơn trên FLEURS và VietMed** khi sạch, ở mọi mức traffic và ở babble
  20 dB (VietMed: đến cả babble 5 dB). Chênh lệch lớn nhất ở VietMed (−3 đến −6 điểm).
- **Qwen3-ASR chịu traffic tốt hơn**: chênh lệch trên VietMed tăng từ −3,0 (sạch) lên −6,3
  (traffic 0 dB), trên FLEURS từ −0,9 lên −1,8. Gộp ba tập, traffic 0 dB làm WER của
  Qwen3-ASR tăng +10,5 điểm, của PhoWhisper +12,4 điểm.
- **PhoWhisper tốt hơn trên VIVOS ở cả 9 điều kiện**, chênh 2,5–4,1 điểm. Khoảng chênh gần
  như không đổi theo SNR, nên đây là khác biệt ở mức nền, không phải khác biệt về độ bền với
  nhiễu ([mục 6](#vivos-dễ-nhất-và-bền-nhất)).
- **Hai model đổi chỗ khi babble to dần.** Trên FLEURS, lợi thế của Qwen3-ASR mất từ babble
  10 dB và đảo ngược ở 0 dB (+36,4 điểm). Trên VietMed, hai model hoà nhau ở babble 0 dB.
  Đây không phải vì Qwen3-ASR nghe kém hơn: bỏ các câu hỏng cả câu, nó lại tốt hơn ở FLEURS
  babble 10 và 5 dB và ở VietMed babble 0 dB, và chỉ kém 2,3 điểm ở FLEURS babble 0 dB
  ([mục 8](#câu-hỏng-giải-thích-các-điểm-qwen3-asr-thua)).
- Một phần lợi thế của Qwen3-ASR trên FLEURS đến từ cách viết số chứ không phải từ nhận dạng
  ([mục 6](#câu-có-chữ-số)).

**Vì sao.**

- **Decoder LLM của Qwen3-ASR có mô hình ngôn ngữ mạnh hơn.** Khi tín hiệu âm thanh còn đủ,
  nó dùng ngữ cảnh để sửa những âm bị nhiễu che, nên có lợi với thuật ngữ y khoa, lời nói
  tự nhiên (VietMed) và câu dài (FLEURS), và có lợi với traffic, loại nhiễu chỉ làm mờ âm.
- **Cùng mô hình ngôn ngữ đó là điểm yếu khi tín hiệu mất hẳn.** Ở babble 0 dB, bằng chứng âm
  thanh yếu và lẫn với giọng khác; decoder tiếp tục sinh văn bản theo prior của nó, lặp lại
  cụm từ vừa sinh, hoặc bám theo giọng nền ([mục 8](#8-các-kiểu-hỏng-ở-babble-0-db)).
- **PhoWhisper chỉ được fine-tune cho tiếng Việt.** Nó không bao giờ ra tiếng Anh, kể cả khi
  giọng nền tiếng Anh rõ hơn giọng chính, và lệch ít hơn trên dữ liệu đọc rõ ràng như VIVOS.
  Cái giá là mô hình ngôn ngữ yếu hơn với từ vựng ngoài miền (VietMed).

## 8. Các kiểu hỏng ở babble 0 dB

Ở babble 0 dB, hai model nghe sai số từ gần như ngang nhau (S ≈ 52). WER của Qwen3-ASR cao
hơn vì nó **chèn nhiều từ hơn** (27,2 so với 14,8) và **bỏ sót nhiều từ hơn** (14,5 so với
8,2). Đọc output của từng câu cho thấy ba kiểu hỏng cả câu, khác hẳn việc nghe sai một từ.
`evaluation.failure_summary` (`docs/evaluation.md` §6) đếm chúng so với transcript tham
chiếu:

| Kiểu | Định nghĩa (output đã chuẩn hoá, so với tham chiếu) | Ví dụ |
|---|---|---|
| **Lặp** | Dài hơn 2 lần tham chiếu, và có một cụm 1–20 từ lặp liền nhau ít nhất 3 lần | Qwen3-ASR: "chúng ta sẽ đi đến một nơi mà chúng ta sẽ có thể đi đến một nơi mà…" (511 từ). PhoWhisper: "ra thiên tòa số số số số…" (434 từ) |
| **Bỏ cụt** | Ngắn hơn một nửa tham chiếu | "tuy nhiên loài chim vẫn có rất nhiều điểm giống với khủng long" → "điều kiện bảo vệ" |
| **Chép giọng nền** | Tham chiếu là tiếng Việt; output có ít nhất 3 từ và không có chữ cái tiếng Việt nào | "And gradually, the day turned into night.", "out of your mind", "Yeah, yeah, yeah." |

### Số câu hỏng và phần WER của chúng

Babble 0 dB. Một câu có thể thuộc hai kiểu; cột **Tổng** đếm mỗi câu một lần. **Điểm WER từ
câu hỏng** là lỗi của các câu hỏng chia cho toàn bộ số từ tham chiếu của nhóm; **WER bỏ câu
hỏng** là WER toàn corpus của các câu còn lại.

| Tập | Model | Lặp | Bỏ cụt | Chép giọng nền | Tổng | WER | Điểm WER từ câu hỏng | WER bỏ câu hỏng |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| FLEURS | PhoWhisper | 9 | 9 | 0 | 18 (2,1%) | 83,47 | 8,15 | 77,23 |
| FLEURS | Qwen3-ASR | 28 | 81 | 29 | 133 (15,6%) | 119,86 | 54,27 | 79,55 |
| VIVOS | PhoWhisper | 0 | 3 | 0 | 3 (0,4%) | 49,82 | 0,23 | 49,71 |
| VIVOS | Qwen3-ASR | 0 | 4 | 3 | 7 (0,9%) | 52,27 | ≈ 0,8 | ≈ 52,0 |
| VietMed | PhoWhisper | 1 | 13 | 0 | 14 (1,4%) | 75,86 | 2,04 | 74,89 |
| VietMed | Qwen3-ASR | 6 | 24 | 9 | 37 (3,7%) | 77,12 | 8,55 | 71,05 |
| **Cả ba tập** | PhoWhisper | 10 | 25 | 0 | 35 (1,3%) | | | |
| **Cả ba tập** | Qwen3-ASR | 34 | 109 | 41 | 177 (6,8%) | | | |

*Dòng VIVOS–Qwen3-ASR đã trừ 2 câu bị đếm nhầm là "chép giọng nền": output chỉ có chữ số
("29, 26", "11:23"). Lỗi này đã được sửa trong code sau khi `failures_zero-shot.csv` được
tạo, nên file CSV vẫn ghi 5 và 9; hai cột cuối của dòng này là giá trị gần đúng. Các dòng khác
không bị ảnh hưởng.*

Ở các điều kiện khác, câu hỏng hiếm: bản sạch 5 câu (PhoWhisper, VietMed), traffic 0 dB 1–4
câu mỗi model, babble 5 dB 6–7 câu mỗi model. Nhưng ở FLEURS, vài câu đã đủ làm lệch so
sánh: ở babble 10 dB, **một** câu lặp của Qwen3-ASR đóng góp 1,94 điểm WER; ở babble 5 dB,
5 câu đóng góp 3,93 điểm.

### Câu hỏng giải thích các điểm Qwen3-ASR thua

| FLEURS | WER Pho / Qwen | WER bỏ câu hỏng, Pho / Qwen |
|---|---:|---:|
| babble 10 dB | 13,73 / 13,81 | 13,73 / 11,88 |
| babble 5 dB | 27,32 / 28,10 | 27,32 / 24,30 |
| babble 0 dB | 83,47 / 119,86 | 77,23 / 79,55 |
| **VietMed** babble 0 dB | 75,86 / 77,12 | 74,89 / 71,05 |

- Ở FLEURS babble 10 và 5 dB, bootstrap ở [mục 7](#7-so-sánh-hai-model) không thấy khác
  biệt. Bỏ vài câu hỏng, Qwen3-ASR thấp hơn PhoWhisper 2–3 điểm, như ở các điều kiện còn lại.
- Ở FLEURS babble 0 dB, khoảng cách +36,4 điểm còn khoảng +2,3 điểm. 15,6% số câu hỏng đóng
  góp 54 điểm WER của Qwen3-ASR, so với 8 điểm ở PhoWhisper.
- Ở VietMed babble 0 dB, hai model hoà nhau, nhưng trên các câu không hỏng Qwen3-ASR thấp hơn
  3,8 điểm.
- Vậy **về nghe, Qwen3-ASR tốt hơn ở gần như mọi nơi trừ VIVOS; điểm yếu của nó là hỏng cả
  câu dưới babble to**. Lưu ý: "WER bỏ câu hỏng" của hai model tính trên hai tập câu hơi khác
  nhau (mỗi model bỏ câu hỏng của riêng nó). Để so sánh chặt, chạy paired bootstrap trên các
  câu không model nào hỏng (`docs/evaluation.md` §6).

Trong 28 câu lặp của Qwen3-ASR ở FLEURS, khoảng 20 câu dài từ 400 từ trở lên, sát giới hạn
`max_new_tokens=512`: model sinh văn bản cho đến khi bị cắt.

### Giải thích

- **Lặp** là lỗi quen thuộc của decoder tự hồi quy khi mất bám vào audio: mỗi token mới dựa
  trên các token trước, và khi audio không còn cho bằng chứng rõ thì vòng lặp tự củng cố.
  PhoWhisper lặp một từ ("số số số…"); Qwen3-ASR lặp cả một câu tiếng Việt trôi chảy, vì
  decoder LLM sinh được câu dài, đúng ngữ pháp. Decode greedy và không có phạt lặp làm lỗi
  này dễ xảy ra hơn.
- **Bỏ cụt**: model nhận ra một đoạn ngắn rồi phát token kết thúc, coi phần còn lại là
  nhiễu. Qwen3-ASR làm vậy nhiều gấp 4 lần PhoWhisper (109 so với 25 câu); lỗi xoá của nó
  trên FLEURS lên 17,8%.
- **Chép giọng nền**: babble là giọng đọc sách tiếng Anh, đôi khi rõ hơn giọng chính. Qwen3-ASR
  là model đa ngôn ngữ; tham số `language="Vietnamese"` không ngăn được nó chép một câu tiếng
  Anh khi câu đó là tiếng nói rõ nhất trong audio. PhoWhisper chỉ được fine-tune cho tiếng
  Việt và luôn ra tiếng Việt, nhưng vẫn có thể chép giọng nền thành âm tiết tiếng Việt vô
  nghĩa (không đo được bằng định nghĩa trên).
- **Câu dài hỏng nhiều hơn**: FLEURS (30,7 từ/câu) có 15,6% câu hỏng, VietMed (22,6) 3,7%,
  VIVOS (10,2) 0,9%. Câu dài cho decoder nhiều bước để lạc, và một câu dài bị lặp tạo ra rất
  nhiều lỗi chèn.
- Vài câu như vậy đủ đẩy WER toàn corpus lên trên 100% (FLEURS: 119,9%). Con số này không
  có nghĩa là mọi từ đều sai.

## 9. Đánh giá

### Mức suy giảm theo điều kiện

| Mức | Điều kiện | WER tăng (gộp ba tập) | Nhận xét |
|---|---|---|---|
| Không đáng kể | traffic 20 dB, babble 20 dB | dưới +0,7 điểm | Trong vùng model đã quen |
| Nhẹ | traffic 10 dB | +1,7 đến +2,1 | |
| Trung bình | traffic 5 dB, babble 10 dB | +4 đến +6 | |
| Nặng | traffic 0 dB, babble 5 dB | +10 đến +19 (gấp 2–2,6 lần) | Vẫn dùng được, nhưng 1 từ sai trên 3–4 từ |
| Hỏng | babble 0 dB | +63 đến +82 (gấp 6–8 lần) | Không dùng được; có lặp, bỏ cụt, chép nhầm giọng |

### Chọn model

- **Không model nào tốt hơn ở mọi nơi.** Qwen3-ASR thắng ở 8/9 điều kiện khi gộp ba tập, và
  rõ nhất ở VietMed và traffic. Trên các câu không hỏng, nó còn tốt hơn ở hầu hết điều kiện
  babble. PhoWhisper thắng trên VIVOS ở mọi điều kiện, và an toàn hơn ở babble rất to: số câu
  hỏng ít hơn 5 lần (35 so với 177 câu ở babble 0 dB), và không bao giờ ra tiếng Anh.
- Với audio có nhiễu nền kiểu đường phố, hoặc miền chuyên ngành, Qwen3-ASR là lựa chọn tốt
  hơn. Với audio có nhiều người nói chồng ở SNR thấp, Qwen3-ASR cần thêm biện pháp chống lặp
  và chống chép nhầm giọng trước khi dùng.

### Ý nghĩa cho bước fine-tune

- **Khoảng cải thiện nằm ở babble 5 và 0 dB.** Traffic và SNR ≥ 10 dB chỉ còn vài điểm. Tiêu
  chí thành công của LoRA với trộn nhiễu online nên đặt ở babble.
- **Giả thuyết**: huấn luyện với babble dạy model bỏ qua giọng nền, nên sẽ giảm mạnh kiểu
  "chép giọng nền" và "bỏ cụt" của Qwen3-ASR. Kiểu "lặp" phụ thuộc nhiều vào cách decode, nên
  có thể giảm ít hơn.
- Với Qwen3-ASR, phần lớn cải thiện có thể đến ở babble 0 dB từ việc **bớt câu hỏng** (54 điểm
  WER ở FLEURS) hơn là từ nghe tốt hơn. Báo cáo cả hai chỉ số của `failure_summary`:
  `failed_pct` (fine-tune có làm mất câu hỏng không) và `wer_without_failed` (có nghe tốt hơn
  trên các câu còn lại không).
- Một phần lỗi ở babble 0 dB có thể giảm bằng cách decode (phạt lặp, giới hạn
  `max_new_tokens` theo độ dài audio) mà không cần huấn luyện. Nên đo riêng phần này để không
  gán toàn bộ cải thiện cho fine-tune.

## 10. Hạn chế và bước tiếp theo

### Hạn chế

- **WER toàn corpus nhạy với câu hỏng.** Một câu lặp có thể đóng góp gần 2 điểm WER cho cả
  một điều kiện ([mục 8](#8-các-kiểu-hỏng-ở-babble-0-db)). Vì vậy báo cáo cả số câu hỏng và
  WER bỏ câu hỏng.
- **Ngưỡng của các kiểu hỏng do mình chọn** (gấp 2 lần, một nửa, 3 lần lặp, ít nhất 3 từ), và
  "chép giọng nền" chỉ bắt được output không có chữ cái tiếng Việt. "WER bỏ câu hỏng" của hai
  model tính trên hai tập câu hơi khác nhau.
- **`failures_zero-shot.csv` được tạo trước khi sửa lỗi đếm số** ở `off_language`: nhóm
  Qwen3-ASR × VIVOS đếm thêm 12 câu chỉ có chữ số (2 ở babble 0 dB, 3 ở babble 5 dB, 7 ở
  traffic). Các nhóm khác không đổi. Lần chạy notebook sau sẽ tạo lại file với số đúng.
- **Cách đọc số.** Câu có chữ số có WER cao hơn một mức cố định
  ([mục 6](#câu-có-chữ-số)), và lợi thế này nghiêng về model viết chữ số. Nên báo cáo kết quả
  FLEURS cả khi có và khi bỏ câu có số.
- **Babble chủ yếu là tiếng Anh.** Qwen3-ASR đã chép giọng nền tiếng Anh trong 41 câu. Với
  babble tiếng Việt, kiểu lỗi này sẽ thành chép nhầm sang câu tiếng Việt của người khác, khó
  phát hiện hơn và có thể nặng hơn. Cần ghi rõ trong báo cáo.
- **Clip traffic có đoạn im lặng.** Một số clip FSD50K im lặng một phần; khi đoạn nhiễu im
  lặng một phần, phần còn lại của câu bị nhiễu hơn mức SNR dự kiến.
- **VietMed khác cả kênh lẫn miền**, nên không tách được ảnh hưởng của nhiễu với ảnh hưởng
  của audio 8 kHz và lời nói tự nhiên.

### Trước khi so sánh với fine-tune

- Nếu thêm phạt lặp hoặc giới hạn `max_new_tokens` theo độ dài audio, phải áp dụng cho
  **cả hai** model, chạy lại zero-shot với cùng cách decode, và ghi rõ trong phần phương pháp.
- So sánh model fine-tune với zero-shot bằng cùng paired bootstrap, dùng predictions trong
  [`data/zero-shot/`](data/zero-shot/). Chạy thêm `failure_summary` cho mọi variant, và
  paired bootstrap trên các câu không variant nào hỏng.

### Gợi ý bố cục phần kết quả trong báo cáo

1. Bảng WER/CER theo tập × điều kiện ([phụ lục](#12-phụ-lục-bảng-số-đầy-đủ)).
2. WER theo SNR ([mục 3](#3-tổng-quan-gộp-ba-tập)), kèm nhận định: ở cùng SNR, babble khó
   hơn traffic hơn 5 dB.
3. Phân rã lỗi S/D/I ([mục 5](#5-theo-loại-nhiễu)) để giải thích cơ chế của từng loại nhiễu.
4. Bảng paired bootstrap ([mục 7](#7-so-sánh-hai-model)) để so sánh hai model.
5. Bảng các kiểu hỏng ở babble 0 dB và WER bỏ câu hỏng
   ([mục 8](#8-các-kiểu-hỏng-ở-babble-0-db)), kèm vài ví dụ.
6. Bảng FLEURS theo `has_digit` ([mục 6](#câu-có-chữ-số)).

## 11. File kết quả

Output của notebook `02_zero_shot`, lưu trong [`data/zero-shot/`](data/zero-shot/). Các hình
nằm trong [`figures/`](figures/).

| File | Nội dung |
|---|---|
| `predictions_phowhisper-small_zero-shot.csv.gz`, `predictions_qwen3-asr-0.6b_zero-shot.csv.gz` | Output thô của từng model, mỗi dòng một cặp (câu, điều kiện); định dạng ở `docs/evaluation.md` |
| `summary_zero-shot.csv` | WER, CER, S/D/I theo model × tập × loại nhiễu × SNR |
| `wer_table.csv`, `cer_table.csv` | WER và CER, một cột cho mỗi model |
| `compare_wer.csv` | Paired bootstrap Qwen3-ASR trừ PhoWhisper |
| `failures_zero-shot.csv` | `failure_summary`: số câu lặp, bỏ cụt, chép giọng nền, và WER bỏ câu hỏng, theo model × tập × loại nhiễu × SNR (tạo trước khi sửa lỗi đếm số, xem [mục 10](#hạn-chế)) |
| `fleurs_by_digit.csv` | WER, CER, S/D/I của FLEURS theo model × `has_digit` × loại nhiễu × SNR |
| `run_phowhisper-small_zero-shot.json`, `run_qwen3-asr-0.6b_zero-shot.json` | Cấu hình lần chạy: model, dtype, GPU, phiên bản code và dataset, thời gian |

Predictions được nén gzip; `evaluation.load_predictions` đọc trực tiếp file `.csv.gz`. Để
chấm lại, cần thêm `test_plan.csv` và `clean.csv` từ Kaggle Dataset `asr-noisy-vi`:

```python
from evaluation import load_predictions, score, summarize

predictions = load_predictions(["docs/results/data/zero-shot/predictions_phowhisper-small_zero-shot.csv.gz",
                                "docs/results/data/zero-shot/predictions_qwen3-asr-0.6b_zero-shot.csv.gz"])
scored = score(predictions, plan, clean)  # plan, clean: DataFrames from asr-data/manifests/
```

## 12. Phụ lục: bảng số đầy đủ

### WER (%)

| Tập | Điều kiện | PhoWhisper | Qwen3-ASR |
|---|---|---:|---:|
| FLEURS | clean | 9,33 | 8,39 |
| FLEURS | traffic 20 dB | 9,49 | 8,78 |
| FLEURS | traffic 10 dB | 11,03 | 9,94 |
| FLEURS | traffic 5 dB | 13,67 | 12,20 |
| FLEURS | traffic 0 dB | 20,94 | 19,20 |
| FLEURS | babble 20 dB | 9,91 | 8,65 |
| FLEURS | babble 10 dB | 13,73 | 13,81 |
| FLEURS | babble 5 dB | 27,32 | 28,10 |
| FLEURS | babble 0 dB | 83,47 | 119,86 |
| VIVOS | clean | 6,20 | 10,10 |
| VIVOS | traffic 20 dB | 6,32 | 10,40 |
| VIVOS | traffic 10 dB | 7,32 | 11,12 |
| VIVOS | traffic 5 dB | 9,19 | 12,92 |
| VIVOS | traffic 0 dB | 13,88 | 16,72 |
| VIVOS | babble 20 dB | 6,76 | 10,85 |
| VIVOS | babble 10 dB | 9,52 | 13,20 |
| VIVOS | babble 5 dB | 17,70 | 21,29 |
| VIVOS | babble 0 dB | 49,82 | 52,27 |
| VietMed | clean | 18,73 | 15,77 |
| VietMed | traffic 20 dB | 19,12 | 16,28 |
| VietMed | traffic 10 dB | 21,53 | 17,76 |
| VietMed | traffic 5 dB | 25,22 | 20,57 |
| VietMed | traffic 0 dB | 33,58 | 27,26 |
| VietMed | babble 20 dB | 19,52 | 16,70 |
| VietMed | babble 10 dB | 26,48 | 21,42 |
| VietMed | babble 5 dB | 40,60 | 34,68 |
| VietMed | babble 0 dB | 75,86 | 77,12 |

### CER (%)

| Tập | Điều kiện | PhoWhisper | Qwen3-ASR |
|---|---|---:|---:|
| FLEURS | clean | 5,18 | 4,42 |
| FLEURS | traffic 20 dB | 5,30 | 4,71 |
| FLEURS | traffic 10 dB | 6,32 | 5,49 |
| FLEURS | traffic 5 dB | 7,99 | 7,11 |
| FLEURS | traffic 0 dB | 12,81 | 11,92 |
| FLEURS | babble 20 dB | 5,47 | 4,65 |
| FLEURS | babble 10 dB | 7,88 | 8,81 |
| FLEURS | babble 5 dB | 17,32 | 19,14 |
| FLEURS | babble 0 dB | 60,07 | 97,07 |
| VIVOS | clean | 2,71 | 4,53 |
| VIVOS | traffic 20 dB | 2,78 | 4,69 |
| VIVOS | traffic 10 dB | 3,29 | 5,10 |
| VIVOS | traffic 5 dB | 4,46 | 6,18 |
| VIVOS | traffic 0 dB | 7,21 | 8,76 |
| VIVOS | babble 20 dB | 3,03 | 4,87 |
| VIVOS | babble 10 dB | 4,73 | 6,12 |
| VIVOS | babble 5 dB | 10,32 | 11,66 |
| VIVOS | babble 0 dB | 34,34 | 35,58 |
| VietMed | clean | 13,56 | 11,48 |
| VietMed | traffic 20 dB | 13,78 | 11,81 |
| VietMed | traffic 10 dB | 15,35 | 12,79 |
| VietMed | traffic 5 dB | 17,91 | 14,63 |
| VietMed | traffic 0 dB | 24,05 | 19,67 |
| VietMed | babble 20 dB | 14,02 | 12,01 |
| VietMed | babble 10 dB | 19,16 | 15,36 |
| VietMed | babble 5 dB | 29,69 | 25,39 |
| VietMed | babble 0 dB | 56,55 | 58,93 |

![CER theo SNR, từng tập và từng loại nhiễu](figures/cer_vs_snr.png)

*CER theo SNR, kèm khoảng tin cậy bootstrap 95%. Bố cục như hình WER ở
[mục 3](#3-tổng-quan-gộp-ba-tập).*

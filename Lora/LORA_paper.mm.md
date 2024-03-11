# LORA (Low-rank-adaptation of LLLM)
## link to the paper
- [LORA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685.pdf)
## Abstract Lora
-   Trước khi lora ra đời ,Adapter và prefix-tuning đã được các công ty lớn triển khai tuy nhiên gặp vấn đề lớn về chi phí tính và lưu trữ .
    - Adapter lại tăng độ trễ (latency) khi suy luận.
    - Prefix-tuning lại khiến cho model khó tối ưu.
-   Lora đóng băng trọng số của model đã train săn được đào tạo trên một tập dữ liệu lớn.
-   Tính chỉnh model đã train cho nhiều đối tượng khác nhau sang một tập dữ liệu nhỏ hơn.
-   Lora đưa các ma trận phân rã thứ hạng thấp có thể huấn luyện vào mô hình có kiến trúc trainsformer.
-  Ma trận phân rã thứ hạng thấp là một phương phap giảm kích thước của ma trận bằng cách giảm số lượng hàng và cột của nó. denta(W) = B * A (A,B là ma trận có kích thước nhỏ hơn W)
-  So với GPT-3 175 tỷ tham số, Lora giảm só lượng tham số có thể huấn luyện 10000 lần. yêu cầu Gpu gấp 3 lần.
-  Lora ngang bằng hoặc tốt hơn trên roberta , Deberta , GPT-2 , GPT-3. Ít tham số đào tạo tuy nhiên thông lượng đào tạo lớn.
-  Không có độ trễ suy luận.
-  **Biểu thức** : h = W0 * x + delta(W) * x  = W0 * x + B*A*x
    -  **W0** là ma trận trọng số của mô hình đã train.(m*n)
    -  **delta(W)** là ma trận phân rã thứ hạng thấp.(m*n)
    -  **r** là hệ số phân rã thứ hạng thấp.(r << min(m,n))
    -  **B** là ma trận có kích thước nhỏ hơn W0.(m*r)
    -  **A** là ma trận có kích thước nhỏ hơn W0.(r*n)

## 4 Our Method
### 4.1 Low-Rank-parametrizered update
-  **Mô tả**
    -   Trong mạng neural có nhiều lớp dày đặc thực hiện phép nhân ma trân.
    -   Tất cả đêu là các lớp có đầy đủ thứ hạng .
-  **Ý tưởng Lora**
    -  Thay vì sử dụng ma trận đầy đủ thứ hạng, ta sử dụng ma trận phân rã thứ hạng thấp. 
    -   **h = W0 * x + delta(W) * x  = W0 * x + B*A*x**
    -   **Ví dụ**
        -   **h** có kích thước (m*n)(1000*1000) weight parameter =1000000
        -   **W0** có kích thước (m*n)(1000*1000) weight parameter = 1000000
        -   **delta(W)** có kích thước (m*n)(1000*1000) weight parameter = 1000000
        -   **Chọn r = 5** : r << min(m,n) (5 << 1000)
        -   **B** có kích thước (m*r)(1000*5) weight parameter = 5000
        -   **A** có kích thước (r*n)(5*1000) weight parameter = 5000
    -  **Đánh giá**
        -  Thay vì sử dụng ma trân thứ hạng đầy đủ delta(W) ta sử dụng ma trận phân rã thứ hạng thấp B*A. Từ đó giảm được số lượng tham số cần huấn luyện đi 100 lần.(1000000 -> 10000)
        -   **Nếu r lơn hơn rank của W** thì phân rã W làm mất mát thông tin làm giảm độ chính xác của mô hình và làm tăng trong lượng tính toán.
-  **Training**
    -   **W0** : Đóng băng trọng số và không cập nhật độ dốc.
    -   **B** và **A** : Cập nhật độ dốc.
    -   Cả **W0** và **delta(W)** đều có cùng input và output có cung vector kich thước.

-  **Tổng quát tỉnh chỉnh**
    -  Lora không cần cập nhật tất cả các trọng số của mô hình.
    -  Khơi tạo tham số cho **A** bằng phân phồi Gaussion random và **B** được khởi tạo bằng 0. ->  **delta(W)** được khởi tạo bằng 0.
    -  Khi tăng số lượng tham số huấn luyện (r) thì lora gân như hội tụ để huấn luyện mô hình bạn đầu.
    -  Không có độ trê suy luận. Do khi triên khai có thể tính toán rõ ràng W = W0 + AB. và thực hiện tính toán như bình thường.
    - Vì **W0** và **AB** đều thuộc (dxk) Nên khi cần khôi phục model chi cần trừ đi **AB** là được **W0**. Hoặc thêm **A`B`** vào **W0** là được **W** cho các tác phụ khác nhau
    -  **Kết quả**
        -  Hoạt động nhanh với bộ nhớ ít
        -  Không có độ trễ suy luận
### APPLYING LORA TO TRANSFORMER
-   Chúng ta có thể áp dụng LoRA cho bất kỳ tập hợp con ma trận trọng số nào trong mạng nơron để giảm số lượng tham số có thể huấn luyện được.
-   Trong kiến trúc Transformer, có bốn ma trận trọng số trong mô-đun tự chú ý (Wq, Wk, Wv, Wo (output)) và hai ma trận trong mô-đun MLP(multi layer perceptron).
##  Đánh giá
-  **Lora** so với  các phương pháp **Adapter** , **prefix-tuning**
    -   lora với lượng tham số ít hơn nhiêu **Gpt-2 medium** (prefix-tuning (25,19M tham số / 354.92M tham số ) , Adapter (11.09M tham số) , Lora (0.35M tham số))
    -    **BLEU**  prefix-tuning : 68.3% , Adapter : 68.9% , Lora : 70.4%
    -    **NIST**  prefix-tuning : 8.62% , Adapter : 8.71% , Lora : 8.85%
    -    **METEOR**  prefix-tuning : 46.2% , Adapter : 46.1% , Lora : 46.8%
    -    **ROUGE-L**  prefix-tuning : 71.0% , Adapter : 71.3% , Lora : 71.8%
    -    **CIDEr**  prefix-tuning : 2.47% , Adapter : 2.47% , Lora : 2.53%
-  Hay là trên 1 số model khác **RoBerta**,**GPT-3**,**Deberta** trong bài báo.
-  **Kết quả đảnh giá về việc nên áp dụng lora cho loại trọng số nào (Wq,Wk,Wv,Wo)**
    - Chọn **GPT-3 175B tham số với tất cả 96 layer** làm mô hình cơ sở.
    - Chọn đánh giá trên **WikiSQL** và **MultiNLI** giới hạn số lương tham số huấn luyên 18M
    - **Kết quả**
        - **r = 8**  chỉ điểu chỉnh đơn cho từng loại trọng số.
            -   Kết quả tốt nhất khi thêm lora với **Wo(Weight output)** WikiSQL(73,2%) , MultiNLI(91.3%)
        - **r = 4**  chỉ điểu chỉnh với 2 cặp (Wq, Wk) và (Wq, Wv)
            -   Kết quả tốt nhất khi thêm lora với **(Wq, Wv)** WikiSQL(73.7%) , MultiNLI(91.3%)
        - **r = 2**  chỉ điểu chỉnh với cả 4 tham số (Wq, Wk, Wv, Wo)
            -   Kết quả  WikiSQL(73.7%) , MultiNLI(91.7%)  
        -  Kết quả tốt nhất khi thêm lora với cả 4 tham số (Wq, Wk, Wv, Wo) với **r = 2**.(2<<96)
## Triển Khai
-   Link git [LORA git](https://github.com/cuoicungtui/LORA_Model_test)
-   **Lora** được triển khai trên **Pytorch**.
-   **Thư viên**
    -  [microsoft/lora](https://github.com/microsoft/LoRA)
    -  [hunggingface/peft](https://github.com/huggingface/peft)
-  Theo thư viện chạy với **CAUSALLM BLOOM**[link git](https://github.com/cuoicungtui/LORA_Model_test/blob/master/CAUSALLM_bloom_model.ipynb)
    - Tổng **3B tham số**  với **30 layer**
    - Thêm  lora vào với **r = 8** cho layer **query_key_value** của tất cả 30 layer. ->  **2457600 tham số training** chiếm  **trainable: 0.0812% tham số**
-  Thử tự setup với **bert-base-uncased** [link git](https://github.com/cuoicungtui/LORA_Model_test/blob/master/Simple_LoRA_Implementation.ipynb)
    - Tổng **109M tham số**  với **12 layer**
    - Thêm lora vào với **r = 3** cho layer **bert.encoder.layer.1.attention.self.query,key,value** của tất cả 12 layer. ->  **165888 tham số training** chiếm  **trainable: 0.15% tham số**
- Có thể tinh chinh băng cách thêm lora vào cho từng Layer một khác nhau bằng cách gọi tên layer đó
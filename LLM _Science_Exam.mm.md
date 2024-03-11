# LLM_Science_Exam Kaggle competitions
## Link competition: [link](https://www.kaggle.com/competitions/kaggle-llm-science-exam/overview)
## Overview 
### Description
-   **Goal of the Competition** Trả lời câu hỏi khó dựa trên cơ sở khoa học được viết bởi ngôn ngữ lớn. Câu hỏi trả lời ở dạng câu hỏi trắc nghiệm.
-   **Evaluation** Điểm số được tính bằng cách đếm số câu trả lời đúng. Đầu ra tối đa được đưa ra là 3 lựa chọn theo thứ tự ưu tiên. Điểm số cho mỗi đáp được tính theo vị trí của câu trả lời đúng. score = 1 / (rank)(rank:1->3). Tổng kết quả bằng cách lấy trung bình điểm số của tất cả các câu hỏi.
## Dataset
### Data competitions
- [link data](https://www.kaggle.com/competitions/kaggle-llm-science-exam/data)
- **Columns**
    -  **prompt** : Văn bản của câu hỏi
    -  **option** : 5 lựa chọn của câu hỏi(A,B,C,D,E)
    -  **answer** : Đáp án đúng của câu hỏi
### Data additional
-   **60k data with contect** [link data](https://www.kaggle.com/datasets/cdeotte/60k-data-with-context-v2)

-  Bổ sung thêm cột context chứa nội dung của câu hỏi được bổ sung từ bộ dữ liệu từ wikipedia. Nhằm tăng cường thông tin của câu hỏi và giúp mô hình có thể dự đoán tốt hơn.
## Cách tiếp cận bài toán trả lời câu hỏi trăc nghiệm
### I/O
-   **Input** : Prompt , option(A,B,C,D,E)
-   **Output** : Answer in (A or B or C or D or E)
### Xác đinh bài toán
#### Multiclassification
#### MultipleChoice
### Processing data
### Select model
### Train model
### Improvement strategy


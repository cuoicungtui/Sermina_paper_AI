# Generative Adversarial Networks (GANs)
## Overview:
-  GAN là một mô hình học máy gồm 2 lớp mạng nơ-ron cạnh tranh với nhau và Gan thường chạy không giám sát.
-  Hai mạng thân kinh tạo nên Gan là mạng sinh(Generator) và mạng phân biệt(Discriminator). Bộ generator là một mạng noron để sinh ra ảnh giả còn discriminator là một mạng nơ-ron để phân biệt ảnh thật và ảnh giả. Quá các vòng huấn luyên làm cho bộ Generator tạo ra ảnh giả càng giống với ảnh thật và bộ Discriminator phân biệt ảnh giả và ảnh thật với độ chính xác cao dần lên.
-  Nói với ngôn ngữ của Lý thuyết trò chơi, trong bối cảnh này, hai mô hình thi đấu với nhau và đối nghịch trong một game có tổng bằng 0.
- Input của mạng sinh(generator) là một vector ngẫu nhiên và output là một ảnh giả.
- Input của mạng phân biệt(discriminator) là một ảnh thật hoặc ảnh giả và output là xác suất ảnh đó là ảnh thật.

## Thử nghiệm với bài toán text-to-image
### Mô tả bài toán
-  Input: một câu văn bất kỳ miêu tả về 1 loại hoa.
- Output: ảnh của loại hoa được miêu tả trong câu văn đó được sinh bởi bộ Generator.
### DATASET
-   [102 Category Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
- Dứ liệu gồm ảnh của 102 loại hoa khác nhau. Và mỗi ảnh có it nhất 5 câu mô tả về loại hoa đó. vị dự như màu sắc, hình dáng, nhụy hoa, cánh hoa, ...
### Mô hình
#### Generator
-  Input: 
    -   Một vector biểu diễn câu văn được chọn là layer cuối cùng của Model RoBERTa. (kích thước 1024) 
    -   Một vector ngẫu nhiên có kích thước 100.Sử dụng làm vector nhiễu . để nối với vector biểu diễn câu văn.
-   Output: ảnh của loại hoa được miêu tả trong câu văn đó. (kích thước 64x64x3)
- Model tạo 1 layer Generator 
    -   Sử dụng các lớp  reshape vector đầu vào về 3D tensor.
    -   Sử dụng các lớp Conv2DTranspose để tăng kích thước của ảnh lên 64x64x3.
    -   Sử dụng hàm kích hoạt là LeakyReLU.
    -   Sử dụng 2 hàm mất mát đó là BinaryCrossentropy và categorical_crossentropy. BinaryCrossentropy dùng để đánh giá độ chính xác của ảnh được sinh ra. categorical_crossentropy dùng để đánh giá độ chính xác của ảnh sau khi được phân loại theo 102 lớp hoa của bộ dữ liệu.
    -   Sử dụng optimizer là Adam.
#### Discriminator
- Input:
    - Ảnh có kiểu dữ liệu là 64x64x3.
    - Text vector biểu diện của câu văn mô tả  loại hoa trong ảnh sau khi đi qua Model Roberta.(kích thước 1024)
- Output: 
    -  xác suất ảnh đó là ảnh thật.  (Activation là sigmoid)
    - Xác suất ảnh đó thuộc về 1 trong 102 loại hoa. (Activation là softmax)
- Model tạo 1 layer Discriminator
    - Sử dụng các lớp tích chập để giảm kích thước của ảnh xuống vector 1 chiều.
    - Kết hợp vector biểu diễn của ảnh vào vector biểu diễn của câu văn.
    - Sử dụng các lớp Dense để phân loại ảnh và đánh giá độ chính xác của ảnh.
    - Sử dụng hàm kích hoạt là LeakyReLU.
    - Sử dụng 2 hàm mất mát đó là BinaryCrossentropy và categorical_crossentropy. BinaryCrossentropy dùng để đánh giá độ chính xác của ảnh được sinh ra. categorical_crossentropy dùng để đánh giá độ chính xác của ảnh sau khi được phân loại theo 102 lớp hoa của bộ dữ liệu.
    - Sử dụng optimizer là Adam.
#### Mô hình GAN
-  Mô hình GAN là sự kết hợp của 2 mô hình Generator và Discriminator.
-  Mô hình GAN sẽ có 2 input là vector biểu diễn của câu văn và ảnh thật.
-  Mô hình GAN sẽ có 2 output là xác suất ảnh đó là ảnh thật và xác suất ảnh đó thuộc về 1 trong 102 loại hoa.
-  Mô hình GAN sẽ có 2 hàm mất mát là BinaryCrossentropy và categorical_crossentropy. BinaryCrossentropy dùng để đánh giá độ chính xác của ảnh được sinh ra. categorical_crossentropy dùng để đánh giá độ chính xác của ảnh sau khi được phân loại theo 102 lớp hoa của bộ dữ liệu.
- Gan = D(G(image,text))

### Các bước Training
-   Bước 1: 
        - Input 
            -   Gồm ảnh thật và câu văn miêu tả ảnh thật. Ảnh giả sử dụng random các ảnh trong dữ liệu và câu văn miêu tả dùng chung với ảnh thật.
            -   Lấy nhãn cảu ảnh thật và ảnh giả được phân lớp theo 102 loại hoa.
            -   Tạo nhẫn cho ảnh thật(1) và ảnh giả.(0) cho việc đánh giá độ chính xác của ảnh.
-   Bước 2: Training
    -   Train Discriminator
        -   **Dữ liệu thật**
            -   **Input:** ảnh thật và câu văn miêu tả ảnh thật.
            -   **Output:** xác suất ảnh đó là ảnh thật(Nhãn là 1) và xác suất ảnh đó thuộc về 1 trong 102 loại hoa.
            -   Loss: BinaryCrossentropy và categorical_crossentropy.
        - **Dữ liệu giả**
            -   **Input:** ảnh giả và câu văn miêu tả ảnh thật .
            -   **Output:** xác suất ảnh đó là ảnh giả(Nhãn là 0) và xác suất ảnh đó thuộc về 1 trong 102 loại hoa.
            -   Loss: BinaryCrossentropy và categorical_crossentropy.
        - **Dữ liệu giả được sinh ra bới Generator.**
            -   **Input:** ảnh giả và câu văn miêu tả ảnh thật.
            -   **Output:** xác suất ảnh đó là ảnh giả(Nhãn là 0) và xác suất ảnh đó thuộc về 1 trong 102 loại hoa.
            -   **Loss:** BinaryCrossentropy và categorical_crossentropy.    
        -   **Tổng Loss** của Discriminator là tổng của 3 Loss trên.    
        -   **Mục Đích** Cho Discriminator học cách phân biệt ảnh thật và ảnh giả.Và phân biệt ảnh thật thuộc về loại hoa nào.
    -   Train Generator
        - **Đóng băng mạng Discriminator.**
        - **Input:** câu văn miêu tả ảnh thật.
        - **Output :** Ảnh được sinh ra bởi Generator.
        - **Input Discriminator :** Đưa ảnh giả và vector biểu điễn 
        - **Output Discriminator:** xác suất ảnh đó là thật (Nhãn là 1) và xác suất ảnh đó thuộc về 1 trong 102 loại hoa nhân của ảnh thật . 
        - **Loss:** BinaryCrossentropy và categorical_crossentropy.
        - **Mục Đích** Khiến cho mạng Generator tạo ra ảnh giả càng giống với ảnh thật và khi phân loại ảnh giả thì xác suất ảnh đó thuộc về 1 trong 102 loại hoa càng cao.
### Demo 
-   [KAGGEL](https://www.kaggle.com/quoctuong/dc-gan-0b7cd3-5da9c1)
-   [Colab](https://colab.research.google.com/drive/1LyIOy677W4zHwVF8wxOz3bK06BBU1p5i?usp=sharing)
### Kết quả
-   Mới Hiểu được mô hình mạng đối đầu Gan
-   Hình ảnh chưa sinh ra được hoa như mô tả trong câu văn.






# Stepwise regression (Hồi quy từng bước)
## Giới thiệu
-   **Stepwise Regression** là một phương pháp kiểm tra lặp đi lặp lại ý nghĩa thống kê của từng biến độc lập trong một mô hình hồi quy tuyến tính.
-  **Mục đích** : xác định biến độc lập nào có đóng góp đáng kể vào mô hình hồi quy tuyến tính.
## **3 Cách triền khai**:
-  **Forward selection**
    - **Bắt đầu** với một mô hình hồi quy tuyến tính rỗng.
    - **Thêm** các biến độc lập một cách tuần tự vào mô hình hồi quy tuyến tính nếu thỏa mãn điều kiện.
    - **Dừng lại** khi tất cả các biến độc lập đều được thêm vào mô hình.
-  **Backward elimination**
    -   **Bắt đầu** với một mô hình hồi quy tuyến tính chứa tất cả các biến độc lập.
    -   **Loại bỏ** từng biến độc lập một cách tuần tự nếu thỏa mãn điều kiện.
    -  **Dừng lại** khi tất cả các biến độc lập đều được loại bỏ khỏi mô hình.
-  **Stepwise regression**
    - **Bắt đầu** với một mô hình hồi quy tuyến tính có chứa hoặc không chứa một số biến độc lập.
    - **Thêm** các biến độc lập một cách tuần tự vào mô hình hồi quy tuyến tính nếu thỏa mãn điều kiện.
    - **Loại bỏ** từng biến độc lập một cách tuần tự nếu thỏa mãn điều kiện.
    - **Dừng lại** khi không thể thêm hoặc loại bỏ biến độc lập nào khác.

- **Đánh giá** 
    -  **p-value** để đánh giá ý nghĩa thống kê của một biến độc lập trong mô hình hồi quy tuyến tính.
    - **p-value** càng nhỏ thì biến độc lập càng có ý nghĩa thống kê trong mô hình hồi quy tuyến tính.(thường chọn ngưỡng 0.05)
    - **p-value** càng lớn thì biến độc lập càng không có ý nghĩa thống kê trong mô hình hồi quy tuyến tính.

## Các bước thực hiện
### LINK  : [Statmodels regression linear model OLS](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html)
### Input
-   **X** : ma trận các biến độc lập.
-   **y** : vector các biến phụ thuộc(nhãn).
-   **alpha** : ngưỡng ý nghĩa thống kê.(0.05)
### Output
-   **X_Selectded** : ma trận các biến độc lập đã được lựa chọn.
### Stop condition
-  Không thể thêm hoặc loại bỏ biến độc lập nào khác. 
-  **Changed = False** dùng thêm hoặc loại bỏ biến độc lập.
### Các bước thực hiện
-  **Bước 1** : Khai báo statsmodels.api để sử dụng hàm OLS.OLS là một hàm trong statsmodels.api dùng để thực hiện mô hình hồi quy tuyến tính.
- **Bước 2** : Khai báo included là một list chứa các biến độc lập đã được lựa chọn.(Có thể rỗng)
- **Bước 3.1** : Sử dụng vòng lặp while để thực hiện việc thêm 
    - **changed** = False : biến đánh dấu việc thêm 
    - **Exculed** = X - included : Các biến độc lập chưa được lựa chọn.
    - **new-pval** new_pval = pd.Series(index=excluded) : có độ lớn bằng số lượng các biến độc lập chưa được lựa chọn. Để lưu giá trị p-value. Ban đầu tất cả các giá trị đều bằng 0.
    -  **For** Duyệt qua các biến độc lập sau khi thêm và gán trị p-value cho các biến độc lập đó. (new_pval)
    - Lấy pvalue của biến độc lập có giá trị nhỏ nhất trong new_pval.
        - Bé hơn alpha(0.05) thì thêm biến độc lập đó vào included.
        - changed = True tiếp tục vòng lặp.
    - **If** changed = False thì dừng vòng lặp.(Không có p-value nào nhỏ hơn alpha)
- **Bước 3.2** : Xóa dần khỏi include list
    - Điều kiện dừng : Không thể loại bỏ biến độc lập nào khác.
    - Lấy danh sách các p-value của các biến độc lập trong included. 
    - Loai bỏ biến có p-value lớn nhất trong included. thỏa mãn điều kiện p-value > alpha(0.1)
    - Không xét tới giá trị đầu tiên của series pval.
### code
```python
X = pd.DataFrame({
    'Feature1': [1, 2, 3, 4, 5],
    'Feature2': [1, 5, 6, 8, 10],
    'Feature3': [3, 4, 5, 6, 7],
    'Feature4': [6, 8, 12, 16, 20]
})
y = pd.Series([5, 6, 7, 8, 9])

```

```python
import pandas as pd
import statsmodels.api as sm

def Stepwise_regision_1(X,y,init_list = [],check_in = 0.05,check_out = 0.1):
    included = list(init_list)
    ## ADD LIST
    while True:
        changed = False
        excluded = set(X.columns) - set(included)
        new_pval = pd.Series(index=excluded)
        for feature in excluded:
            model = sm.OLS(y,sm.add_constant(X[included + [feature]])).fit()
            new_pval[feature] = model.pvalues[feature]
        best_pval = new_pval.min()
        
        if(best_pval<check_in):
            changed = True
            included.append(new_pval.idxmin())
            print(f'Added feature: {new_pval.idxmin()}, p-value: {best_pval:.4f}')
            
    ##   REMOVE LIST   
    
        model = sm.OLS(y,sm.add_constant(X[included])).fit()
        new_pval = model.pvalues.iloc[1:]
        max_pval = new_pval.max()
        if max_pval > check_out:
            changed = True
            included.remove(new_pval.idxmax())
            print(f'Remove feature: {new_pval.idxmax()}, p-value: {max_pval:.4f}')
            
        if not changed:
            break
            
    return model, included
```

```python
model, result = Stepwise_regision_1(X,y)
y_predicted = model.predict(sm.add_constant(X[feature]))
```

```python
Added feature: Feature3, p-value: 0.0000
Remove feature: Feature2, p-value: 0.9835
['Feature1', 'Feature3']
```

- **Link Code** : [Stepwise Regression in Python](https://www.kaggle.com/quctngngvng/stepwise-regision)







<!-- import statsmodels.api as sm

# Fit the regression model
model = sm.OLS(y, X).fit()

# Get the coefficients
coefficients = model.params

# Get the standard errors of coefficients
std_errors = model.bse -->

<!-- 
t-test = coefficient/standard error
p-value = P(|t| > 1.96) = 2 * P(t > 1.96) = 2 * cdf(-t(1.96)) -->

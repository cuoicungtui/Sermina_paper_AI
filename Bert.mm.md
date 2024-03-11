# Bert
## output 
-  pooled_output: shape = (batch_size, hidden_size)
-  last_hidden_state: shape = (batch_size, sequence_length, hidden_size) 
-  hidden_states: (num_layer+1) * shape_in_one_layer = (batch_size, sequence_length, hidden_size)(Thường num_layer = 12 or 24)

## GlobalAveragePooling1d(x, mask=mask).
- tổng hợp toàn bộ trạng thái ẩn của bert kết hợp với mask
- link: [link](https://discuss.huggingface.co/t/bert-output-for-padding-tokens/1550/4)
-  input 
    - output[0] của bert model shape = (batch_size, sequence_length, hidden_size)
    - mask shape = (batch_size, sequence_length)
    - x = GlobalAveragePooling1D()(output[0], encoded_input["attention_mask"])
    - x shape = (batch_size, hidden_size)
- python
```python
from keras.layers import GlobalAveragePooling1D
from transformers import BertTokenizer, TFBertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertModel.from_pretrained("bert-base-uncased")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='tf')
output = model(encoded_input)

x = GlobalAveragePooling1D()(output[0], encoded_input["attention_mask"]) 
```
## Thường cho kết quả kém hơn so với [CLS] (pooled_output)
## Đề xuất sử dụng Tổng hợp 4 trạng thái ẩn cuối cùng của bert
```python
outputs = self.bert(input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids, 
                    head_mask=head_mask)

hidden_states = outputs[1]
pooled_output = torch.cat(tuple([hidden_states[i] for i in [-4, -3, -2, -1]]), dim=-1)
pooled_output = pooled_output[:, 0, :]
pooled_output = self.dropout(pooled_output)
# classifier of course has to be 4 * hidden_dim, because we concat 4 layers
logits = self.classifier(pooled_output)
```
## [Sequence Classification pooled output vs last hidden state](https://lightrun.com/answers/huggingface-transformers-sequence-classification-pooled-output-vs-last-hidden-state)
## Quote
- Trong nhiệm vụ phân loại
    -   link: [link](https://github.com/huggingface/transformers/issues/4048)
    - Kết quả viêc sử dụng pooled output cho kết quả kém hơn so với **TFBertForSequenceClassification** (70%,73%) /  79%
    - Trong nhiệm vụa bsa-vlsp-2018 link: [link](https://github.com/ds4v/absa-vlsp-2018)
    - python
```python
    
```
- Khai báo bert model [lib ](https://pypi.org/project/transformers/)
    - python  Pythorch version
```
>>> from transformers import AutoTokenizer, AutoModel

>>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
>>> model = AutoModel.from_pretrained("bert-base-uncased")

>>> inputs = tokenizer("Hello world!", return_tensors="pt")
>>> outputs = model(**inputs)
```
- python Tensorflow version
```
>>> from transformers import AutoTokenizer, TFAutoModel
>>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
>>> model = TFAutoModel.from_pretrained("bert-base-uncased")

>>> inputs = tokenizer("Hello world!", return_tensors="tf")
>>> outputs = model(**inputs)
```

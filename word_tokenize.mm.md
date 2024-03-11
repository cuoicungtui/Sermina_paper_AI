# word_tokenize
## NLTK 
python code
```python
import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize

text = "This is a sentence."
tokens = word_tokenize(text)
print(tokens)  # Output: ["This", "is", "a", "sentence", "."]
```

## huffingface
- Ví dụ sử dụng tokenizers từ Hugging Face để thực hiện subword tokenizer:
- Lưu ý rằng các tokenizer có thể cần được huấn luyện trước khi sử dụng, đặc biệt là với các tokenizer dựa trên mô hình ngôn ngữ (language model-based tokenizers) như tokenizers từ Hugging Face.
python code
```python
from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files="data.txt", vocab_size=1000, min_frequency=2)

text = "apple"
tokens = tokenizer.encode(text).tokens
print(tokens)  # Output: ["ap", "ple"]
```
- Embedding 
python code
```python
from tokenizers import ByteLevelBPETokenizer

# Huấn luyện tokenizer
tokenizer = ByteLevelBPETokenizer()
files = ["data.txt"]
tokenizer.train(files, vocab_size=1000, min_frequency=2)

# Lấy từ điển và số chiều của embeddings
vocab = tokenizer.get_vocab()
embedding_dim = tokenizer.get_vocab_size()

# Lấy word embedding của một từ hoặc subword
word_or_subword = "apple"
embedding = vocab[word_or_subword]

print("Word or Subword:", word_or_subword)
print("Word Embedding:", embedding)
print("Embedding Dimension:", embedding_dim)
```
- BYTETOKENIZER
python code
```python

import keras_nlp.tokenizers.ByteTokenizer()
>>> inputs = ["hello", "hi"]
>>> tokenizer = keras_nlp.tokenizers.ByteTokenizer(sequence_length=8)
>>> seq1, seq2 = tokenizer(inputs)
>>> np.array(seq1)
array([104, 101, 108, 108, 111,   0,   0,   0], dtype=int32)
>>> np.array(seq2)
array([104, 105,   0,   0,   0,   0,   0,   0], dtype=int32)
```
-  Embeding glove 
```
import tqdm

path_glove = "/kaggle/input/glovedata/glove.6B.200d.txt"
EMBEDDING_VECTOR_DIM = 200

def contruct_embedding_matrix(glove_file,word_index):
    embedding_dict = {}
    with open(glove_file,'r') as f:
        for line in f:
            value = line.split()
            word = value[0]
            if word in word_index.keys():
                vector = np.asarray(value[1:],'float32')
                embedding_dict[word] = vector
    num_word = len(word_index)+1
    
    embedding_matrix = np.zeros((num_word, EMBEDDING_VECTOR_DIM))
    
    for word,i in tqdm.tqdm(word_index.items()):
        if i < num_word:
            embedding_vector = embedding_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector[:EMBEDDING_VECTOR_DIM]
    return embedding_matrix

embedding_matrix = contruct_embedding_matrix
(path_glove,vocab_dic)


embedding = Embedding(input_dim = vocab_size,
                      output_dim = input_dim,
                      input_length = max_len,
                      embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix), 
                      trainable = True
                     )(input_layer1)
```
- Custom loss
```
class CustomCrossEntropy(Loss):
    def __init__(self, name='custom_cross_entropy'):
        super(CustomCrossEntropy, self).__init__(name=name)

    @tf.function
    def call(self, y_true, y_pred):    
        sum_loss = 0.0
        for i in range(len(y_true)):
            y_true_clean = y_true[i][y_true[i] != -100]
            y_pred_clean = y_pred[i][y_true[i] != -100]
            y_true_one_hot = tf.one_hot(tf.cast(y_true_clean, tf.int32), depth=n_tags)
            loss =tf.reduce_sum(tf.keras.losses.categorical_crossentropy(y_true_one_hot, y_pred_clean)) 
            sum_loss+=loss
            
        return sum_loss/tf.cast(len(y_true), dtype=tf.float32) 


model.compile(optimizer = adam,loss = CustomCrossEntropy(),metrics = ['accuracy'])
```

- Hiện thị Model plot 
```
tf.keras.utils.plot_model(model, show_shapes=True, dpi=48)
```

- Split Num_head keras
```
def Model2(max_len,embed_dim,vocab_size,idx_taget,embedding_matrix):
    vocab_size = vocab_size+1
    input_dim = embed_dim
    output_dim = 50
    max_len = max_len
    pad_len = max_len
    drop_out = 0.5
    n_tags = len(idx_taget)
    n_heads = 4
    embedding_matrix = embedding_matrix 
    lr = 1e-3
    pos_len = len(pos_tags_id)+1

    
    input_layer1 = Input(shape = max_len,name = 'Sequence id')
    input_post   = Input(shape = max_len,name =  'Pos sequen id')
    embedding = Embedding(input_dim = vocab_size,
                          output_dim = input_dim,
                          input_length = max_len,    
                          trainable = True,
                         )(input_layer1)
    
    embed_pos = Embedding(input_dim = pos_len,
                         output_dim = 50,
                         input_length = max_len,
                         trainable = True)(input_post)
    
    head_atten_pos = tf.keras.layers.Attention()([embed_pos,embed_pos])
    head_atten_pos =  LayerNormalization(epsilon=0.001)(head_atten_pos+embed_pos)
    
    embedding_n_heads = tf.reshape(embedding,(tf.shape(embedding)[0],max_len,n_heads,input_dim//n_heads))
    
    embedding_trainpost = tf.transpose(embedding_n_heads, [0, 2, 1, 3]) 
    
    head_tensors = tf.split(embedding_trainpost, num_or_size_splits=n_heads, axis=1)
    embed1 = tf.squeeze(head_tensors[0],axis = 1)
    embed2 = tf.squeeze(head_tensors[1],axis = 1)#(batch_size,max_len,dim//head = 50 )
    embed3 = tf.squeeze(head_tensors[2],axis = 1)
    embed4 = tf.squeeze(head_tensors[3],axis = 1)
    
    head_Sequential_BI_LSTM = Sequential(name="sequential_BILSTM")  # output(batch_size,max_len,150)
    head_Sequential_BI_LSTM.add(Bidirectional((LSTM(units = output_dim//2,dropout = drop_out,
                                            return_sequences = True)),merge_mode  = 'concat'))
    
    head_Sequential_CNN = Sequential(name="sequential_CNN") # output(batch_size,max_len,150)
    head_Sequential_CNN.add( Conv1D(100,2,activation = 'relu',padding='same'))
    head_Sequential_CNN.add(LayerNormalization(epsilon=0.001))
    head_Sequential_CNN.add(GaussianDropout(drop_out))
    head_Sequential_CNN.add( Conv1D(150,3,activation = 'relu',padding='same'))
    head_Sequential_CNN.add(LayerNormalization(epsilon=0.001))
    head_Sequential_CNN.add(GaussianDropout(drop_out))
    head_Sequential_CNN.add( Conv1D(50,5,activation = 'relu',padding='same'))
    head_Sequential_CNN.add(LayerNormalization(epsilon=0.001))
    head_Sequential_CNN.add(GaussianDropout(drop_out))
    
    head1 = head_Sequential_BI_LSTM(embed1)
    head2 = head_Sequential_CNN(embed2)
    
    head3 =  tf.keras.layers.Attention()([head2,embed4])
    head3 =  LayerNormalization(epsilon=0.001)(head3+embed4)
    
    head4 = tf.keras.layers.Attention()([head1,embed3])
    head4 =  LayerNormalization(epsilon=0.001)(head4+embed3)
    
#     print(head4.shape)
#     print(head3.shape)

#     expand_dim_head1 = tf.expand_dims(head1, axis = 1)
#     expand_dim_head2 = tf.expand_dims(head2, axis = 1)

    expand_dim_head1 = tf.expand_dims(head_atten_pos, axis = 1)
    expand_dim_head3 = tf.expand_dims(head3, axis = 1)
    expand_dim_head4 = tf.expand_dims(head4, axis = 1)
    
    concat_layer = Concatenate(axis=1)
    concatenated_tensor = concat_layer([expand_dim_head1,expand_dim_head3, expand_dim_head4])
    trainspose_tensor = tf.transpose(concatenated_tensor, [0, 2, 1, 3]) 
    reshape_tensor = tf.reshape(trainspose_tensor,(tf.shape(trainspose_tensor)[0],max_len,(n_heads-1)*tf.shape(trainspose_tensor)[-1]))
    reshape_tensor = LayerNormalization(epsilon=0.001)(reshape_tensor)
    
    dense_layer = Sequential(name="sequential_Dense")
    dense_layer.add(Dense(200,activation = 'relu'))
    dense_layer.add(Dropout(drop_out))
    dense_layer.add(Dense(100,activation = 'relu'))
    dense_layer.add(Dropout(drop_out))
    dense_layer.add(Dense(50,activation = 'relu'))
    dense_layer.add(Dropout(drop_out))
    dense_layer.add(Dense(n_tags,activation = 'softmax'))

    classifier = TimeDistributed(dense_layer)(reshape_tensor)
    model = Model(inputs = [input_layer1,input_post],outputs = classifier )
    adam = Adam(learning_rate = lr)
    model.compile(optimizer = adam,loss = CustomCrossEntropy(),metrics = ['accuracy'])
    return model

```
# 情感分析

本节介绍如何使用gluon来创建一个情感分类模型，目标是给定一句话，判断这句话包含的是“正面”还是“负面”的情绪。为此，我们构造了一个简单的神经网络，其中包括embedding层，biLSTM层以及dense层。下面就让我们一起来构造一个情感分析模型吧。


## 准备工作

### 加载mxnet和gluon

```{.python .input}
import mxnet as mx
from mxnet import gluon, autograd
from collections import Counter
```

### 读取IMDB

我们首先需要下载数据集。我们使用Stanford's Large Movie Review Dataset作为数据集，下载地址：http://ai.stanford.edu/~amaas/data/sentiment/ 。这个数据集有25,000条从IMDB下载的关于电影的评论，其中12,500条被标注成“正面”的评论，另外12,500条被标注成“负面”的评论。

```{.python .input}
def readIMDB(file_url):
    #TODO:
    import os
    reviews = []
    files = os.listdir(file_url)
    for file in files:
        with open(file,'r',encoding='utf8') as rf:
            review = rf.read().replace('\n','')
            reviews.append(review)
    return reviews

file_url = 'aclImdb/train/'
pos_reviews, neg_reviews = readIMDB(file_url)
```

### 创建词典

```{.python .input}
token_counter = Counter()
def count_token(reviews):
    for review in reviews:
        for token in review.split():
            if token not in token_counter:
                token_counter[token] = 1
            else:
                word_counter[token] += 1

def token_to_idx():
    idx = 0
    token_dict = {}
    for token in token_counter.most_common():
        token_dict[word[0]] = idx
        idx += 1
    return token_dict
```

### 指定分词工具并且分词

```{.python .input}
import spacy
spacy_en = spacy.load('en')

def tokenizer(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]
```

```{.python .input}
train_tokenized = []
train_labels = []
for text, score in train:
    train_tokenized.append(tokenizer(text))
    train_labels.append(score)
test_tokenized = []
test_labels = []
for text, score in test:
    test_tokenized.append(tokenizer(text))
    test_labels.append(score)
```

### 将分好词的数据转化成ndarray

```{.python .input}
#根据词典，将数据转换成特征向量
def encode_samples(x_raw_samples, vocab):
    x_encoded_samples = []
    for sample in x_raw_samples:
        x_encoded_sample = []
        for word in sample:
            if word in vocab.token_to_idx:
                x_encoded_sample.append(vocab.token_to_idx[word])
            else:
                x_encoded_sample.append(0)
        x_encoded_samples.append(x_encoded_sample)            
    return x_encoded_samples

#生成分类标签
def encode_labels(y_raw_samples):
    y_encoded_samples = []
    for score in y_raw_samples:
        if score >= 7:
            y_encoded_samples.append(1)
        elif score <= 4:
            y_encoded_samples.append(0)
    return y_encoded_samples

#将特征向量补成定长
def pad_samples(x_encoded_samples, maxlen = 500, val = 0):
    x_samples = []
    for sample in x_encoded_samples:
        if len(sample) > maxlen:
            new_sample = sample[:maxlen]
        else:
            num_padding = maxlen - len(sample)
            new_sample = sample
            for i in range(num_padding):
                new_sample.append(val)
        x_samples.append(new_sample)
    return x_samples
```

```{.python .input}
x_encoded_train = encode_samples(train_tokenized, vocab)
x_encoded_test = encode_samples(test_tokenized, vocab)
```

```{.python .input}
#指定context
context = mx.gpu(0)
x_train = mx.nd.array(pad_samples(x_encoded_train, 500, 0), ctx = context)
x_test = mx.nd.array(pad_samples(x_encoded_test, 500, 0), ctx = context)
```

```{.python .input}
y_train = mx.nd.array(encode_labels(train_labels), ctx = context)
y_test = mx.nd.array(encode_labels(test_labels), ctx = context)
```

## 创建情感分析模型

神经网络结构比较简单，如下图所示。

<img src="phuong huong xac dinh.PNG">

```{.python .input}
nclass = 2
lr = 0.001
epochs = 1
batch_size = 1
bi = True

model = gluon.nn.Sequential()
with model.name_scope():
    #TODO
    model.add(gluon.nn.embedding(len(token_dict), 200))
    #可以使用bidirectional的选项
    model.add(gluon.rnn.LSTM(200, 1, bidirectional=bi))
    model.add(gluon.nn.HybridLambda('SequenceLast'))
    model.add(gluon.nn.Dense(nclass, flatten=False))

model.initialize(mx.init.Xavier(), ctx = context)
trainer = gluon.Trainer(model.collect_params(), 'sgd',
                       {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

## 训练模型

```{.python .input}
#使用accuracy作为评价指标
def eval(x_samples, y_samples):
    accuracy = mx.metric.Accuracy()
    for i, data in enumerate(x_samples[:1000]):
        data = mx.nd.reshape(data, (-2, batch_size)).as_in_context(context)
        target = y_samples[i].as_in_context(context)
        output = model(data)
        predicts = mx.nd.argmax(output, axis=1)
        accuracy.update(preds=predicts, labels=target)
    return accuracy.get()[1]
```

```{.python .input}
for epoch in range(epochs):
    for i, data in enumerate(x_train[:1000]):
        data = mx.nd.reshape(data, (-2, batch_size)).as_in_context(context)
        target = y_train[i].as_in_context(context)
        with autograd.record():
            output = model(data)
            L = loss(output, target)
        L.backward()
        trainer.step(batch_size)
        if i % 100 == 0:
            print("Batch %s. loss %s"%(i, L.asnumpy()))
    train_accuracy = eval(x_train, y_train)
    test_accuracy = eval(x_test, y_test)
    print("Epoch %s. Train_acc %s, Test_acc %s"%(epoch, train_accuracy, test_accuracy))
```

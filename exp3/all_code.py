# numpy sklearn jieba gensim
import numpy as np
from sklearn.model_selection import train_test_split
import re
import string
import jieba
from sklearn.feature_extraction.text import CountVectorizer  #载入词袋模型特征
from sklearn.feature_extraction.text import TfidfTransformer  #载入Tfidf转换器
from sklearn.feature_extraction.text import TfidfVectorizer  #载入Tfidf特征
import gensim
from sklearn.naive_bayes import MultinomialNB  #导入多项朴素贝叶斯分类算法
from sklearn.linear_model import SGDClassifier  #导入随机梯度下降分类器
from sklearn.linear_model import LogisticRegression  #导入逻辑回归分类器
from sklearn import metrics


def get_data():
    """
    获取数据
    :return：文本数据，对应的labels
    """
    with open('./dataset/data/ham_data.txt', encoding='utf-8') as ham_f, open('./dataset/data/spam_data.txt',
                                                                              encoding='utf-8') as spam_f:
        ham_data = ham_f.readlines()
        spam_data = spam_f.readlines()

        ham_label = np.ones(len(ham_data)).tolist()
        spam_label = np.zeros(len(spam_data)).tolist()

        corpus = ham_data + spam_data

        labels = ham_label + spam_label

    return corpus, labels


def prepare_datasets(corpus, labels, test_data_proportion=0.3):
    """
    :param corpus:
    :param labels:
    :param test_data_proportion:
    :return:
    """
    train_x, test_x, train_y, test_y = train_test_split(corpus, labels, test_size=test_data_proportion, random_state=42)
    return train_x, test_x, train_y, test_y


def remove_empty_docs(corpus, labels):
    filtered_corpus = []
    filtered_labels = []
    for doc, label in zip(corpus, labels):
        if doc.strip():
            filtered_corpus.append(doc)
            filtered_labels.append(label)

    return filtered_corpus, filtered_labels


corpus, labels = get_data()

print(f'总的数据量：{len(labels)}')

corpus, labels = remove_empty_docs(corpus, labels)

print(f'样本之一：{corpus[10]}')
print(f'样本的label：{labels[10]}')
label_name_map = ['垃圾邮件', '正常邮件']
print(f'实际类型：{label_name_map[int(labels[10])]},'
      f'{label_name_map[int(labels[5900])]}')

# 对数据进行划分
train_corpus, test_corpus, train_labels, test_labels = prepare_datasets(corpus, labels, test_data_proportion=0.3)

#%%
with open('./dataset/stop_words.utf8', encoding='utf-8') as f:
    stopword_list = f.readlines()


# jieba分词
def tokenize_text(text):
    tokens = jieba.lcut(text)
    tokens = [token.strip() for token in tokens]
    return tokens


# 去除特殊字符
def remove_special_characters(text):
    tokens = tokenize_text(text)
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


# 去除停用词
def remove_stopwords(text):
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


# 当tokenize=True时候，会对文本进行分词，除此以外还会对文本进行去除特殊符号、停用词等预处理
def normalize_corpus(corpus, tokenize=False):
    normalized_corpus = []
    for text in corpus:
        text = remove_special_characters(text)
        text = remove_stopwords(text)
        normalized_corpus.append(text)
        if tokenize:
            text = tokenize_text(text)
            normalized_corpus.append(text)
    return normalized_corpus


#%%
def bow_extractor(corpus, ngram_range=(1, 1)):
    vectorizer = CountVectorizer(min_df=1, ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features


# Tfidf特征转换
def tfidf_transformer(bow_matrix):
    transformer = TfidfTransformer(norm='l2', smooth_idf=True, use_idf=True)
    tfidf_matrix = transformer.fit_transform(bow_matrix)
    return transformer, tfidf_matrix


# tfidf特征提取
def tfidf_extractor(corpus, ngram_range=(1, 1)):
    vectorizer = TfidfVectorizer(min_df=1, norm='l2', smooth_idf=True, use_idf=True, ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features


# 进行归一化
norm_train_corpus = normalize_corpus(train_corpus)
norm_test_corpus = normalize_corpus(test_corpus)

# 词袋模型特征
bow_vectorizer, bow_train_features = bow_extractor(norm_train_corpus)
bow_test_features = bow_vectorizer.transform(norm_test_corpus)

# tf-idf特征
tfidf_vectorizer, tfidf_train_features = tfidf_extractor(norm_train_corpus)
tfidf_test_features = tfidf_vectorizer.transform(norm_test_corpus)

# 训练数据tokenize化
tokenized_train = [jieba.lcut(text) for text in norm_train_corpus]
print(tokenized_train[2:10])
tokenized_test = [jieba.lcut(text) for text in norm_test_corpus]

# build word2vec模型
model = gensim.models.Word2Vec(tokenized_train, window=100, min_count=30, sample=1e-3)

mnb = MultinomialNB()
svm = SGDClassifier(loss='hinge')
lr = LogisticRegression()


# 首先声明了mnb, svm, lr 三个分类器，然后传入train_predict_evaluate_model函数中
def train_predict_evaluate_model(classifier, train_features, train_labels, test_features, test_labels):
    # build model
    classifier.fit(train_features, train_labels)
    # predict using model
    predictions = classifier.predict(test_features)
    # evaluate model prediction performance
    get_metrics(true_labels=test_labels, predicted_labels=predictions)
    return predictions


def get_metrics(true_labels, predicted_labels):
    print('Accuracy:', np.round(metrics.accuracy_score(true_labels, predicted_labels), 2))
    print('Precision:', np.round(metrics.precision_score(true_labels, predicted_labels, average='weighted'), 2))
    print('F1 Score:', np.round(metrics.f1_score(true_labels, predicted_labels, average='weighted'), 2))


#%%
print('基于词袋模型的多项朴素贝叶斯')
mnb_bow_predictions = train_predict_evaluate_model(classifier=mnb, train_features=bow_train_features,
                                                   train_labels=train_labels, test_features=bow_test_features,
                                                   test_labels=test_labels)

# 输出结果：
# 自己跑

# 基于词袋模型特征的逻辑回归
print('基于词袋模型特征的逻辑回归')
lr_bow_predictions = train_predict_evaluate_model(classifier=lr, train_features=bow_train_features,
                                                  train_labels=train_labels, test_features=bow_test_features,
                                                  test_labels=test_labels)
# 输出结果：
# 自己跑

# 基于词袋模型的支持向量机方法
print('基于词袋模型的支持向量机方法')
svm_bow_predictions = train_predict_evaluate_model(classifier=svm, train_features=bow_train_features,
                                                   train_labels=train_labels, test_features=bow_test_features,
                                                   test_labels=test_labels)

# 输出结果：
# 自己跑

# 基于tf-idf特征的多项朴素贝叶斯
print('基于tf-idf特征的多项朴素贝叶斯')
mnb_tfidf_predictions = train_predict_evaluate_model(classifier=mnb, train_features=tfidf_train_features,
                                                     train_labels=train_labels, test_features=tfidf_test_features,
                                                     test_labels=test_labels)

# 输出结果：
# 自己跑

# 基于tf-idf特征的逻辑回归
print('基于tf-idf特征的逻辑回归')
lr_tfidf_predictions = train_predict_evaluate_model(classifier=lr, train_features=tfidf_train_features,
                                                    train_labels=train_labels, test_features=tfidf_test_features,
                                                    test_labels=test_labels)

# 输出结果：
# 自己跑

# 基于tf-idf特征的支持向量机方法
print('基于tf-idf特征的支持向量机方法')
svm_tfidf_predictions = train_predict_evaluate_model(classifier=svm, train_features=tfidf_train_features,
                                                     train_labels=train_labels, test_features=tfidf_test_features,
                                                     test_labels=test_labels)

#%%
num = 0
for document, label, predicted_label in zip(test_corpus, test_labels, svm_tfidf_predictions):
    if label == 0 and predicted_label == 0:
        print('邮件类型：', label_name_map[int(label)])
        print('预测的邮件类型：', label_name_map[int(predicted_label)])
        print('文本：-')
        print(re.sub('\n', '', document))
        num += 1
        if num == 4:
            break

num = 0
for document, label, predicted_label in zip(test_corpus, test_labels, svm_tfidf_predictions):
    if label == 1 and predicted_label == 1:
        print('邮件类型：', label_name_map[int(label)])
        print('预测的邮件类型：', label_name_map[int(predicted_label)])
        print('文本：-')
        print(re.sub('\n', '', document))
        num += 1
        if num == 4:
            break

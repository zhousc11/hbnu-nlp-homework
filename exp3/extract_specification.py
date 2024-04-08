# 载入Tfidf特征
import gensim
import jieba
import numpy as np
# 导入逻辑回归分类器
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
# 载入Tfidf转换器
from sklearn.feature_extraction.text import TfidfTransformer
# 载入词袋模型特征
from sklearn.feature_extraction.text import TfidfVectorizer
# 导入随机梯度下降分类器
from sklearn.linear_model import LogisticRegression
# 导入多项朴素贝叶斯分配算法
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB

from exp3.dataprocess import train_corpus, test_corpus
from exp3.stopword_filter import normalize_corpus


# 词袋模型特征提取
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

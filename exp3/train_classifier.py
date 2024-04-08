# 基于词袋模型的多项朴素贝叶斯
from sklearn import svm

from exp3.dataprocess import train_labels, test_labels
from exp3.extract_specification import train_predict_evaluate_model, bow_train_features, mnb, bow_test_features, lr, \
    tfidf_train_features, tfidf_test_features

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

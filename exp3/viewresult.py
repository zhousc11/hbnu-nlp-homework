import re

from exp3.dataprocess import test_corpus, test_labels, label_name_map
from exp3.train_classifier import svm_tfidf_predictions

num = 0
for document, label, predicted_label in zip(test_corpus, test_labels, svm_tfidf_predictions):
    if label == 0 and predicted_label == 0:
        print('邮件类型：', label_name_map[int(label)])
        print('预测的邮件类型：', label_name_map[int(predicted_label)])
        print('文本：-')
        print(re.sub('\n', '', document))
        num += 1
        if num== 4:
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
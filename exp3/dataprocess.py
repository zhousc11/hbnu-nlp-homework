# numpy sklearn jieba gensim
import numpy as np
from sklearn.model_selection import train_test_split


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

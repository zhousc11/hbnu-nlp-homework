import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text
from keras.integration_test.preprocessing_test_utils import preprocessing
from matplotlib import pyplot as plt

from exp4.processdatasets import inp, targ, example_input_batch

example_text = tf.constant('¿Todavía está en casa?')

print(example_text.numpy())
print(tf_text.normalize_utf8(example_text, 'NFKD').numpy())


def tf_lower_and_split_punct(text):
    # 对字符进行切分
    text = tf_text.normalize_utf8(text, 'NFKD')
    text = tf.strings.lower(text)
    # 保持空格，从a到z，并选择标点符号。
    text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')
    # 在标点符号周围添加空格。
    text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
    # 去空格。
    text = tf.strings.strip(text)
    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
    return text


print(example_text.numpy().decode())
# 输出解码结果(德语)
print(tf_lower_and_split_punct(example_text).numpy().decode())
# 对句子进行头尾标注
# 输出结果：
# ¿Todavía está en casa?
# [START] ¿ todavia esta en casa ? [END]


max_vocab_size = 5000

input_text_processor = preprocessing.TextVectorization(
    standardize=tf_lower_and_split_punct,
    max_tokens=max_vocab_size)

input_text_processor.adapt(inp)
# this is first 10 words in vocabulary
print(input_text_processor.get_vocabulary()[:10])

output_text_processor = preprocessing.TextVectorization(
    standardize=tf_lower_and_split_punct,
    max_tokens=max_vocab_size)
output_text_processor.adapt(targ)
print(output_text_processor.get_vocabulary()[:10])

example_tokens = input_text_processor(example_input_batch)
print(example_tokens[:3, :10])

input_vocab = np.array(input_text_processor.get_vocabulary())
tokens = input_vocab[example_tokens[0].numpy()]
' '.join(tokens)

plt.subplot(1, 2, 1)
plt.pcolormesh(example_tokens)
plt.title('Token IDs')
plt.subplot(1, 2, 2)
plt.pcolormesh(example_tokens != 0)
plt.title('Mask')

#嵌入维度
embedding_dim = 256
#隐藏单元个数
units = 1024

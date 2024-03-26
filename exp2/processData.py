# 提炼数据集，将标记为ns的提取出来，作为地名识别语料
def tag_line(words, mark):
    chars = []
    tags = []
    temp_word = ''
    for word in words:
        word = word.strip('\t ')
        if temp_word == '':
            bracket_pos = word.find('[')
            w, h = word.split('/')
            if bracket_pos == -1:
                if len(w) == 0:
                    continue
                chars.extend(w)
                if h == 'ns':
                    tags += ['s'] if len(w) == 1 else ['B'] + ['M'] * (len(w) - 2) + ['E']
                else:
                    tags += ['0'] * len(w)
            else:
                w = w[bracket_pos + 1:]
                temp_word += w
        else:
            bracket_pos = word.find(']')
            w, h = word.split('/')
            if bracket_pos == -1:
                temp_word += w
            else:
                w = temp_word + w
                h = word[bracket_pos + 1:]
                temp_word = ''
                if len(w) == 0:
                    continue
                chars.extend(w)
                if h == 'ns':
                    tags += ['s'] if len(w) == 1 else ['B'] + ['M'] * (len(w) - 2) + ['E']
                else:
                    tags += ['0'] * len(w)
    assert temp_word == ''
    return chars, tags


# 划分成训练集和测试集
def corpusHandler(corpusPath):
    import os
    root = os.path.dirname(corpusPath)
    with open(corpusPath, encoding='UTF-8') as corpus_f, \
            open('./train.txt', 'w+', encoding='UTF-8') as train_f, \
            open('./test.txt', 'w+', encoding='UTF-8') as test_f:
        pos = 0
        for line in corpus_f:
            line = line.strip('\r\n\t')
            if line == '':
                continue
            isTest = True if pos % 5 == 0 else False
            words = line.split()[1:]
            word = line.split()[1:]
            if len(words) == 0:
                continue
            line_chars, line_tags = tag_line(words, pos)
            saveobj = test_f if isTest else train_f
            for k, v in enumerate(line_chars):
                saveobj.write(v + '\t' + line_tags[k] + '\n')
            saveobj.write('\n')
            pos += 1


corpusHandler('1980_01rmrb.txt')

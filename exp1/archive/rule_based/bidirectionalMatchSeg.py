from forwardMatchSeg import forward_max_match
from backwardMatchSeg import backward_max_match
# import segmentation dictionary
from dictionary import dictionary


def bidirectional_max_match(sentence, dictionary, max_word_length):
    # 正向最大匹配分词
    forward_result = forward_max_match(sentence, dictionary, max_word_length)
    # 反向最大匹配分词
    backward_result = backward_max_match(sentence, dictionary, max_word_length)

    # 比较正向和反向分词结果的分词数量，选择分词数量较少的作为最终结果
    # 如果分词数量相同，则可以进一步基于其他规则选择，这里简单地选择正向分词结果
    if len(forward_result) > len(backward_result):
        return backward_result
    elif len(forward_result) < len(backward_result):
        return forward_result
    else:
        return forward_result


# Read the text from file
if __name__ == '__main__':
    try:
        with open('../../NBA.txt', 'r', encoding='utf-8') as f:
            text = f.read()

        max_word_length = max(len(word) for word in dictionary)

        # 移除文本中的空格和特殊字符
        text = text.replace('\u3000', '').replace('\n', '')

        # 分词
        result = bidirectional_max_match(text, dictionary, max_word_length)

        # output the result
        print('/'.join(result))

    except FileNotFoundError:
        print(f'The file "NBA.txt" does not exist in the current directory. \nDownload the file from Chaoxing course '
              f'and'
              f'try again.')
    except Exception as e:
        print(f'Unexpected error occurred: {e}')
    except PermissionError:
        print(f'Permission denied. Please check if you have the permission to read the file.')

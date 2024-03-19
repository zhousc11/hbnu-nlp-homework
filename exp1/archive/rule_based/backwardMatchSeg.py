# import segmentation dictionary
from dictionary import dictionary


def backward_max_match(sentence, dictionary, max_word_length):
    index = len(sentence)
    result = []
    while index > 0:
        word = None
        for size in range(max_word_length, 0, -1):
            if index - size < 0:  # 确保不会向左越界
                continue
            piece = sentence[index - size:index]  # 从右向左获取子字符串
            if piece in dictionary:
                word = piece
                result.insert(0, word)  # 将匹配到的词插入到结果列表的开头
                index -= size
                break
        if word is None:  # 如果没有在字典中找到匹配的词
            word = sentence[index - 1]
            result.insert(0, word)  # 将单个字符插入到结果列表的开头
            index -= 1
    return result


# Read the text from file
if __name__ == '__main__':
    try:
        with open('../../NBA.txt', 'r', encoding='utf-8') as f:
            text = f.read()

        max_word_length = max(len(word) for word in dictionary)

        # 移除文本中的空格和特殊字符
        text = text.replace('\u3000', '').replace('\n', '')

        # 分词
        result = backward_max_match(text, dictionary, max_word_length)

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

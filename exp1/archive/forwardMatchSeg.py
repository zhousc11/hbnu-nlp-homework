# import segmentation dictionary
from dictionary import dictionary


def forward_max_match(sentence, dictionary, max_word_length):
    index = 0
    result = []
    while index < len(sentence):
        word = None
        for size in range(max_word_length, 0, -1):
            if index + size > len(sentence):
                continue
            piece = sentence[index:index + size]
            if piece in dictionary:
                word = piece
                result.append(word)
                index += size
                break
        if word is None:
            word = sentence[index]
            result.append(word)
            index += 1
    return result


# Define the segment dict

# Read the text from file
if __name__ == '__main__':
    try:
        with open('../NBA.txt', 'r', encoding='utf-8') as f:
            text = f.read()

        max_word_length = max(len(word) for word in dictionary)

        # 移除文本中的空格和特殊字符
        text = text.replace('\u3000', '').replace('\n', '')

        # 分词
        result = forward_max_match(text, dictionary, max_word_length)

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

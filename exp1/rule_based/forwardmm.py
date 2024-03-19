from exp1.archive.rule_based.dictionary import dictionary


class MM(object):
    def __init__(self):
        # define maximum number of window size
        # set to 6 as 凯文-杜兰特
        self.window_size = 6
        # define segmentation function

    def cut(self, text):
        # store segmentation result
        result = []
        # current processing position
        index = 0
        text_length = len(text)
        # define match dictionary
        dic = dictionary
        # loop match
        while text_length > index:
            # forward match process
            for size in range(self.window_size + index, index, -1):
                # get processing text
                piece = text[index:size]
                # if match word in dict, navigate by the matched word length
                if piece in dic:
                    index = size - 1
                    result.append(piece)
                    break
                # current processing position + 1 so that the next word can be processed
            index = index + 1
        return result


if __name__ == '__main__':
    # define the text to be segmented
    try:
        with open('../NBA.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        # remove special characters
        text = text.replace('\u3000', '').replace('\n', '')
        # instantiate the MM class
        tokenizer = MM()
        # call the method in the MM class
        print(tokenizer.cut(text))
    except FileNotFoundError as e:
        print(f'File not found: {e}')
    except PermissionError as e:
        print(f'Permission denied: {e}')
    except Exception as e:
        print(f'Unknown error: {e}')

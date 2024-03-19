from exp1.archive.rule_based.dictionary import dictionary


class BMM(object):
    def __init__(self):
        # define maximum number of window size
        # set to 6 as 凯文-杜兰特
        self.window_size = 6
        # define segmentation function

    def cut(self, text):
        # store segmentation result
        result = []
        # current processing position, different from forward match
        # last index of the text
        index = len(text)
        # define match dictionary
        dic = dictionary
        # loop match, different from forward match since processing from the end
        while index > 0:
            # forward match process
            for size in range(index - self.window_size, index):
                # get processing text
                piece = text[size:index]
                # if match word in dict, navigate by the matched word length
                if piece in dic:
                    index = size + 1
                    result.append(piece)
                    break
                # current processing position + 1 so that the next word can be processed
            index = index - 1
        # reverse the result list to make it in the correct order
        result.reverse()
        return result


if __name__ == '__main__':
    # define the text to be segmented
    try:
        with open('../NBA.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        text = text.replace('\u3000', '').replace('\n', '')
        tokenizer = BMM()
        print(tokenizer.cut(text))
    except FileNotFoundError as e:
        print(f'File not found: {e}')
    except PermissionError as e:
        print(f'Permission denied: {e}')
    except Exception as e:
        print(f'Unknown error: {e}')

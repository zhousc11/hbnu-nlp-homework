from archive.dictionary import dictionary


class BidMM(object):
    def __init__(self):
        # maximum word length
        self.window_size = 6  # 6 for 杜兰特

    # def segmentation function
    def cut(self, text):
        # store the forward segmentation result
        left_result = []
        # store the backward segmentation result
        right_result = []
        # left segmentation position
        left_index = 0
        # right segmentation position
        right_index = len(text)
        # text len
        text_length = len(text)
        # sum the word count of forward segmentation up
        left_count = 0
        # sum the word count of backward segmentation up
        right_count = 0
        # define the dictionary
        dic = dictionary

        # execute forward segmentation
        while text_length > left_index:
            # process text from right to left
            for size in range(self.window_size + left_index, left_index, -1):
                # get the processed word
                piece = text[left_index:size]
                # if the word is in the dictionary, navigate to the next word by plussing the word length
                if piece in dic:
                    left_index = size - 1
                    break
            # add the current position by 1 to process remaining text
            left_index = left_index + 1
            # judge if current word is single character
            if len(piece) == 1:
                left_count = left_count + 1
            left_result.append(piece)

        # execute backward segmentation
        while right_index > 0:
            # process text from right to left
            for size in range(right_index - self.window_size, right_index):
                # get the processing word
                piece = text[size:right_index]
                # if the word is in the dictionary, navigate to the next word by minusing the word length
                if piece in dic:
                    right_index = size + 1
                    break
            # minus the current position by 1 to process remaining text
            right_index = right_index - 1
            # judge if current word is single character
            if len(piece) == 1:
                right_count = right_count + 1
            right_result.append(piece)
        right_result.reverse()

        # return the one with less single character if the word count is not the same
        if len(left_result) < len(right_result):
            return left_result
        elif len(left_result) > len(right_result):
            return right_result
        # if word count is the same, return any one
        else:
            if left_count <= right_count:
                return left_result
            else:
                return right_result


if __name__ == '__main__':
    # open file
    try:
        with open('NBA.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        # remove the line break, etc.
        text = text.replace('\u3000', '').replace('\n', '')
        # initialize the tokenizer
        tokenizer = BidMM()
        print(tokenizer.cut(text))
    except FileNotFoundError as e:
        print(f'File not found error: {e}')
    except PermissionError as e:
        print(f'Permission error: {e}')
    except Exception as e:
        print(f'Unknown Error: {e}')

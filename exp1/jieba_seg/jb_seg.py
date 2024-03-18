import jieba


# run the main function
if __name__ == '__main__':
    # load the file
    try:
        with open('../NBA.txt', 'r', encoding='utf-8') as f:
            content = f.read()
        # use jieba to segment the content
        seg_list = jieba.cut(content, cut_all=False)
        # print the result
        print('/'.join(seg_list))
    except FileNotFoundError as e:
        print('File not found:', e)
    except PermissionError as e:
        print('Permission denied:', e)
    except Exception as e:
        print('Error occurred:', e)

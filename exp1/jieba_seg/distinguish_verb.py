import jieba.posseg

# run the main function
if __name__ == '__main__':
    # load the file
    try:
        with open('../NBA.txt', 'r', encoding='utf-8') as f:
            content = f.read()
        # 使用jieba辨别词性
        words = jieba.posseg.cut(content)
        # print the result
        for word, flag in words:
            print('%s %s' % (word, flag), end='/')
    except FileNotFoundError as e:
        print('File not found:', e)
    except PermissionError as e:
        print('Permission denied:', e)
    except Exception as e:
        print('Error occurred:', e)

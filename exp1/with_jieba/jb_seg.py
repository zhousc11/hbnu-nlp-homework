import re
import jieba.posseg as psg

re_han_internal = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&\._]+)")

# words to be processed
try:
    with open('../NBA.txt', 'r', encoding='utf-8') as f:
        content = f.read()
except FileNotFoundError:
    print(f'The file "NBA.txt" does not exist in the current directory. \nDownload the file from Chaoxing course '
          f'and'
          f'try again.')
except PermissionError:
    print(f'Permission denied. Please check if you have the permission to read the file.')
except Exception as e:
    print(f'Unexpected error occurred: {e}')

# segmentation and part of speech tagging
seg_list = psg.cut(content)
print(' '.join(['{0}/{1}'.format(w,t) for w, t in seg_list]))

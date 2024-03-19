import jieba.posseg as pseg
from exp1.read_file import read_file


def segment_and_tag(text):
    words = pseg.cut(text)
    for word, flag in words:
        print('%s %s' % (word, flag), end='/')


def main():
    file_path = '../../NBA.txt'
    content = read_file(file_path)
    content = content.replace('\n', '').replace(' ', '').replace('\t', '').replace('\r', '')
    if content is not None:
        segment_and_tag(content)


if __name__ == '__main__':
    main()

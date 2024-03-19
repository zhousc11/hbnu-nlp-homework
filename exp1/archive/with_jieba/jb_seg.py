import jieba
from exp1.read_file import read_file


def segment_text(text):
    # use jieba to segment
    seg_list = jieba.cut(text, cut_all=False)
    return '/'.join(seg_list)


def main():
    # load the file
    file_path = '../../NBA.txt'
    content = read_file(file_path)
    content = content.replace('\n', '').replace(' ', '').replace('\t', '').replace('\r', '')
    if content is not None:
        # segment the text
        segmented_text = segment_text(content)
        print(segmented_text)


if __name__ == '__main__':
    main()

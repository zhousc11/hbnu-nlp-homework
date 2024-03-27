def f1(path):
    # open rts file
    with open(path, encoding='utf-8', errors='ignore') as f:
        all_tag = 0
        loc_tag = 0
        pred_loc_tag = 0
        correct_tag = 0
        correct_loc_tag = 0

        states = ['B', 'M', 'E', 'S']
        for line in f:

            line = line.strip()
            if line == '':
                continue
            _, r, p = line.split()
            all_tag += 1
            if r == p:
                correct_tag += 1
                if r in states:
                    correct_loc_tag += 1
            if r in states:
                loc_tag += 1
            if p in states:
                pred_loc_tag += 1
        loc_P = 1.0 * correct_loc_tag / pred_loc_tag
        loc_R = 1.0 * correct_loc_tag / loc_tag
        print('loc_P:{0}, loc_R:{1}, loc_F1:{2}'.format(loc_P, loc_R, (2 * loc_P * loc_R) / (loc_P + loc_R)))


print('记得一定要将文件转换为UTF-8才可以正常使用老师的代码！！！\n' * 3, end='')
f1("D:\\Users\\Documents\\NLP_Homework\\Practice2\\4.1\\data\\test.rst")

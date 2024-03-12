def backward_max_match(sentence, dictionary, max_word_length):
    index = len(sentence)
    result = []
    while index > 0:
        word = None
        for size in range(max_word_length, 0, -1):
            if index - size < 0:  # 确保不会向左越界
                continue
            piece = sentence[index - size:index]  # 从右向左获取子字符串
            if piece in dictionary:
                word = piece
                result.insert(0, word)  # 将匹配到的词插入到结果列表的开头
                index -= size
                break
        if word is None:  # 如果没有在字典中找到匹配的词
            word = sentence[index - 1]
            result.insert(0, word)  # 将单个字符插入到结果列表的开头
            index -= 1
    return result


dictionary = {'北京', 'NB-A', '全明星', '训练', '有趣', '一个', '球员', '投篮', '全队', '加入', '和', '凯文-杜兰特',
              '互动', '媒体', '照片', '赛前', '热身', '一个人', '其他', '看来', '开始', '拒绝', '外界', '刻意', '避开',
              '孤独', '詹姆斯-哈登', '之后', '传球', '了解', '送出', '上篮', '助攻', '记者', '推特', '表示', '忽视',
              '抢篮板', '天呐', '看见', '更衣柜', '刚好', '对面', '发布', '小视频', '清楚', '相隔', '名球员', '身旁',
              '考辛斯', '位于', '对角', '位置', '左侧', '汤普森', '库里', '格林', '勇士', '四人组', '挨在', '一块',
              '尽管', '昔日', '效力', '关系', '非常', '恶劣', '客场', '击败', '赛后', '进入', '牛排馆', '吃饭', '全程',
              '毫无', '交流', '全明星周末', '期间', '迎面', '碰到', '装作', '没有', '看见', '宛若', '对方', '空气',
              '破裂', '感情', '修复', '依靠', '时间', '三分球', '大赛', '现场', '主持人', '介绍', '到场', '嘉宾',
              '口误', '端坐', '场边', '俄克拉荷马', '不爽', '本届', '看点', '重逢', '火花', '掌握', '主帅',
              '史蒂夫-科尔', '手里', '完成', 'MVP', '三连庄', '张卫平', '指导', '预测', '结果', '归属', '认为', '当选',
              '机会', '连续', '两年', '成为', '（', '）', '轰下', '分夺得', '砍下', '蝉联', '成为了', '继', '之后',
              '第二位', '球员', '如果', '那么', '将', '第一位', '因此', '全力', '冲击', '为什么', '这么', '说呢',
              '首先', '前', '凯尔特人队', '快船队', '主教练', '道格-里弗斯', '向', '提建议', '死敌', '疯狂', '地',
              '发挥', '免得', '将来', '见面', '时候', '玩命', '现在', '西部', '雷霆', '究其', '原因', '还是', '因为',
              '冗长', '肥皂剧', '一定', '痛快', '其次', '四巨头', '控制', '出场', '时间', '埋怨', '教练', '对手',
              '因为', '怠慢', '记下', '仇来', '不出', '意外', '听从', '建议', '玩命', '第三点', '性格', '娱乐性质',
              '比赛', '没有', '特别', '认真', '上场', '就', '干', '这就是', '大伙', '心照不宣', '目前', '保持', '场均',
              '三双', '先发', '心里', '一直', '憋着', '一口气', '证明', '自己', '如此', '战个', '再加上', '应该',
              '稳稳', '地', '收下', '但是', '不能', '说', '一定', '能', '拿', '只是', '个', '预测', '具体', '看',
              '临场', '发挥', '（栗旬）', 'NBA'}

# Read the text from file
if __name__ == '__main__':
    try:
        with open('../NBA.txt', 'r', encoding='utf-8') as f:
            text = f.read()

        max_word_length = max(len(word) for word in dictionary)

        # 移除文本中的空格和特殊字符
        text = text.replace('\u3000', '').replace('\n', '')

        # 分词
        result = backward_max_match(text, dictionary, max_word_length)

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

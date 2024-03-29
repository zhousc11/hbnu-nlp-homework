from dataprocess import seg_to_list, word_filter, tfidf_extract, textrank_extract, topic_extract

text = '6月19日,《2012年度“中国爱心城市”公益活动新闻发布会》在京举行。' + \
       '中华社会救助基金会理事长许嘉璐到会讲话。基金会高级顾问朱发忠,全国老龄' + \
       '办副主任朱勇,民政部社会救助司助理巡视员周萍,中华社会救助基金会副理事长耿志远,' + \
       '重庆市民政局巡视员谭明政。晋江市人大常委会主任陈健倩,以及10余个省、市、自治区民政局' + \
       '领导及四十多家媒体参加了发布会。中华社会救助基金会秘书长时正新介绍本年度“中国爱心城' + \
       '市”公益活动将以“爱心城市宣传、孤老关爱救助项目及第二届中国爱心城市大会”为主要内容,重庆市' + \
       '、呼和浩特市、长沙市、太原市、蚌埠市、南昌市、汕头市、沧州市、晋江市及遵化市将会积极参加' + \
       '这一公益活动。中国雅虎副总编张银生和凤凰网城市频道总监赵耀分别以各自媒体优势介绍了活动' + \
       '的宣传方案。会上,中华社会救助基金会与“第二届中国爱心城市大会”承办方晋江市签约,许嘉璐理' + \
       '事长接受晋江市参与“百万孤老关爱行动”向国家重点扶贫地区捐赠的价值400万元的款物。晋江市人大' + \
       '常委会主任陈健倩介绍了大会的筹备情况。'

pos = True
seg_list = seg_to_list(text, pos)
filter_list = word_filter(seg_list, pos)

print('TF-IDF模型结果：')
tfidf_extract(filter_list)

print('TextRank模型结果：')
textrank_extract(text)

print('LSI模型结果：')
topic_extract(filter_list, 'LSI', pos)

print('LDA模型结果：')
topic_extract(filter_list, 'LDA', pos)

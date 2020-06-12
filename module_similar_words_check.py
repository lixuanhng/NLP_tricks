import jieba

vocab = [['购买','买取','买来'], ['诸葛亮', '姜维'], ['A', 'B', 'C'], ['了', '的']]
text1 = '我从对方那里购买了一个苹果'
text2 = '甲方买取了部分设备'

words1 = jieba.lcut(text1)
words2 = jieba.lcut(text2)

for i in range(len(vocab)):
    common_word1 = list(set(words1) & set(vocab[i]))
    common_word2 = list(set(words2) & set(vocab[i]))
    if common_word1 and common_word2:  # 没有近义词出现时，交集列别为空
        if common_word1[0] in vocab[i] and common_word2[0] in vocab[i]:
            print('第' + str(i) + '组中出现近义词')
            print(text1 + '：' + common_word1[0] + '，该词index为' + str(text1.index(common_word1[0])))
            print(text2 + '：' + common_word2[0] + '，该词index为' + str(text2.index(common_word2[0])))
            print('-------------------------------------------------------')

"""
本段代码尝试使用 hanlp 对句子进行语义分词，并考虑与语法分析结合起来
"""

import hanlp
text = '甲方根据资金落实情况，保证在设备进厂后40个工作日内支付该项进度款'
tokenizer = hanlp.load('PKU_NAME_MERGED_SIX_MONTHS_CONVSEG')  # 载入模型
tokens = tokenizer(text)  # 分词
print(tokens)

tagger = hanlp.load(hanlp.pretrained.pos.CTB5_POS_RNN_FASTTEXT_ZH)  # 载入词性标注模型
tags = tagger(tokens)  # 词性标注
print(tags)

# 将词与词性合并（以元祖形式按位合并为列表）
token_tag_group = []
for i in range(len(tokens)):
    token_tag_group.append((tokens[i], tags[i]))
print(token_tag_group)

# 语义分析
semantic_parser = hanlp.load(hanlp.pretrained.sdp.SEMEVAL16_NEWS_BIAFFINE_ZH)
parsers = semantic_parser(token_tag_group)
print(parsers)
print(dir(parsers))

for item in parsers:
    print(dir(item))
    print(item.REL)
    break
    # print(str(item.id) + item.form + item.cpos + item.head + str(item.rel))

"""
测试模块代码
"""
from model_similar_words_check_and_substitute import SimilarWordsCheck
from model_punctuation_transform_EntoZh import PunctuationTransform


# 测试代码1
texts1 = ['在与卖方签定合同后的十日内', '支付总货款的', '在履行合同期间']  # 已知风险例句列表
texts2 = ['出卖人于合同签订后的', '按时发出货物', '在合同有效期内', '支付货物全款的']  # 带判断例句列表
# 载入同义词词表
similar_path = 'D:/CepuTech/cepu_data/similar_words.txt'
with open(similar_path, 'r', encoding='utf-8') as file1:
    vocab_group_list = file1.readlines()
# 对同义词词表进行去除特殊符号，并按组生成列表
vocab_group_list = [group_str.strip('\n').split(' ') for group_str in vocab_group_list]
print(vocab_group_list)
# 所有成分对比，找出两两之间的包含的近义词个数
model = SimilarWordsCheck(texts1, texts2, vocab_group_list)
matrix = model.sentences_similar_matrix()
print(matrix)


# 测试代码2（切记，如遇分句，必去前后空格）
test = '合同规定:甲乙双方在签定合同后,1. 甲方需要向乙方供货;二. 乙方向甲方支付1,700.00元.另一方面'
model = PunctuationTransform(test)
precess_result = model.punctuation_transform_en_to_zh()
print(precess_result)
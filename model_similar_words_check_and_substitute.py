import re
import pandas as pd
import numpy as np


class SimilarWordsCheck:
    """
    该类用来处理近义词查找及替换。
    本段代码通过近义词词库（txt文件）用来比较两个句子的相似程度
    需要对比的两个句子成分中，如果在其中一个句子中出现了同义词词库中的某个词，则检查另一个句子中
    是否出现了该词所在组中的其他词，如果存在，则表示两个句子成分中包含相同意义的词
    """
    def __init__(self, texts1, texts2, vocab_group_list):
        self.referred_group = texts1
        self.processed_group = texts2
        self.vocab_group = vocab_group_list

    def sentences_similar_matrix(self):
        """
        本函数目的：通过分别取自已知风险成分列表和待检查成分列表的句子进行两两比较，得到两个句子的所包含的同义词个数
        并制成矩阵形式，行为已知风险成分，列为待检查成分
        param referred_group: 已知风险成分列表，list
        param processed_group: 待检查成分列表，list
        param vocab_group: 近义词词表，list
        return matrix_df: DataFrame格式的每两个成分之间的近义词个数，row为已知风险成分，column为待检查成分
        """
        # 找到两两成分之间包含的近义词个数
        matrix_counts = []
        for processed in self.processed_group:
            row_counts = []
            for referred in self.referred_group:
                # 检查已知风险句和待判断句是否有近义词
                similar_count, processed_sub = SimilarWordsCheck.pair_sentences_comparing(self, referred, processed)
                row_counts.append(similar_count)
            matrix_counts.append(row_counts)
        # 将结果转化为矩阵形式
        matrix_df = pd.DataFrame(np.array(matrix_counts))
        return matrix_df

    def pair_sentences_comparing(self, referred, processed):
        """
        本函数目的：当 A 组中的词 A1 已经出现在风险成分中时，检查 A 组中是否有其他词，如 A2，出现在待检查成分中，若有则可替换
        param referred: 已知风险成分，str
        param processed: 待检查成分，str
        param vocab_group: 近义词词表，list
        return 包含近义词个数 similar_count（int），近义词替换后待检查成分 processed（str）
        """
        flag_word = ''  # 标记在text1中找到的词
        similar_count = 0  # 标记找到的近义词的组数
        for No in range(len(self.vocab_group)):
            for sin_word in self.vocab_group[No]:
                if sin_word in referred:  # 检查每个组中的每个词是否出现在测试句子中
                    flag_word = sin_word  # 标记找到该词
                    for can_word in self.vocab_group[No]:
                        if can_word in processed:
                            processed = re.sub(can_word, flag_word, processed)  # 近义词替换
                            similar_count += 1
                            break
        # print('找到的近义词的组数为' + str(similar_count) + '个')
        # print('processed 句子变为：' + '\n' + processed)
        return similar_count, processed



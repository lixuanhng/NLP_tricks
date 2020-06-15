import re
import pandas as pd
import numpy as np


class SimilarWordsCheck:
    """
    This class is used to targetting and replacing the similar words in a pair of sentences,
    So that we can compare the the similarity in two sentences, according to the Similar_Word_Vocab (SWV).
    The method is if we can find word 'A1' showing up in SWV in one sentence, 
    then we go check if the other one includes the other similar words (A2, A3, ..., An) from the same group (A1, A2, A3, ..., An) as well. 
    If the answer is Yes, both of those sentences have the similar words which can be replaced without any wrong semantics. 
    """
    def __init__(self, texts1, texts2, vocab_group_list):
        self.referred_group = texts1
        self.processed_group = texts2
        self.vocab_group = vocab_group_list

    def sentences_similar_matrix(self):
        """
        This function is used to compare the two group (called referred_group and process_grooup. each one is a list holding several sentence)
        and check how many simiar words each pair from 2 groups both has. The result will be a matrix.
        Column number is the number of element in referred_group; Row number is the number of element in processed_group
        param referred_group: list
        param processed_group: list
        param vocab_group: Similar_Words_Vacab，list
        return matrix_df: the DataFrame of matrix result about how many similar the each pair has
        """
        matrix_counts = []
        for processed in self.processed_group:
            row_counts = []
            for referred in self.referred_group:
                # check if the pair has the similar words
                similar_count, processed_sub = SimilarWordsCheck.pair_sentences_comparing(self, referred, processed)
                row_counts.append(similar_count)
            matrix_counts.append(row_counts)
        # transform results to the matrix in DataFrame type
        matrix_df = pd.DataFrame(np.array(matrix_counts))
        return matrix_df

    def pair_sentences_comparing(self, referred, processed):
        """
        This function focus on if we can find word 'A1' showing up in SWV in one sentence, 
        then we go check if the other one includes the other similar words (A2, A3, ..., An) from the same group (A1, A2, A3, ..., An) as well. 
        param referred: sentence，str
        param processed: the other sentence，str
        param vocab_group: Similar_Words_Vacab，list
        return: similar_count（simiar count, int），processed（replaced_processd_sentence, str）
        """
        flag_word = ''
        similar_count = 0
        for No in range(len(self.vocab_group)):
            for sin_word in self.vocab_group[No]:
                if sin_word in referred:
                    flag_word = sin_word
                    for can_word in self.vocab_group[No]:
                        if can_word in processed:
                            processed = re.sub(can_word, flag_word, processed)  # similar words replaced
                            similar_count += 1
                            break
        return similar_count, processed


# demo test
if __name__ == '__main__':
    texts1 = ['在与卖方签定合同后的十日内', '支付总货款的', '在履行合同期间']  # referred_group
    texts2 = ['出卖人于合同签订后的', '按时发出货物', '在合同有效期内', '支付货物全款的']  # processed_group
    
    # load similar words vocabulary
    similar_path = './lixuanhng/NLP_tricks/test_data/similar_words.txt'
    with open(similar_path, 'r', encoding='utf-8') as file1:
        vocab_group_list = file1.readlines()
    vocab_group_list = [group_str.strip('\n').split(' ') for group_str in vocab_group_list]

    # compare the pair components from different groups separately and find the common similar words in way of matrix
    model = SimilarWordsCheck(texts1, texts2, vocab_group_list)
    matrix = model.sentences_similar_matrix()
    print(matrix)

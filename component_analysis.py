

class ComponentComparison:
    """
    本段代码用来处理第三层关系抽取，即比较的两个句子相同位置成分的相似程度
    截至目前，本部分的灵感来源于 risks_check_configuration.py，
    对两个成分分词，分别取出两个成分中的实体，相关动词，和介词，再两两比较，由对比结果生成
    """
    def __init__(self, risk_tokens, check_tokens, risk_comp, check_comp, preps, entities, behaviors, path):
        self.tokens_r = risk_tokens   # 已知风险句的tokens
        self.tokens_c = check_tokens  # 待检查句的tokens
        self.comp_r = risk_comp   # 需要对比的已知风险的成分
        self.comp_c = check_comp  # 需要对比的待检查的成分
        self.preps_vocab = preps  # 介词词表
        self.entities_vocab = entities    # 实体词表
        self.behaviors_vocab = behaviors  # 相关行为词表
        self.similar_path = path  # 近义词词表路径

    def load_similar_vocab(self):
        with open(self.similar_path, 'r', encoding='utf-8') as file1:
            similar_group_list = file1.readlines()
        # 对同义词词表进行去除特殊符号，并按组生成列表
        similar_group_list = [group_str.strip('\n').split(' ') for group_str in similar_group_list]
        return similar_group_list

    def key_words_extraction_from_component(self):
        # 对成分字符串进行分词
        risk_comp_words = [token for token in self.tokens_r if token in self.comp_r]
        check_comp_words = [token for token in self.tokens_c if token in self.comp_c]
        # step1. 找出每个成分中的实体名词
        risk_entities = [item for item in risk_comp_words if item in self.entities_vocab]
        check_entities = [item for item in check_comp_words if item in self.entities_vocab]
        # step2. 找出每个成分中的相关动词
        risk_behaviors = [item for item in risk_comp_words if item in self.behaviors_vocab]
        check_behaviors = [item for item in check_comp_words if item in self.behaviors_vocab]
        # step3. 找出每个成分中的介词表示
        risk_preps = []
        check_preps = []
        for sin_prep in preps_list:
            if '......' in sin_prep:
                preps_words = sin_prep.split('......')  # 将介词表示中的每个词都取出来
                if set(preps_words) < set(risk_comp_words):  # 若组成介词的词组被包含于成分分词列表，则该介词存在于该成分
                    risk_preps.append(sin_prep)
                if set(preps_words) < set(check_comp_words):
                    check_preps.append(sin_prep)

        risk_key_words_dict = {'entity': risk_entities, 'behaviors': risk_behaviors, 'preps': risk_preps}
        check_key_words_dict = {'entity': check_entities, 'behaviors': check_behaviors, 'preps': check_preps}
        return risk_key_words_dict, check_key_words_dict

    @ staticmethod
    def single_item_in_key_set(key_a, key_b, similar_group_list):
        """
        当在每个成分中，取出的每种关键词是唯一的时候（即一个行为动词，一个实体名词，一个介词）
        本段代码用来比较每种唯一的关键词是否属于近义词，并且适用于一切关键词的比较
        """
        entity_group = [key_a, key_b]  # 将两个
        # 检查当前 entity_group 是否存在于近义词词表中
        entity_group_in = [sin_group for sin_group in similar_group_list if set(entity_group) < set(sin_group)]
        if entity_group_in:
            flag_key = 1  # 实体相近
        else:
            flag_key = -1  # 实体不相近
        return flag_key

    def component_comparing(self):
        similar_group_list = ComponentComparison.load_similar_vocab(self)  # 读入近义词词表
        risk_key_words_dict, check_key_words_dict = ComponentComparison.key_words_extraction_from_component(self)

        # step1. 检查实体是否匹配（原则：value值为空，则返回0；不为空但不相近，返回-1；不为空相近，返回1）
        flag_entity = 0
        if risk_key_words_dict['entities'] and check_key_words_dict['entities']:
            # 当在每个成分中，取出的每种关键词唯一时
            if len(risk_key_words_dict['entities']) == len(check_key_words_dict['entities']) == 1:
                # 将两个待匹配的实体组合成列表
                entity_a, entity_b = risk_key_words_dict['entities'][0], check_key_words_dict['entities'][0]
                flag_entity = ComponentComparison.single_item_in_key_set(entity_a, entity_b, similar_group_list)
            elif len(risk_key_words_dict['entities']) > 1 or len(check_key_words_dict['entities']) > 1:
                

        # step2. 检查相关动词是否匹配（原则如上）
        flag_behaviors = 0
        if risk_key_words_dict['behaviors'] and check_key_words_dict['behaviors']:
            if len(risk_key_words_dict['behaviors']) == len(check_key_words_dict['behaviors']) == 1:
                behaviors_a, behaviors_b = risk_key_words_dict['behaviors'][0], check_key_words_dict['behaviors'][0]
                flag_behaviors = ComponentComparison.single_item_in_key_set(behaviors_a, behaviors_b, similar_group_list)

        # step3. 检查相关介词是否匹配（原则如上）
        flag_preps = 0
        if risk_key_words_dict['preps'] and check_key_words_dict['preps']:
            if len(risk_key_words_dict['preps']) == len(risk_key_words_dict['preps']) == 1:
                preps_a, preps_b = risk_key_words_dict['preps'][0], check_key_words_dict['preps'][0]
                flag_preps = ComponentComparison.single_item_in_key_set(preps_a, preps_b, similar_group_list)

        # 最终的成分的相似向量表示为三个部分的
        check_comp_vector = [flag_entity, flag_behaviors, flag_preps]
        return check_comp_vector


if __name__ == '__main__':
    # 分词结果
    tokens_g = ['甲方', '根据', '资金', '落实', '情况', '，', '保证', '在', '设备', '进厂', '后', '40', '个', '工作日', '内', '支付', '该项', '进度款']
    tokens_3 = ['甲方', '按照', '储备金', '情况', '，', '确保', '在', '设备', '进厂', '后', '40', '天内', '支付', '货款']
    # 原始成分文本
    comp_g = '根据资金落实情况'
    comp_3 = '按照储备金情况'
    preps_list = ['根据......情况', '按照......情况', '自......起']
    entities_list = ['甲方', '进度款', '总价款', '资金', '工作日', '货物', '全款', '储备金']
    behaviors_list = ['支付', '签定', '验收', '付清', '到位']
    similar_path = 'D:/cepu_data/similar_words.txt'
    model_1 = ComponentComparison(tokens_g, tokens_3, comp_g, comp_3, preps_list, entities_list, behaviors_list, similar_path)
    comp_3_vector = model_1.component_comparing()
    print(comp_3_vector)


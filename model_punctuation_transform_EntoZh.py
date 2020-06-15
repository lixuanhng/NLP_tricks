import re


class PunctuationTransform:
    """
    本段函数用来将英文标点转化为中文标点
    """
    def __init__(self, text):
        self.text_str = text

    def punctuation_transform_en_to_zh(self):
        """
        标点符号转化
        """
        refresh_text = self.text_str
        # 处理英文句号
        targets_1 = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
                     '一', '二', '三', '四', '五', '六', '七', '八', '九', '十',
                     ')', '）', '>', ']']
        p1 = re.compile('.(?=\\.)')  # 找到英文句号并取其前一位
        match_1 = re.findall(p1, refresh_text)
        if match_1:
            for item in match_1:
                if item not in targets_1:
                    refresh_text = re.sub(item + '.', item + '。', refresh_text)  # 非特殊用途需要替换
        # 处理英文逗号
        targets_2 = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
        p2 = re.compile('.(?=,)')  # 找到英文逗号并取其前一位
        match_2 = re.findall(p2, refresh_text)
        if match_2:
            for item in match_2:
                if item not in targets_2:
                    refresh_text = re.sub(item + ',', item + '，', refresh_text)  # 非特殊用途需要替换
        # 处理英文问号
        refresh_text = re.sub('\\?', '？', refresh_text)
        # 处理英文小括号
        refresh_text = re.sub('\\(', '（', refresh_text)
        refresh_text = re.sub('\\)', '）', refresh_text)
        # 处理英文冒号
        refresh_text = re.sub(':', '：', refresh_text)
        # 处理英文分号
        refresh_text = re.sub(';', '；', refresh_text)
        return refresh_text
import re


class PunctuationTransform:
    """
    The class focuses on the punctuation transformation from En to Zh in a Chinese text.
    The input is the text you would like to change and update. Data type is 'str'.
    The output is the changed and updated text. Data type is 'str'.
    """
    def __init__(self, text):
        self.text_str = text

    def punctuation_transform_en_to_zh(self):
        """
        punctuation transformation
        """
        refresh_text = self.text_str
        
        # replace English full stop with Chinese full stop.
        targets_1 = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
                     '一', '二', '三', '四', '五', '六', '七', '八', '九', '十',
                     ')', '）', '>', ']']
        p1 = re.compile('.(?=\\.)')
        match_1 = re.findall(p1, refresh_text)
        if match_1:
            for item in match_1:
                if item not in targets_1:
                    refresh_text = re.sub(item + '.', item + '。', refresh_text)
        
        # replace English comma with Chinese comma.
        targets_2 = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
        p2 = re.compile('.(?=,)')
        match_2 = re.findall(p2, refresh_text)
        if match_2:
            for item in match_2:
                if item not in targets_2:
                    refresh_text = re.sub(item + ',', item + '，', refresh_text)  # 非特殊用途需要替换
        
        # replace English question mark with Chinese question mark.
        refresh_text = re.sub('\\?', '？', refresh_text)
        
        # replace English parentheses with Chinese parentheses.
        refresh_text = re.sub('\\(', '（', refresh_text)
        refresh_text = re.sub('\\)', '）', refresh_text)
        
        # replace English colon with Chinese colon.
        refresh_text = re.sub(':', '：', refresh_text)
        
        # replace English semicolon with Chinese semicolon.
        refresh_text = re.sub(';', '；', refresh_text)
        return refresh_text

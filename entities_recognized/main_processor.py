"""
Purpose:    本段代码的目的，是通过人工做好的关键实体词，结合 BiLSTM 和 CRF训练一个实体识别的模型
Note:   首先，这里暂时认为一个实体就属于一个要素，实体识别出来的结果就是一个要素被打上了标签
Need:   实体词库，根据实体词库的标注来作为训练数据
        模式匹配下，使用 ahocorasick
        使用 审理查明的数据集，在桌面
resource:   https://github.com/shiyybua/NER
"""


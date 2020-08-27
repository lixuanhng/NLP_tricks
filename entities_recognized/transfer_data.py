"""
Purposes:
转换脚本函数
本段代码的目的是根据已经标注好的数据，将对应的原始文本数据中每一个字都打上BOI标签
------------------------------------------------------------------------------------
Date：
2020/8/26
"""

import os
from collections import Counter


class TransferData:
    def __init__(self):
        cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.label_dict = {'检查和检验': 'CHECK',
                           '症状和体征': 'SIGNS',
                           '疾病和诊断': 'DISEASE',
                           '治疗': 'TREATMENT',
                           '身体部位': 'BODY'}

        self.cate_dict = {'O': 0,
                          'TREATMENT-I': 1,
                          'TREATMENT-B': 2,
                          'BODY-B': 3,
                          'BODY-I': 4,
                          'SIGNS-I': 5,
                          'SIGNS-B': 6,
                          'CHECK-B': 7,
                          'CHECK-I': 8,
                          'DISEASE-I': 9,
                          'DISEASE-B': 10}
        # 原始数据：
        # 1) <name> 文件为实体词在原文中的词序号及对应的实体类型
        # 2) <name.txtoriginal> 文件为<name>对应的原文
        self.origin_path = os.path.join(cur, 'data_origin')
        # print(self.origin_path)
        # 需要写入的文件，train.txt 中的数据格式是每行一个字及这个字对应的BOI标签
        self.train_filepath = os.path.join(cur, 'data/train.txt')
        # print(self.train_filepath)
        return

    def transfer(self):
        """

        :return:
        """
        # 将处理完的数据写入train_file.txt
        with open(self.train_filepath, 'w+', encoding='utf-8') as train_file:
            # os.walk() 目录树中游走，输出在目录中的文件名
            # root 所指的是当前正在遍历的这个文件夹的本身的地址
            # dirs 是一个 list ，内容是该文件夹中所有的目录的名字(不包括子目录)
            # files 同样是 list , 内容是该文件夹中所有的文件(不包括子目录)
            for root, dirs, files in os.walk(self.origin_path):
                for file in files:
                    # 当前读取文件的路径
                    filepath = os.path.join(root, file)
                    if 'original' not in filepath:
                        # 如果读到的文件不是文本原文，则跳过
                        continue
                    # 将读到原文的文件名中的'.txtoriginal'去掉
                    label_filepath = filepath.replace('.txtoriginal', '')
                    # 获取文本数据中的文本
                    content = open(filepath).read().strip()
                    res_dict = {}  # 针对每个文本构建
                    # 读取原文对应的标注文本
                    for line in open(label_filepath):
                        # 每一行为数据的标注，举例：'头部\t42\t43\t身体部位'
                        res = line.strip().split('\t')
                        start = int(res[1])  # 实体词的头部位置B
                        end = int(res[2])  # 实体词的内部位置I
                        label = res[3]  # 实体词所属类型
                        label_id = self.label_dict.get(label)
                        # 给文本中每个词打上标签
                        for i in range(start, end+1):
                            if i == start:
                                label_cate = label_id + '-B'
                            else:
                                label_cate = label_id + '-I'
                            res_dict[i] = label_cate
                    # 将非实体的字也打上标签
                    for indx, char in enumerate(content):
                        char_label = res_dict.get(indx, 'O')
                        print(char, char_label)
                        train_file.write(char + '\t' + char_label + '\n')
        return


if __name__ == '__main__':
    handler = TransferData()
    train_datas = handler.transfer()
"""
本段代码用来构建LSTM+CRF层完成实体识别的训练
"""

import numpy as np
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense, TimeDistributed, Dropout
from keras_contrib.layers.crf import CRF
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


class LSTMNER:
    def __init__(self):
        cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])  # 当前文件路径
        self.train_path = os.path.join(cur, 'data/train.txt')  # 标注好的数据
        self.vocab_path = os.path.join(cur, 'model/vocab.txt')  # 需要写入的词表的路径
        self.embedding_file = os.path.join(cur, 'model/token_vec_300.bin')  # 预训练词向量模型
        self.model_path = os.path.join(cur, 'model/tokenvec_bilstm2_crf_model_20.h5')  # 训练后要保存的模型参数
        self.datas, self.word_dict = self.build_data()
        self.class_dict ={'O': 0,
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
        self.EMBEDDING_DIM = 300  # 词向量维度
        self.EPOCHS = 5
        self.BATCH_SIZE = 128
        self.NUM_CLASSES = len(self.class_dict)
        self.VOCAB_SIZE = len(self.word_dict)
        self.TIME_STAMPS = 150
        self.embedding_matrix = self.build_embedding_matrix()

    def build_data(self):
        """
        构造数据集
        :return:    1. datas: 字和字标签的集合，每一个元素为以短句为单位的字和字标签的列表
                    2. word_dict: {字：id} 的字典
        """
        datas = []
        sample_x = []  # 存放【字】
        sample_y = []  # 存放【字的标注】
        vocabs = {'UNK'}
        for line in open(self.train_path):
            line = line.rstrip().split('\t')
            if not line:
                continue
            char = line[0]  # line = [char, category] 字和标注
            if not char:
                continue
            cate = line[-1]
            sample_x.append(char)
            sample_y.append(cate)
            vocabs.add(char)  # 将当前字添加到字表中，这里的vocab难道不应该去重吗？
            ### 以每个短句截止，将短句中出现的字及标注分别添加到datas中
            if char in ['。', '?', '!', '！', '？']:
                datas.append([sample_x, sample_y])
                sample_x = []
                sample_y = []
        # 构建 {字：id} 的字典，
        word_dict = {wd: index for index, wd in enumerate(list(vocabs))}
        # 将每个字写入文件中
        self.write_file(list(vocabs), self.vocab_path)
        return datas, word_dict

    def modify_data(self):
        """
        将数据转换成 keras所需的格式
        :return:
        """
        # 根据每个短句将其中的【字】和【字标签】对应的ID取出来组成训练集
        x_train = [[self.word_dict[char] for char in data[0]] for data in self.datas]
        y_train = [[self.class_dict[label] for label in data[1]] for data in self.datas]
        # 对文本长度进行填充，长度固定为 self.Time_STAMPS，超过截断，不足补齐
        x_train = pad_sequences(x_train, self.TIME_STAMPS)  # pad_sequences 返回二维向量
        y = pad_sequences(y_train, self.TIME_STAMPS)
        y_train = np.expand_dims(y, 2)  # 将2位置添加1个数据位置
        return x_train, y_train

    def write_file(self, wordlist, filepath):
        """
        保存字典文件
        :param wordlist:
        :param filepath:
        :return:
        """
        with open(filepath, 'w+') as f:
            f.write('\n'.join(wordlist))

    def load_pretrained_embedding(self):
        """
        加载预训练词向量
        :return:
        """
        embeddings_dict = {}  # 构建embedding{字：字向量}字典
        with open(self.embedding_file, 'r') as f:
            # 该embedding模型的数据格式是
            # 每行由字和300维的字向量组成，以空格分开
            for line in f:
                values = line.strip().split(' ')
                if len(values) < 300:
                    continue
                word = values[0]  # 记录字
                coefs = np.asarray(values[1:], dtype='float32')  # 记录字向量
                embeddings_dict[word] = coefs
        print('Found %s word vectors.' % len(embeddings_dict))
        return embeddings_dict

    def build_embedding_matrix(self):
        """
        加载词向量矩阵，实际上是通过将每一个词的 embedding 竖着排下来而已
        :return:
        """
        embedding_dict = self.load_pretrained_embedding()
        embedding_matrix = np.zeros((self.VOCAB_SIZE + 1, self.EMBEDDING_DIM))  # 生成embedding矩阵
        for word, i in self.word_dict.items():
            embedding_vector = embedding_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix

    def tokenvec_bilstm2_crf_model(self):
        """
        构建模型：使用预训练向量进行模型训练
        :return:
        """
        model = Sequential()
        embedding_layer = Embedding(self.VOCAB_SIZE + 1,  # 最大整数+1
                                    self.EMBEDDING_DIM,  # 字向量的embedding维度
                                    weights=[self.embedding_matrix],  # 字向量参数
                                    input_length=self.TIME_STAMPS,
                                    trainable=False,  # 是否训练
                                    mask_zero=True)  # 是否把0看作为一个应该被遮蔽的特殊的padding值，所有层都必须支持 masking
        model.add(embedding_layer)
        model.add(Bidirectional(LSTM(128, return_sequences=True)))  # 双向 LSTM 网络层
        model.add(Dropout(0.5))
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(Dropout(0.5))
        model.add(TimeDistributed(Dense(self.NUM_CLASSES)))  # 实体类型的全连接层
        crf_layer = CRF(self.NUM_CLASSES, sparse_target=True)
        model.add(crf_layer)
        model.compile('adam', loss=crf_layer.loss_function, metrics=[crf_layer.accuracy])
        model.summary()
        return model

    '''训练模型'''
    def train_model(self):
        x_train, y_train = self.modify_data()  # 获取整理后的数据
        model = self.tokenvec_bilstm2_crf_model()
        history = model.fit(x_train[:], y_train[:], validation_split=0.2, batch_size=self.BATCH_SIZE, epochs=self.EPOCHS)
        self.draw_train(history)
        model.save(self.model_path)
        return model

    '''绘制训练曲线'''
    def draw_train(self, history):
        # Plot training & validation accuracy values
        plt.plot(history.history['acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train'], loc='upper left')
        plt.show()
        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train'], loc='upper left')
        plt.show()
        # 7836/7836 [==============================] - 205s 26ms/step - loss: 17.1782 - acc: 0.9624
        '''
        6268/6268 [==============================] - 145s 23ms/step - loss: 18.5272 - acc: 0.7196 - val_loss: 15.7497 - val_acc: 0.8109
        6268/6268 [==============================] - 142s 23ms/step - loss: 17.8446 - acc: 0.9099 - val_loss: 15.5915 - val_acc: 0.8378
        6268/6268 [==============================] - 136s 22ms/step - loss: 17.7280 - acc: 0.9485 - val_loss: 15.5570 - val_acc: 0.8364
        6268/6268 [==============================] - 133s 21ms/step - loss: 17.6918 - acc: 0.9593 - val_loss: 15.5187 - val_acc: 0.8451
        6268/6268 [==============================] - 144s 23ms/step - loss: 17.6723 - acc: 0.9649 - val_loss: 15.4944 - val_acc: 0.8451
        '''


if __name__ == '__main__':
    ner = LSTMNER()
    ner.train_model()
"""
Purpose：
attention 模型的 Keras实现
使用的 Keras中 imdb数据集包含 5万条来自网络电影数据库的评论，来实现一个文本分类的模型
代码链接：https://github.com/Shicoder/DeepLearning_Demo/tree/master/attention
该代码是同级目录下 multi_head_attention_keras.py 代码中一部分，先搞懂这部分代码，然后再去看那个代码
-------------------------------------------------------------------------------------------------------
Notice:
这部分代码的主代码部分添加了位置信息编码，而原处理函数中没有添加
以下的两个类都继承了 keras.engine.topology.Layer，使得这两个层可以作为模型层直接添加到 keras.models的 Model中
"""
from __future__ import print_function
from keras import backend as K
from keras.engine.topology import Layer

from keras.preprocessing import sequence
from keras.datasets import imdb


class PositionEmbedding(Layer):
    """
    在阅读理解中，Q就是篇章的向量序列，K和 V为问题的向量序列
    但如果，K和 V按行打乱（相当于句子中的词序打乱），那么Attention的结果还是一样的，说明Attention模型
    本质上还是一个“词袋模型”，但是对于NLP任务中，词之间的顺序很重要
    所有这里加入了 Position Embedding位置向量，每个位置对应一个向量，并结合词向量（位置向量大小与词向量一样，两者相加而非拼接）
    在传统的 RNN和 CNN 中位置信息编码是辅助手段，但是在 Attention中是核心成分
    """
    def __init__(self, size=None, mode='sum', **kwargs):
        self.size = size  # 必须为偶数，这里的size指的是词向量的维度
        self.mode = mode
        super(PositionEmbedding, self).__init__(**kwargs)

    def call(self, x):
        if (self.size == None) or (self.mode == 'sum'):
            self.size = int(x.shape[-1])  # embedding层的第三个参数
        # 取出embedding层的batch_size和seq_len的Tensor，backend.shape()对元tensor做了切片
        batch_size, seq_len = K.shape(x)[0], K.shape(x)[1]
        '''计算sin和cos函数中的分母'''
        # keras.backend.pow(x,a) 元素级的指数运算操作，x为tensor，a为指数
        # keras.backend.arange(start, stop, step, dtype) 创建一个包含整数序列的1D tensor
        position_j = 1. / K.pow(10000., \
                                2 * K.arange(self.size / 2, dtype='float32') / self.size)
        # 在axis索引处添加一个维度
        position_j = K.expand_dims(position_j, 0)
        '''计算sin和cos函数中的分子'''
        # keras.backend.ones_like(x)  实例化与另一个张量相同形状的全1变量
        # keras.backend.cumsum(x, axis=1)  在某一指定轴，计算张量中的值的累加值
        position_i = K.cumsum(K.ones_like(x[:, :, 0]), 1) - 1  # K.arange不支持变长，只好用这种方法生成
        position_i = K.expand_dims(position_i, 2)
        '''分子分母合起来，得到sin和cos函数中的值'''
        position_ij = K.dot(position_i, position_j)
        '''将两个向量合并，获得位置编码向量'''
        # keras.backend.concatenate(tensors, axis=-1)  基于指定的轴，连接张量的列表
        position_ij = K.concatenate([K.cos(position_ij), K.sin(position_ij)], 2)
        if self.mode == 'sum':
            return position_ij + x  # 在论文中推荐使用向量相加的方式，而非向量拼接
        elif self.mode == 'concat':
            return K.concatenate([position_ij, x], 2)

    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2] + self.size)


class Attention(Layer):
    """
    Attention层的好处就是一步捕捉到全局的联系，因为它直接把序列两两比较
    主要是【多头自注意力模型】
    """
    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head  # 多头的个数，外部传值为8
        self.size_per_head = size_per_head  # 每个头的大小，外部传值为16，就是论文中使用的dk的值，需要开根号
        self.output_dim = nb_head * size_per_head  # 输出维度为 8*16，即所有注意力头拼接起来之后的尺寸
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        在Attention层中初始化 WQ, WK, WV 三个参数向量，并确定三个参数向量都是可训练的
        :param input_shape:
        :return:
        """
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)

    def Mask(self, inputs, seq_len, mode='mul'):
        """
        Mask的意义在于，用 0补全长度不足最大长度的向量后，在进行反向传播时，为了避免影响模型本身的效果，因此提出了在训练时将补全的位置mask掉
        这一步需要放在softmax之前
        具体内容见：https://blog.csdn.net/yeziyezi210/article/details/103864518
        mask.shape=[batch_size, seq_len]或[batch_size, seq_len, 1]
        :param inputs:
        :param seq_len:
        :param mode:
        :return:
        """
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:, 0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape) - 2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12

    def call(self, x):
        # 如果只传入Q_seq,K_seq,V_seq，那么就不做Mask
        # 如果同时传入Q_seq,K_seq,V_seq,Q_len,V_len，那么对多余部分做Mask
        if len(x) == 3:
            Q_seq, K_seq, V_seq = x
            Q_len, V_len = None, None
        elif len(x) == 5:
            Q_seq, K_seq, V_seq, Q_len, V_len = x
        # 对Q、K、V做线性变换
        Q_seq = K.dot(Q_seq, self.WQ)  # K.dot 为矩阵相乘
        # keras.backend.reshape(x, shape)，将张量重塑为指定的尺寸
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        # keras.backend.permute_dimensions(x, pattern)，重新排列张量的轴
        Q_seq = K.permute_dimensions(Q_seq, (0, 2, 1, 3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0, 2, 1, 3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0, 2, 1, 3))
        # 【Attention过程】计算内积，然后mask，然后softmax
        # keras.backend.batch_dot(x, y, axes)，批量化的点积，axes表示目标维度的整数或列表
        # 其功能相当于先将矩阵进行对位乘法，然后按照axes轴进行聚合加法（就是都加到一边，axes=0表示行的维度不变，每一个列相加）
        A = K.batch_dot(Q_seq, K_seq, axes=[3, 3]) / self.size_per_head ** 0.5  # 计算查询向量和键向量的积再对键向量维度开方
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        A = K.softmax(A)  # 进行softmax得到归一化分数
        # 输出并mask
        O_seq = K.batch_dot(A, V_seq, axes=[3, 2])  # softmax的值乘以值向量
        O_seq = K.permute_dimensions(O_seq, (0, 2, 1, 3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], self.output_dim


max_features = 20000  # 文本中单词词频在前max_features个的词
maxlen = 80  # 最大文本长度
batch_size = 32


print('Loading data...')

# imdb数据集包含5万条来自网络电影数据库的评论，其中2万五千条为训练数据，两万五千条为测试数据，正负评论各占50%，lable是1和0
# load_data()函数自动加载分割好的训练集和测试集
# x_train, labels_train = f['x_train'], f['y_train']
# x_test, labels_test = f['x_test'], f['y_test']
# num_words指的是只保留训练集词频rank在前num_words个的词
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

# 将所有的评论都统一成一样的长度，长度为maxlen
print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

from keras.models import Model
from keras import layers

'''以下为keras的建模过程'''
# 实例化一个keras tensor，这里参数shape创建了一个二维数组
S_inputs = layers.Input(shape=(None,), dtype='int32')
# Layers.Embedding：输入维度（input_dim）为最大文本长度，输出维度为词向量维度
# 输入数据的维度为 (batch_size, sequence_length) batch大小和序列文本长度 (2D tensor)
# 输出数据的维度为 (batch_size, sequence_length, output_dim)，output_dim 是词向量维度 (3D tensor)
embeddings = layers.Embedding(max_features, 128)(S_inputs)
# 以上两步为常见的构建keras模型的embedding层
# 下面这一步将原来的【词向量】基础上加上【位置向量】，能够稍微提高准确率，其实也是构建模型，需要训练参数
embeddings = PositionEmbedding()(embeddings)
# 计算注意力
O_seq = Attention(8,16)([embeddings,embeddings,embeddings])
# 平均池化层
O_seq = layers.GlobalAveragePooling1D()(O_seq)
O_seq = layers.Dropout(0.5)(O_seq)
outputs = layers.Dense(1, activation='sigmoid')(O_seq)

model = Model(inputs=S_inputs, outputs=outputs)
# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=5,
          validation_data=(x_test, y_test))
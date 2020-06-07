# Inheriting Keras.Model base class to build CNN custom model
# Show the whole model.sunmmary(), especially the part of 'output shape'

from tensorflow import keras
from tensorflow.keras import layers as klayers

class MLP(keras.Model):
    def __init__(self, input_shape, **kwargs):
        super(MLP, self).__init__(**kwargs)
        # Add input layer
        self.input_layer = klayers.Input(input_shape)
        
        self.embedding = klayers.Embedding(10000, 7, input_length=200)
        self.conv_1 = klayers.Conv1D(16, kernel_size=5, name = "conv_1", activation = "relu")
        self.pool_1 = klayers.MaxPool1D()
        self.conv_2 = klayers.Conv1D(128, kernel_size=2, name = "conv_2", activation = "relu")
        self.pool_2 = klayers.MaxPool1D()
        self.flatten = klayers.Flatten()
        self.dense = klayers.Dense(1,activation = "sigmoid")

        # Get output layer with `call` method
        self.out = self.call(self.input_layer)

        # Reinitial
        super(MLP, self).__init__(
            inputs=self.input_layer,
            outputs=self.out,
            **kwargs)
    
    def build(self):
        # Initialize the graph
        self._is_graph_network = True
        self._init_graph_network(
            inputs=self.input_layer,
            outputs=self.out)

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.conv_1(x)  # x.shape(batch_size, length_text+1-kernel_size, filters)
        x = self.pool_1(x)
        x = self.conv_2(x)
        x = self.pool_2(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x


if __name__ == '__main__':
    mlp = MLP(200)
    mlp.summary()

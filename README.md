# NLP_tricks
Kinds of tricks make up the world and improve our effort!

## Categories:
1. Change all the English punctuation into Chinese punctuation. Go for

module_punctuation_transform_from_En_to_Zh.py

2. Calculate the Precision, Recall and F1 score when valuating the multi-label classification prediction. Go for

module_Precision_Recall_F1_calculaton.py

3. Find if the pair sentences have the similar words in each of one.

module_similar_words_check_and_substitute.py

4. Print split timeline when checking the train epoch result in format.

printbar_split_timeline.py

5. Inherit model base class to build CNN custom model.

Demo_build_custom_CNN_model.py

and demo result as followed:

Model: "mlp_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 200)]             0         
_________________________________________________________________
embedding (Embedding)        (None, 200, 7)            70000     
_________________________________________________________________
conv_1 (Conv1D)              (None, 196, 16)           576       
_________________________________________________________________
max_pooling1d (MaxPooling1D) (None, 98, 16)            0         
_________________________________________________________________
conv_2 (Conv1D)              (None, 97, 128)           4224      
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, 48, 128)           0         
_________________________________________________________________
flatten (Flatten)            (None, 6144)              0         
_________________________________________________________________
dense (Dense)                (None, 1)                 6145      
=================================================================
Total params: 80,945
Trainable params: 80,945
Non-trainable params: 0
_________________________________________________________________

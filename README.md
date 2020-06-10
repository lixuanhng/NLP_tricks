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

Demo_build_custom_CNN_model.py and demo result as followed:

Model: "mlp_1"
layer (type)                 Output Shape              Param #   
input_1 (InputLayer)         [(None, 200)]             0         
embedding (Embedding)        (None, 200, 7)            70000     
conv_1 (Conv1D)              (None, 196, 16)           576       
max_pooling1d (MaxPooling1D) (None, 98, 16)            0         
conv_2 (Conv1D)              (None, 97, 128)           4224      
max_pooling1d_1 (MaxPooling1 (None, 48, 128)           0         
flatten (Flatten)            (None, 6144)              0         
dense (Dense)                (None, 1)                 6145      
Total params: 80,945
Trainable params: 80,945
Non-trainable params: 0

6. Transfer list group words to dict_binary words.

module_transfer_list_to_dict.py

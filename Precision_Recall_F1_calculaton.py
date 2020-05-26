"""
This function focuses on Precision, Recall and F1 score calculation 
"""
def F1_valuation(prediction_array, true_array):
    '''
    :param prediction_array: prediction multi-label list such as [[1, 0], [1, 1]]
    :param true_array: true multi-label ilst such as [[1, 0], [0, 1]] 
    :return: Precision, Recall, F1
    '''
    Precision = 0    # define Precision
    Recall = 0       # define Recall
    F1 = 0           # calculate F1 score by Macro Average rule
    n = 0            # n is the number of the elements(list) in true and prediction multi-label list
    num_vector = len(true_array)    # the length of array
    while n < num_vector:
        FP = 0    # define the count of scenario that judge is WORNG in POSITIVE example
        TP = 0    # define the count of scenario that judge is TRUE in POSITIVE example
        FN = 0    # define the count of scenario that judge is WORNG in NEGATIVE example
        if prediction_array[n] == true_array[n]:
            # If they are exactly same, then P，R，F1 + 1
            Precision += 1
            Recall += 1
            F1 += 1
        else:
            # if not, compare each elements.
            m = 0     # m is the number of the elements in both binary vector
            num_ele = len(true_array[n])    # the length of binary vector
            while m < num_ele:
                if prediction_array[n][m] == 1:
                    if true_array[n][m] == 1:
                        TP += 1
                    else:
                        FP += 1
                else:
                    if true_array[n][m] == 1:
                        FN += 1
                m += 1
            
            # After comparison between 2 binary vectors, use TP, FN, FP calculate P, R and F1
            # Pay attention when denominator equals 0
            if TP == 0 & (FP == 0 | FN == 0):
                Precision_step = 0
                Recall_step = 0
                F1_step = 0
            else:
                Precision_step = TP / (TP + FP)
                Recall_step = TP / (TP + FN)
                F1_step = 2*Precision_step*Recall_step/(Precision_step+Recall_step)
            
            Precision += Precision_step
            Recall += Recall_step
            F1 += F1_step
        # Then move to the next list
        n += 1
    # calculate the average of Precision, Recall and F1 score calculation separately
    Precision = Precision / num_vector
    Recall = Recall / num_vector
    F1 = F1 / num_vector

    return Precision, Recall, F1

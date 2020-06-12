# helper function to get mood classes
# for use with extracted features from WESAD, extracted using data_wrangling.py

def get_mood_class(data, classification_thresholds, loss_type):

    # initialize a label column
    data['label'] = 0

    if loss_type == "classification":
        
        # prevent duplicate classes
        classification_thresholds = list(set(classification_thresholds))
        
        # binary classification
        if len(classification_thresholds) == 1:
            # mark stress data 
            data.loc[data['2'] == 1, 'label'] = 1

        elif len(classification_thresholds) >= 2:
            data['label'] = 0
            data.loc[data['1'] == 1, 'label'] = 1
            data.loc[data['2'] == 1, 'label'] = 2

    # drop index and dummy ariables for labels, now redundant
    data = data.drop(['Unnamed: 0','0','1','2'], axis = 1)

    return data

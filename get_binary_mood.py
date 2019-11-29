
'''Change ordinal mood to binary or multi-class
Maintenance:
10/1/19 Created
'''

import pandas as pd
import numpy as np

# helper function to get binary mood
def get_mood_class(data, prediction_classes):

    # drop Index column because Python has its own index
    data = data.drop('Index', axis=1) 
       
    # replacing . with _ in names so that we can reference the columns in python
    data = data.rename(columns={"mood.1": "mood_1", "mood.2": "mood_2", "mood.3": "mood_3", "mood.4":"mood_4", "mood.5":"mood_5", "mood.6":"mood_6", "mood.7":"mood_7"})
    
    # list of columns to iterate through
    columns_to_change = ['mood','mood_1','mood_2','mood_3','mood_4','mood_5','mood_6','mood_7']
   
    # binary classification
    if len(prediction_classes) == 1:
        cutoff = prediction_classes[0]
        for columnName in data[columns_to_change]:
            columnSeriesObj = data[columnName]
            data.loc[columnSeriesObj < cutoff, columnName] = float(0) 
            data.loc[columnSeriesObj >= cutoff, columnName] = float(1)

    elif len(prediction_classes) >= 2:
        for columnName in data[columns_to_change]:
            columnSeriesObj = data[columnName]
            multi_class_label = float(0)
            lower_cutoff = 0
            for idx in range(0, len(prediction_classes)):
                upper_cutoff = prediction_classes[idx]
                data.loc[(columnSeriesObj >= lower_cutoff) & (columnSeriesObj < upper_cutoff), columnName] = multi_class_label
                multi_class_label += 1
                lower_cutoff = prediction_classes[idx]
            data.loc[columnSeriesObj >= upper_cutoff, columnName] = multi_class_label
    
    return data
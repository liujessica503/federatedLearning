
'''Change ordinal mood to binary, 1 if above all users' median mood, 0 if not
Maintenance:
10/1/19 Created
'''

import pandas as pd
import numpy as np

# helper function to get binary mood
def get_binary_mood(data):
    # drop Index column because Python has its own index
    data = data.drop('Index', axis=1) 
       
    # replacing . with _ in names so that we can reference the columns in python
    data = data.rename(columns={"mood.1": "mood_1", "mood.2": "mood_2", "mood.3": "mood_3", "mood.4":"mood_4", "mood.5":"mood_5", "mood.6":"mood_6", "mood.7":"mood_7"})
    data.mood.value_counts()
    
    # using universal median mood instead of each user's median mood
    data.loc[data.mood < 7, 'mood'] = 0 
    data.loc[data.mood >= 7, 'mood'] = 1 
    data.loc[data.mood_1 < 7, 'mood_1'] = 0 
    data.loc[data.mood_1 >= 7, 'mood_1'] = 1 
    data.loc[data.mood_2 < 7, 'mood_2'] = 0 
    data.loc[data.mood_2 >= 7, 'mood_2'] = 1 
    data.loc[data.mood_3 < 7, 'mood_3'] = 0 
    data.loc[data.mood_3 >= 7, 'mood_3'] = 1 
    data.loc[data.mood_4 < 7, 'mood_4'] = 0 
    data.loc[data.mood_4 >= 7, 'mood_4'] = 1 
    data.loc[data.mood_5 < 7, 'mood_5'] = 0 
    data.loc[data.mood_5 >= 7, 'mood_5'] = 1 
    data.loc[data.mood_6 < 7, 'mood_6'] = 0 
    data.loc[data.mood_6 >= 7, 'mood_6'] = 1 
    data.loc[data.mood_7 < 7, 'mood_7'] = 0 
    data.loc[data.mood_7 >= 7, 'mood_7'] = 1 
    
    return data
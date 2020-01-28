# helper function to get binary mood

def get_mood_class(data, classification_thresholds, loss_type):

    # drop Index column because Python has its own index
    data = data.drop('Index', axis=1)

    # replacing . with _ in names so we can reference the columns in python
    data = data.rename(
        columns={
            "mood.1": "mood_1",
            "mood.2": "mood_2",
            "mood.3": "mood_3",
            "mood.4": "mood_4",
            "mood.5": "mood_5",
            "mood.6": "mood_6",
            "mood.7": "mood_7",
        }
    )

    # list of columns to iterate through
    columns_to_change = [
        'mood',
        'mood_1',
        'mood_2',
        'mood_3',
        'mood_4',
        'mood_5',
        'mood_6',
        'mood_7',
    ]

    if loss_type == "classification":
        
        # prevent duplicate classes
        classification_thresholds = list(set(classification_thresholds))
        
        # binary classification
        if len(classification_thresholds) == 1:
            cutoff = classification_thresholds[0]
            for columnName in data[columns_to_change]:
                columnSeriesObj = data[columnName]
                data.loc[columnSeriesObj < cutoff, columnName] = float(0)
                data.loc[columnSeriesObj >= cutoff, columnName] = float(1)

        elif len(classification_thresholds) >= 2:
            for columnName in data[columns_to_change]:
                columnSeriesObj = data[columnName]
                multi_class_label = float(0)
                lower_cutoff = 0
                for idx in range(0, len(classification_thresholds)):
                    upper_cutoff = classification_thresholds[idx]
                    data.loc[
                        (columnSeriesObj >= lower_cutoff) &
                        (columnSeriesObj < upper_cutoff),
                        columnName
                    ] = multi_class_label
                    multi_class_label += 1
                    lower_cutoff = classification_thresholds[idx]
                data.loc[
                    columnSeriesObj >= upper_cutoff,
                    columnName
                ] = multi_class_label

    return data

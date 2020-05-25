import numpy as np

from typing import Any, List

from BaseModel import BaseModel


class MovingMeanModel(BaseModel):

    def train(
        self, user_day_data: Any, test_user_day_data: Any, test_callback=0
    )->None:
        pass

    def predict(self, user_day_data: Any, userID=None) -> List[float]:
        X_test = user_day_data.get_X()
        if X_test.shape[0] != 1:
            prediction = np.array(
                [X_test[
                    [
                        'mood_1',
                        'mood_2',
                        'mood_3',
                        'mood_4',
                        'mood_5',
                        'mood_6',
                        'mood_7'
                    ]
                ].mean(1)]
            ).transpose()
        else:
            prediction = np.array(
                X_test[
                    [
                        'mood_1',
                        'mood_2',
                        'mood_3',
                        'mood_4',
                        'mood_5',
                        'mood_6',
                        'mood_7'
                    ]
                ].mean(1)
            )

        return prediction

    def get_score(self, user_day_data)->str:
        return ""

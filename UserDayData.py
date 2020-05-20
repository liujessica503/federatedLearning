from typing import Any, List, Tuple
import numpy as np


class UserDayData:

    def __init__(self, X, user_day_pairs: List[Tuple[int]], y=None)->Any:
        if y is None:
            y = np.empty(X.shape[0])
        self.X = X
        self.y = y
        self.user_day_pairs = user_day_pairs

    def get_X(self):
        return self.X

    def get_y(self):
        return self.y

    def get_user_day_pairs(self):
        return self.user_day_pairs

    def get_users(self):
        return [x[0] for x in self.user_day_pairs]

    # get by users

    def _get_rows_for_users(self, users: List[int])->List[int]:
        rows = []
        for i in users:
            rows = rows + [
                ind for ind, v in enumerate(self.user_day_pairs) if v[0] == i
            ]

        return rows

    def get_data_for_users(self, users: List[int])->Any:
        rows = self._get_rows_for_users(users)
        return self.X.iloc[rows, :], self.y[rows]

    def get_subset_for_users(self, users: List[int])->Any:
        rows = self._get_rows_for_users(users)
        return UserDayData(
            X=self.X.iloc[rows, :],
            y=self.y[rows],
            user_day_pairs=[self.user_day_pairs[i] for i in rows]
        )

    # get by days

    def _get_rows_for_days(self, days: List[int])->List[int]:
        rows = []
        for i in days:
            rows = rows + [
                ind for ind, v in enumerate(self.user_day_pairs) if v[1] == i
            ]
        return rows

    def get_data_for_days(self, days: List[int])->Any:
        rows = self._get_rows_for_days(days)
        return self.X.iloc[rows, :], self.y[rows]

    def get_subset_for_days(self, days: List[int])->Any:
        rows = self._get_rows_for_days(days)
        return UserDayData(
            X=self.X.iloc[rows, :],
            y=self.y[rows],
            user_day_pairs=[self.user_day_pairs[i] for i in rows]
        )

    # get by users and days

    def _get_rows_for_users_on_days(
        self, users: List[int], days: List[int]
    )->List[int]:

        rows = []
        for i in users:
            for j in days:
                rows = rows + [
                    ind for ind, v in enumerate(self.user_day_pairs) if (
                        v[0] == i and v[1] == j
                    )
                ]
        return rows

    def get_data_for_users_on_days(
        self, users: List[int], days: List[int]
    )->Any:

        rows = self._get_rows_for_users_on_days(users, days)
        return self.X.iloc[rows, :], self.y[rows]

    def get_subset_for_users_on_days(
        self, users: List[int], days: List[int]
    )->Any:
        rows = self._get_rows_for_users_on_days(users, days)
        return UserDayData(
            X=self.X.iloc[rows, :],
            y=self.y[rows],
            user_day_pairs=[self.user_day_pairs[i] for i in rows]
        )

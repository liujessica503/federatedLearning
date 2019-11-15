from typing import Any, List, Tuple


class UserDayData:

    def __init__(self, X, y, user_day_pairs: List[Tuple[int]])->Any:
        self.X = X
        self.y = y
        self.user_day_pairs = user_day_pairs

    def get_X(self):
        return self.X

    def get_y(self):
        return self.y

    def get_user_day_pairs(self):
        return self.user_day_pairs

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

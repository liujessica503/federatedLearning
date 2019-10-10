import unittest
from UserDayData import UserDayData

class testUserDayData(unittest.TestCase):

    def setUp(self):
        self.user_day_pairs = [
            (0,0),
            (0,1),
            (0,3),
            (1,0),
            (1,4),
            (2,0),
            (2,1),
        ]
        self.data = UserDayData([], [], self.user_day_pairs)

    # testing _get_rows_for_days

    def test_get_rows_for_users_empty(self):
        self.assertEqual(self.data._get_rows_for_users([]), [])

    def test_get_rows_for_users_no_users(self):
        self.assertEqual(self.data._get_rows_for_users([4]), [])

    def test_get_rows_for_users_all_users(self):
        self.assertEqual(
            self.data._get_rows_for_users([0,1,2]), 
            list(range(len(self.user_day_pairs))),
        )

    def test_get_rows_for_users_0(self):
        self.assertEqual(
            self.data._get_rows_for_users([0]), list(range(3))
        )

    # testing _get_rows_for_days

    def test_get_rows_for_days_empty(self):
        self.assertEqual(self.data._get_rows_for_days([]), [])

    def test_get_rows_for_days_no_days(self):
        self.assertEqual(self.data._get_rows_for_days([2]), [])

    def test_get_rows_for_days_all_days(self):
        self.assertEqual(
            sorted(self.data._get_rows_for_days([0,1,3,4])), 
            list(range(len(self.user_day_pairs))),
        )

    def test_get_rows_for_days_0(self):
        self.assertEqual(
            self.data._get_rows_for_days([0]), [0,3,5]
        )

    # testing _get_rows_for_users_on_days

    def test_get_rows_for_users_on_days_empty(self):
        self.assertEqual(self.data._get_rows_for_users_on_days([], []), [])

    def test_get_rows_for_users_on_days_no_users(self):
        self.assertEqual(self.data._get_rows_for_users_on_days([4], [0,1,3,4]), [])

    def test_get_rows_for_users_on_days_no_days(self):
        self.assertEqual(self.data._get_rows_for_users_on_days([0,1,2], [2]), [])

    def test_get_rows_for_users_on_days_all_users_and_days(self):
        self.assertEqual(
            sorted(self.data._get_rows_for_users_on_days([0,1,2], [0,1,3,4])), 
            list(range(len(self.user_day_pairs))),
        )

    def test_get_rows_for_users_on_days_0_0(self):
        self.assertEqual(
            self.data._get_rows_for_users_on_days([0], [0]), [0]
        )

    def test_get_rows_for_users_on_days(self):
        self.assertEqual(
            sorted(self.data._get_rows_for_users_on_days([0,1], [0,3])), 
            [0, 2, 3]
        )


if __name__ == '__main__':
    unittest.main()
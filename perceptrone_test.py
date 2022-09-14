import unittest
from perceptrone import Perceptrone

class PerceptroneTest(unittest.TestCase):
    """
    PerceptroneTest

    Just a test class to test the Perceptrone implementation.
    """
    def test_perceptrone_train(self):
        """
        test_perceptrone_train

        This method tests the perceptrone training algorithm.
        """
        data_true = [
            [0, 0, 1],
            [0, 0, 2]
        ]

        data_false = [
            [1, 0, 0]
        ]

        p = Perceptrone(3)
        new_weights = p.train(data_true, data_false)
        expected = [-1, 0, 1]

        self.assertEqual(new_weights, expected)

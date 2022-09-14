"""
MIT License

Copyright (c) 2022 lambdax (and soon Felix)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import sys

#pylint: disable=pointless-string-statement
"""
class Layer():
    def __init__(self, perceptrone_list): #pylint: disable=unused-argument
        pass
"""

class Perceptrone():
    """
    Perceptrone

    Perceptrone with t = 0 and -t as a weight.

    """
    def __init__(self, input_number: int, _tuple = False) -> None:
        """
        __init__

        Constructor

        :param input_number: The number of inputs in a perceptrone.
        :type input_number: int
        :param tuple: This is for image processing to manage operations with tuples, defaults to False
        :type _tuple: bool, optional
        """

        self._input_number = input_number
        self._tuple = _tuple
        self._t = 0
        self._weights = []

    def set_t(self, value: float) -> None:
        """
        set_t

        setter-method for self._t

        :param value: new value _t
        :type value: float
        """
        
        self._t = value 

    def set_weights(self, value_list: list) -> None:
        """
        set_weights

        setter-method for self._weights

        :param value_list: new list of weights
        :type value_list: list
        """
        self._weights = value_list

    def get_output(self, input_list: list) -> int:
        """
        get_output

        This method returns the output of the perceptrone function with a given input.

        :param input: input data.
        :type input: list
        :return: boolean zero or one.
        :rtype: int
        """
        sum = 0 #pylint: disable=redefined-builtin
        for index, input_value in enumerate(input_list):
            if self._tuple:
                red, green, blue, alpha =  input_value  #pylint: disable=unused-variable
                sum += (red + green + blue) * int(self._weights[index])
            else:
                sum += input_value * int(self._weights[index])
        if sum >= self._t:
            return 1
        else:
            return 0

    def _increment_weights(self, vector: list) -> None:
        """
        _increment_weights

        This method increments each weight in self._weights

        :param vector: incrementing by vector
        :type vector: list
        """
        for index in range(len(vector)):
            if self._tuple:
                red, green, blue, _ = vector[index]
                value = (red + green + blue)
            else:
                value = vector[index]
            self._weights[index] += value

    def _decrement_weights(self, vector: list) -> None:
        """
        _decrement_weights

        This method decrements each weight in self._weights

        :param vector: decrementing by vector
        :type vector: list
        """
        for index in range(self._input_number):
            if self._tuple:
                red, green, blue, alpha = vector[index]
                value = (red + green + blue)
            else:
                value = vector[index]
            self._weights[index] -= value

    def train(self, data_true: list, data_false: list, epoch: int = (sys.maxsize * 2 + 1)) -> list:
        """
        train

        This method trains the perceptrone to a specific boolean function.
        The function must be linearly seperable.
             
        epoch = None means that the perceptrone trains until the end. For a none linearly seperable function,
        this could cause an infinity loop.

        :param data_true: The data that should return one by the boolean function.
        :type data_true: list
        :param data_false: The data that should return zero by the boolean function.
        :type data_false: list
        :param epoch: number of cycles of training, defaults to (sys.maxsize * 2 + 1).
        :type epoch: int, optional
        :return: new weights list.
        :rtype: list
        """
        #Set input t's to 1
        for vector in data_true:
            vector.insert(0, 1)

        for vector in data_false:
            vector.insert(0, 1)

        #Set weights to 0
        self._weights.clear()
        for index in range(self._input_number + 1):
            self._weights.append(0)

        #Learning algorithm
        success = 0
        len_data = len(data_true) + len(data_false)
        current_epoch = 1
        while success < len_data and current_epoch <= epoch:
            print(f"Epoch {current_epoch}")
            success = 0
            for vector in data_true:
                sum = 0 #pylint: disable=redefined-builtin
                for index, input_value in enumerate(vector):
                    if self._tuple:
                        red, green, blue, _ = input_value
                        sum += ((red + green + blue) * self._weights[index])
                    else:
                        sum += (input_value * self._weights[index])
                if sum <= self._t:
                    self._increment_weights(vector)
                else:
                    success += 1

            for vector in data_false:
                sum = 0
                for index, input_value in enumerate(vector):
                    if self._tuple:
                        red, green, blue, alpha = input_value
                        sum += (red + green + blue) * self._weights[index]
                    else:
                        sum += input_value * self._weights[index]
                if sum >= self._t:
                    self._decrement_weights(vector)
                else:
                    success += 1

            current_epoch += 1

        #Set new weights
        self._weights = [self._weights[index] for index in range(1, len(self._weights))]

        print("FINISHED", self._weights)

        return self._weights


if __name__ == "__main__":
    data_true = [
        [1, 0, 0, 6],
        [1, 5, 0, 6]
    ]
    data_false = [
        [5, 2, 3, 4],
        [1, 8, 9, 4]
    ]
    p = Perceptrone(4)
    new_weights = p.train(data_true, data_false, epoch=500)
    p.set_weights(new_weights)

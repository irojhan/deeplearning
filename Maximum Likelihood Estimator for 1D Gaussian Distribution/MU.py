import numpy as np

class MU:
    __data: np.ndarray
        
    @property
    def data(self) -> np.ndarray:
        return self.__data
        
    @data.setter
    def data(self, data: list[float]) -> None:
        self.__data = np.array(data)

    def __init__(self, data: list[float]) -> None:
        self.data = data

    def estimate_mean(self) -> float:
        return float(np.mean(self.data))

"""
Write your class here, no need to change the name of class and function.
In the constructor, read data from given .txt file,  use numpy.loadtxt.
"""

class MLE:
    def __init__(self, filename: str="data1.txt") -> None:
        super().__init__()

    def estimate_variance(self) -> float:


"""
Call class MLE and estimator mean and standard derivation
"""

mle =
mu =
sigma =

print(mu)
print(sigma)

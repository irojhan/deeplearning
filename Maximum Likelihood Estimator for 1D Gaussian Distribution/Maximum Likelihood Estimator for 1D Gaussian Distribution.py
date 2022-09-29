import numpy as np
import math
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

class MLE(MU):
    def __init__(self, filename: str="data1.txt") -> None:
        super().__init__(filename)
        self.data=np.loadtxt(filename)

    def estimate_variance(self) -> float:
        variance_sumation=0;
        xi=self.data
        for i in range(0,len(xi)):
           variance_sumation+=(xi[i]-MU.estimate_mean(xi))**2
        variance_value=(variance_sumation/len(xi))
        return variance_value
    def estimate_sigma(self) -> float:
        variance_sumation=0;
        xi=self.data
        for i in range(0,len(xi)):
           variance_sumation+=(xi[i]-MU.estimate_mean(xi))**2
        variance_value=(variance_sumation/len(xi))
        sigma_estimation=math.sqrt(variance_value)
        return sigma_estimation
    def estimate_mean(self) -> float:
        xi=self.data
        mean=(sum(xi)/len(xi))
        return mean

        
"""
Call class MLE and estimator mean and standard derivation
"""
mle =MLE()
mu=mle.estimate_mean()
sigma = mle.estimate_sigma()
variance = mle.estimate_variance()
print('mu =', mu)
print('sigma =', sigma)
print('variance =', variance)

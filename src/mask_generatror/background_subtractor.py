import numpy as np 
#abstract base clase 
from abc import ABC, abstractmethod 

class backgroundSubtractor(ABC):
    """Abstract class for background subtraction operations """
    
    @abstractmethod
    def initialize(self, frame: np.ndarray) -> None:
        pass

    @abstractmethod 
    def update_background(self, frame: np.ndarray) -> None:
        pass 

    @abstractmethod 
    def get_foreground_mask(self, frame: np.ndarray) -> None:
        pass


import cv2 
import numpy as np 
from .background_subtractor import backgroundSubtractor 

class runningAverageSubtractor(backgroundSubtractor):

    def __init__(self, learning_rate: float = 0.01, threshold: float = 30.0):
        self.learning_rate = learning_rate
        self.threshold = threshold 
        self.background_model = None 
        self.initialized = False 
    

    def initialize(self, frame: np.ndarray) -> None:
        self.background_model = frame.astype(np.float32)
        self.initialized = True 

    def update_background(self, frame: np.ndarray) -> None:
        if not self.initialized:
            self.initialize(frame)
            return 

        cv2.accumulateWeighted(frame, self.background_model, self.learning_rate)

    def get_foreground_mask(self, frame: np.ndarray) -> np.ndarray:
        if not self.initialized:
            return np.zeros(frame.shape[:2], dtype=np.uint8)
    
        #convert background to same type as frame 
        background = cv2.convertScaleAbs(self.background_model)

        #calculate difference
        diff = cv2.absdiff(frame, background)


        _, mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)

        return mask 

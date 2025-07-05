import cv2
import numpy as np 
from .background_subtractor import backgroundSubtractor


class GaussianMixtureSubtractor(backgroundSubtractor):

    def __init__(self, history: int = 500, var_threshold: float = 16.0, detect_shadows: bool = True):
        self.initialized = False

        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history = history,
            varThreshold=var_threshold,
            detectShadows = detect_shadows
        )

    def initialize(self, frame: np.ndarray) -> None:
        self.bg_subtractor.apply(frame)
        self.initialized = True

    def update_background(self, frame: np.ndarray) -> None:
        if not self.initialized:
            self.initialize(frame)

    def get_foreground_mask(self, frame: np.ndarray) -> np.ndarray:
        return self.bg_subtractor.apply(frame)


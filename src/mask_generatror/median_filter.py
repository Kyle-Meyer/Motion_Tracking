import cv2 
import numpy as np 
from collections import deque 
from .background_subtractor import backgroundSubtractor

class MedianFilterSubtractor(backgroundSubtractor):

    def __init__(self, buffer_size: int = 20, threshold: float = 30.0):
        self.buffer_size = buffer_size
        self.threshold = threshold
        self.frame_buffer = None 
        self.background_model = None 
        self.initialized = False 
        self.frame_count = 0 

    def initialize(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        self.frame_buffer = deque(maxlen=self.buffer_size)
        self.frame_buffer.append(frame.copy())
        self.background_model = frame.astype(np.float32)
        self.initialized = True 
        self.frame_count = 1

    def update_background(self, frame: np.ndarray) -> None:
        if not self.initialized:
            self.initialize(frame)
            return 

        # Add frame to the queue
        self.frame_buffer.append(frame.copy())
        self.frame_count += 1
        
        # Update background model only after we have enough frames
        if len(self.frame_buffer) >= self.buffer_size:
            frame_stack = np.stack(list(self.frame_buffer), axis=0)
            self.background_model = np.median(frame_stack, axis=0)

    def get_foreground_mask(self, frame: np.ndarray) -> np.ndarray:  
        if not self.initialized or self.background_model is None:
            return np.zeros(frame.shape[:2], dtype=np.uint8)

        # Ensure background model is same type as input frame
        background = self.background_model.astype(frame.dtype)

        # Calculate absolute difference
        diff = cv2.absdiff(frame, background)

        # Convert to grayscale if needed
        if len(diff.shape) == 3:
            diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # Apply threshold to create binary mask
        _, mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        
        return mask 

import cv2 
import numpy as np 
from typing import Tuple 

class MaskProcessor:
    def __init__(self, 
                 gaussian_blur_kernel: Tuple[int, int] = (5, 5),
                 morphology_kernel_size: int = 5, 
                 min_contour_area: int = 500):
        self.gaussian_blur_kernel = gaussian_blur_kernel
        self.morphologyKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morphology_kernel_size, morphology_kernel_size))
        self.min_contour_area = min_contour_area

    def apply_preprocessing(self, frame: np.ndarray) -> np.ndarray:
        #convert to grayscale 
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        #apply gaussian blur to reduce noise 
        blurred = cv2.GaussianBlur(frame, self.gaussian_blur_kernel, 0)

        return blurred

    def clean_mask(self, mask: np.ndarray) -> np.ndarray:
        #initial round of noise removal
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morphologyKernel)

        #fill holes that would have resulted from the above operation 
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, self.morphologyKernel)

        return cleaned

    def filter_by_area(self, mask: np.ndarray) -> np.ndarray:
        #find contours 
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        #create a new mask with only large contours 
        filtered_mask = np.zeros_like(mask);

        for contour in contours:
            #if these are contours we want, fill them in
            if cv2.contourArea(contour) >= self.min_contour_area:
                cv2.fillPoly(filtered_mask, [contour], 255)

        return filtered_mask

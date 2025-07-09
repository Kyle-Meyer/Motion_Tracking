import cv2 
from cv2.gapi import mask
import numpy as np 
from typing import Tuple 

class MaskProcessor:
    def __init__(self, 
                 gaussian_blur_kernel: Tuple[int, int] = (5, 5),
                 morphology_kernel_size: int = 5, 
                 min_contour_area: int = 500,
                 skip_area_filtering: bool = False,
                 use_gentle_cleaning: bool = False,
                 fill_person_gaps: bool = False):

        self.gaussian_blur_kernel = gaussian_blur_kernel
        self.morphologyKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morphology_kernel_size, morphology_kernel_size))
        self.min_contour_area = min_contour_area
        self.skip_area_filtering = skip_area_filtering
        self.use_gentle_cleaning = use_gentle_cleaning
        self.fill_person_gaps = fill_person_gaps

    def apply_preprocessing(self, frame: np.ndarray) -> np.ndarray:

        frame = cv2.medianBlur(frame, 5)
        #apply gaussian blur to reduce noise 
        blurred = cv2.GaussianBlur(frame, self.gaussian_blur_kernel, 0)

        return blurred

    def clean_mask(self, mask: np.ndarray) -> np.ndarray:
        # Ensure mask is binary
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)

        #initial round of noise removal
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morphologyKernel)

        #fill holes that would have resulted from the above operation 
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, self.morphologyKernel)

        return cleaned
    def clean_mask_gentle(self, mask: np.ndarray) -> np.ndarray:
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        
        #use a small ellipsoid kernel for erosion 
        small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, small_kernel)
        
        #now use a larger kernel for closing and connecting regions
        large_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, large_kernel)

        return cleaned

    def filter_by_area(self, mask: np.ndarray) -> np.ndarray:
        if self.skip_area_filtering:
            return mask 

        #find contours 
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        #create a new mask with only large contours 
        filtered_mask = np.zeros_like(mask);

        for contour in contours:
            #if these are contours we want, fill them in
            if cv2.contourArea(contour) >= self.min_contour_area:
                cv2.fillPoly(filtered_mask, [contour], 255)

        return filtered_mask

    def fill_gaps_in_person(self, mask: np.ndarray) -> np.ndarray:
        if not self.fill_person_gaps:
            return mask 

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mask 

        #work with the largest contour in the image basically assuming this is a person 
        largest_contour = max(contours, key=cv2.contourArea)

        # get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)

        #dialate within the bounding box 
        roi_mask = mask[y:y+h, x:x+w]

        #apply the closing operation 
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        filled_roi = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel)

        #put back into full mask 
        result_mask = mask.copy()
        result_mask[y:y+h, x:x+w] = filled_roi 
        
        return result_mask

import numpy as np 
from typing import Optional, List 
from .enums import BackgroundModelType
from .running_average_subtractor import runningAverageSubtractor
from .gaussian_mixture_subtractor import GaussianMixtureSubtractor
from .mask_processor import MaskProcessor
from .median_filter import MedianFilterSubtractor

class MotionDetector:

    def __init__(self,
                 background_model_type: BackgroundModelType = BackgroundModelType.RUNNING_AVERAGE,
                 subtractor_params: Optional[dict] = None,
                 processor_params: Optional[dict] = None):
        subtractor_params = subtractor_params or {}
        if background_model_type == BackgroundModelType.RUNNING_AVERAGE:
            self.bg_subtractor = runningAverageSubtractor(**subtractor_params)
        elif background_model_type == BackgroundModelType.GAUSSIAN_MIXTURE:
            print("running gaussian .....")
            self.bg_subtractor = GaussianMixtureSubtractor(**subtractor_params)
        elif background_model_type == BackgroundModelType.MEDIAN_FILTER:
            self.bg_subtractor = MedianFilterSubtractor(**subtractor_params)
        else:
            raise ValueError(f"Unsupported background model type {background_model_type}")

        processor_params = processor_params or {}
        self.mask_processor = MaskProcessor(**processor_params)

        self.masks = []
        self.frame_count = 0

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        self.frame_count += 1

        #start our preprocessing
        processed_frame = self.mask_processor.apply_preprocessing(frame)

        #update background model 
        self.bg_subtractor.update_background(processed_frame)

        #get foreground mask 
        raw_mask = self.bg_subtractor.get_foreground_mask(processed_frame)

        #clean up mask 
        cleaned_mask = self.mask_processor.clean_mask(raw_mask)

        #Filter by area 
        area_filtered_mask = self.mask_processor.filter_by_area(cleaned_mask)
        
        #fill gaps if chosen 
        final_mask = self.mask_processor.fill_gaps_in_person(area_filtered_mask)

        #Store the mask 
        self.masks.append(final_mask.copy())

        return final_mask 

    def get_masks(self) -> List[np.ndarray]:
        return self.masks 

    def get_frame_count(self) -> int:
        return self.frame_count

    def reset(self) -> None:
        self.masks.clear()
        self.frame_count = 0




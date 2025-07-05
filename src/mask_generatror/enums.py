from enum import Enum 

""" Enums to try the different background modeling types """
class BackgroundModelType(Enum):
    RUNNING_AVERAGE = 'running_average'
    GAUSSIAN_MIXTURE = 'gaussian_mixture'
    MEDIAN_FILTER = 'median_filter'

import cv2 
import numpy as np 
from typing import List, Tuple, Optional 
from dataclasses import dataclass 

#data struct to represent the bounding box 
@dataclass 
class BoundingBox:
    x: int 
    y: int 
    width: int 
    height: int 
    frame_number: int 
    confidence: float = 1.0 

    @property 
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property 
    def area(self) -> int:
        return self.width * self.height 

    def get_rect(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)

class BoundingBoxTracker:
    def __init__(self,
                 min_contour_area: int = 500,
                 max_distance_threshold: float = 50.0,
                 min_consecutive_frames: int = 30,
                 padding: int = 10):
        self.min_contour_area = min_contour_area
        self.max_distance_threshold = max_distance_threshold
        self.min_consecutive_frames = min_consecutive_frames
        self.padding = padding

        self.current_tracks: List[List[BoundingBox]] = []
        self.completed_tracks: List[List[BoundingBox]] = []
        self.frame_count = 0 

    def extract_bounding_boxes_from_mask(self, mask: np.ndarray) -> List[BoundingBox]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = [] 
        for contour in contours:
            if cv2.contourArea(contour) >= self.min_contour_area:
                x, y, w, h = cv2.boundingRect(contour)

                height, width = mask.shape 
                x = max(0, x - self.padding)
                y = max(0, y - self.padding)
                w = min(width - x, w + 2 * self.padding)
                h = min(height - y, h + 2 * self.padding)

                box = BoundingBox(x, y, w, h, self.frame_count)
                boxes.append(box)
        
        return boxes 

    def calculated_distance(self, box1: BoundingBox, box2: BoundingBox) -> float:
        center1 = box1.center 
        center2 = box2.center 
        return np.sqrt((center1[0]- center2[0])**2 + (center1[1] - center2[1])**2) 

    def match_boxes_to_tracks(self, new_boxes: List[BoundingBox]) -> List[Tuple[int, int]]:
        matches = [] 
        
        if not self.current_tracks or not new_boxes:
            return matches 

        distances = [] 
        for track_idx, track in enumerate(self.current_tracks):
            last_box = track[-1]
            for box_idx , new_box in enumerate(new_boxes):
                distance = self.calculated_distance(last_box, new_box)
                distances.append((distance, track_idx, box_idx))

        #greedy sort 
        distances.sort()
        used_tracks = set()
        used_boxes = set() 

        for distance, track_idx, box_idx in distances:
            if (distance <= self.max_distance_threshold and 
                track_idx  not in used_tracks and 
                box_idx not in used_boxes):

                matches.append((track_idx, box_idx))
                used_tracks.add(track_idx)
                used_boxes.add(box_idx)

        return matches 
    
    def get_track_path(self, track: List[BoundingBox]) -> List[Tuple[int, int]]:
        return [box.center for box in track]

    def draw_track_paths(self, frame: np.ndarray,
                         tracks: Optional[List[List[BoundingBox]]] = None, 
                         path_color: Tuple[int, int, int] = (0,0,255),
                         path_thickness: int = 2, 
                         show_start_end: bool = True) -> np.ndarray: 
        if tracks is None:
            tracks = self.get_valid_tracks()

        result = frame.copy()

        for track_idx, track in enumerate(tracks):
            if len(track) >= self.min_consecutive_frames:
                path = self.get_track_path(track)

                for i in range (1, len(path)):
                    cv2.line(result, path[i-1], path[i], path_color, path_thickness)

                if show_start_end and len(path) > 1:
                    cv2.circle(result, path[0], 8, (0, 255, 0), -1)
                    cv2.circle(result, path[-1], 8, (255, 0, 0), -1)

                    label = f"Track {track_idx+1} ({len(track)} frames)"
                    cv2.putText(result, label, (path[0][0] + 10, path[0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return result 


    def update_tracks(self, mask: np.ndarray) -> None:
        self.frame_count += 1 
        new_boxes = self.extract_bounding_boxes_from_mask(mask)

        matches = self.match_boxes_to_tracks(new_boxes)

        matched_tracks = set() 
        matched_boxes = set() 

        for track_idx, box_idx in matches:
            self.current_tracks[track_idx].append(new_boxes[box_idx])
            matched_tracks.add(track_idx)
            matched_boxes.add(box_idx)

        #move unmatched tracks to completed if they are long enough 
        remaining_tracks = [] 
        for i, track in enumerate(self.current_tracks):
            if i not in matched_tracks:
                if len(track) >= self.min_consecutive_frames:
                    self.completed_tracks.append(track)
            else:
                remaining_tracks.append(track)

        #start new tracks for unmatched boxes 
        for i, box in enumerate(new_boxes):
            if i not in matched_boxes:
                remaining_tracks.append([box])

        self.current_tracks = remaining_tracks

    def finalize_tracking(self) -> None:
        for track in self.current_tracks:
            if len(track) >= self.min_consecutive_frames:
                self.completed_tracks.append(track)
        self.current_tracks.clear()

    def get_valid_tracks(self) -> List[List[BoundingBox]]:
        valid_tracks = []

        valid_tracks.extend(self.completed_tracks)

        for track in self.current_tracks:
            if len(track) >= self.min_consecutive_frames:
                valid_tracks.append(track)
        
        return valid_tracks

    def get_current_bounding_boxes(self) -> List[BoundingBox]:
        current_boxes = []
        for track in self.current_tracks:
            if track and track[-1].frame_number == self.frame_count:
                current_boxes.append(track[-1])
        return current_boxes 

    def draw_bounding_boxes(self, frame: np.ndarray,
                            boxes: Optional[List[BoundingBox]] = None,
                            color: Tuple[int, int, int] = (0,255,0),
                            thickness: int = 2) -> np.ndarray:
        if boxes is None: 
            boxes = self.get_current_bounding_boxes()

        result = frame.copy() 
        for box in boxes:
            # the actual box
            cv2.rectangle(result, (box.x, box.y),
                          (box.x + box.width, box.y + box.height),
                          color, thickness)

            #add frame count label 
            label = f"Frame {box.frame_number}"
            cv2.putText(result, label, (box.x, box.y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # completed frame 
        return result 

    def draw_combined_visualization(self, frame: np.ndarray) -> np.ndarray:
        result = self.draw_track_paths(frame)

        result = self.draw_bounding_boxes(result)
        
        return result 

    def get_statistics(self) -> dict: 
        valid_tracks = self.get_valid_tracks()

        stats = {
            'total_valid_tracks': len(valid_tracks),
            'frames_processed': self.frame_count,
            'current_active_tracks': len(self.current_tracks),
            'completed_tracks': len(self.completed_tracks),
            'min_consecutive_frames': self.min_consecutive_frames
        }

        if valid_tracks:
            track_lengths = [len(track) for track in valid_tracks]
            stats.update({
                'average_track_length': np.mean(track_lengths),
                'longest_track_length': max(track_lengths),
                'shortest_track_length': min(track_lengths)
            })
        return stats 

    def reset(self) -> None:
        self.current_tracks.clear()
        self.completed_tracks.clear()
        self.frame_count = 0

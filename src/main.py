import cv2 
import os 
import glob
import argparse 
from mask_generatror import MotionDetector, BackgroundModelType
from bounding_box import BoundingBoxTracker, BoundingBox

def save_masks(masks, output_dir, prefix="mask_"):
    os.makedirs(output_dir, exist_ok=True)

    for i, mask in enumerate(masks):
        filename = f"{prefix}{i:04d}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, mask)

def save_bounding_box_frames(frames_with_boxes, output_dir, prefix="bbox_"):
    os.makedirs(output_dir, exist_ok=True)
    for i, mask in enumerate(frames_with_boxes):
        filename = f"{prefix}{i:04d}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, mask)

def main():
    parser = argparse.ArgumentParser(description='Motion Detection and Tracking')
    parser.add_argument('--input-dir', required=True, help='Directory containing images')
    parser.add_argument('--pattern', default='*.jpg', help='Image file pattern (e.g., "*.jpg", "*.png")')
    parser.add_argument('--output', help='where to save the output')
    parser.add_argument('--model', default='running_average', 
                       choices=['running_average', 'gaussian_mixture'],
                       help='Background model type')
    parser.add_argument('--display', action='store_true', help='Display results in real-time')
    parser.add_argument('--min-consecutive-frames', type=int, default=30,
                        help='Minimum consecutive frames for valid tracking')
    parser.add_argument('--save-bbox-frames', action='store_true',
                        help='Save frames with bounding boxes drawn')
    args = parser.parse_args()

    model_map = {
        'running_average': BackgroundModelType.RUNNING_AVERAGE,
        'gaussian_mixture': BackgroundModelType.GAUSSIAN_MIXTURE
    }

    image_pattern = os.path.join(args.input_dir, args.pattern)
    image_files = sorted(glob.glob(image_pattern))

    if not image_files:
        print(f"Error, no images found in {args.input_dir} matching {args.pattern}")
        return 

    # Initialize motion detector with proper parameters for each model
    if args.model == 'running_average':
        print("Running average across pixel width")

        detector = MotionDetector(
            background_model_type=model_map[args.model],
            subtractor_params={
                'learning_rate': 0.01,
                'threshold': 16.0
            },
            processor_params={
                'gaussian_blur_kernel': (3,3), 
                'morphology_kernel_size': 3, 
                'min_contour_area': 200,
                'skip_area_filtering': False,
                'use_gentle_cleaning': True,
                'fill_person_gaps': True
            }
        )
    else:  # gaussian_mixture
        detector = MotionDetector(
            background_model_type=model_map[args.model],
            subtractor_params={
                'history': 200,
                'var_threshold': 8.0,
                'detect_shadows': True
            },
            processor_params={
                'min_contour_area': 1000
            }
        )
    #init bounding box tracker 
    bbox_tracker = BoundingBoxTracker(
                min_contour_area=detector.mask_processor.min_contour_area,
                max_distance_threshold = 75.0,
                min_consecutive_frames=args.min_consecutive_frames,
                padding = 10
    )

    masks = []
    frames_with_boxes = []
    frame_count = 0 

    for image_path in image_files:
        #load the image 
        frame = cv2.imread(image_path)
        if frame is None: 
            print(f"Warning could not load {image_path}")
            continue 

        #generate the mask 
        mask = detector.process_frame(frame)
        masks.append(mask)
        
        #update the bounding box tracker and store the drawn frames
        bbox_tracker.update_tracks(mask)
        frame_with_boxes = bbox_tracker.draw_bounding_boxes(frame)
        frames_with_boxes.append(frame_with_boxes)

        frame_count += 1 
        if frame_count % 10 == 0:
            print(f"Processed {frame_count}/{len(image_files)} images")

        #display results (optional)
        if args.display:
            cv2.imshow('Original Image', frame)
            cv2.imshow('Generated Mask', mask)
            cv2.imshow('bounding boxes', frame_with_boxes)

            #quit out key 
            if cv2.waitKey(30) & 0xFF == ord('q'): #30ms delay between frames 
                break
        
    
    bbox_tracker.finalize_tracking()

    if args.display:
        cv2.destroyAllWindows()

    #save output 
    if args.output:
        masks_output_dir = os.path.join(args.output, 'masks')
        save_masks(detector.get_masks(), masks_output_dir)
        if args.save_bbox_frames:
            bbox_output_dir = os.path.join(args.output, 'bounding_boxes')
            save_bounding_box_frames(frames_with_boxes, bbox_output_dir)

    print(f"\nProcessing complete!")
    print(f"Total frames processed: {detector.get_frame_count()}")
    print(f"Masks generated: {len(detector.get_masks())}")
    print(f"Masks saved to: {masks_output_dir}")

    if len(detector.get_masks()) >= 60:
        print("60+ frames generated")
    else:
        print("did NOT generate 60 or more frames")

if __name__ == "__main__":
    main()



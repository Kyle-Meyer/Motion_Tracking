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

def save_path_frames(frames_with_paths, output_dir, prefix="paths_"):
    os.makedirs(output_dir, exist_ok=True)
    for i, frame in enumerate(frames_with_paths):
        filename = f"{prefix}{i:04d}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, frame)

def main():
    parser = argparse.ArgumentParser(description='Motion Detection and Tracking')
    
    # Basic arguments
    parser.add_argument('--input-dir', required=True, help='Directory containing images')
    parser.add_argument('--pattern', default='*.jpg', help='Image file pattern (e.g., "*.jpg", "*.png")')
    parser.add_argument('--output', help='where to save the output')
    parser.add_argument('--model', default='gaussian_mixture', 
                       choices=['running_average', 'gaussian_mixture', 'median_filter'],
                       help='Background model type')
    parser.add_argument('--display', action='store_true', help='Display results in real-time')
    parser.add_argument('--min-consecutive-frames', type=int, default=30,
                        help='Minimum consecutive frames for valid tracking')
    parser.add_argument('--save-bbox-frames', action='store_true',
                        help='Save frames with bounding boxes drawn')
    
    # Running Average Subtractor parameters
    parser.add_argument('--learning-rate', type=float, default=0.005,
                        help='Learning rate for running average (default: 0.005)')
    parser.add_argument('--ra-threshold', type=float, default=40.0,
                        help='Threshold for running average subtractor (default: 40.0)')
    
    # Gaussian Mixture Subtractor parameters
    parser.add_argument('--history', type=int, default=700,
                        help='History length for Gaussian Mixture Model (default: 700)')
    parser.add_argument('--var-threshold', type=float, default=75.0,
                        help='Variance threshold for Gaussian Mixture Model (default: 75.0)')
    parser.add_argument('--detect-shadows', action='store_true', default=True,
                        help='Enable shadow detection for Gaussian Mixture Model')
    parser.add_argument('--no-detect-shadows', dest='detect_shadows', action='store_false',
                        help='Disable shadow detection for Gaussian Mixture Model')
    
    # Median Filter Subtractor parameters
    parser.add_argument('--buffer-size', type=int, default=20,
                        help='Buffer size for median filter (default: 20)')
    parser.add_argument('--mf-threshold', type=float, default=40.0,
                        help='Threshold for median filter subtractor (default: 40.0)')
    
    # Mask Processor parameters
    parser.add_argument('--gaussian-blur-kernel', type=int, nargs=2, default=[1, 1],
                        help='Gaussian blur kernel size (width height) (default: 1 1)')
    parser.add_argument('--morphology-kernel-size', type=int, default=1,
                        help='Morphology kernel size (default: 1)')
    parser.add_argument('--min-contour-area', type=int, default=1000,
                        help='Minimum contour area for filtering (default: 1000)')
    parser.add_argument('--skip-area-filtering', action='store_true',
                        help='Skip area filtering in mask processing')
    parser.add_argument('--use-gentle-cleaning', action='store_true',
                        help='Use gentle cleaning for mask processing')
    parser.add_argument('--fill-person-gaps', action='store_true',
                        help='Fill gaps in person detection')
    
    # Bounding Box Tracker parameters
    parser.add_argument('--max-distance-threshold', type=float, default=75.0,
                        help='Maximum distance threshold for tracking (default: 75.0)')
    parser.add_argument('--bbox-padding', type=int, default=10,
                        help='Padding for bounding boxes (default: 10)')

    args = parser.parse_args()

    model_map = {
        'running_average': BackgroundModelType.RUNNING_AVERAGE,
        'gaussian_mixture': BackgroundModelType.GAUSSIAN_MIXTURE,
        'median_filter': BackgroundModelType.MEDIAN_FILTER
    }

    image_pattern = os.path.join(args.input_dir, args.pattern)
    image_files = sorted(glob.glob(image_pattern))

    if not image_files:
        print(f"Error, no images found in {args.input_dir} matching {args.pattern}")
        return 

    # Build subtractor and processor parameters based on model type
    if args.model == 'running_average':
        print("Running average across pixel width")
        subtractor_params = {
            'learning_rate': args.learning_rate,
            'threshold': args.ra_threshold
        }
        processor_params = {
            'gaussian_blur_kernel': tuple(args.gaussian_blur_kernel), 
            'morphology_kernel_size': args.morphology_kernel_size, 
            'min_contour_area': args.min_contour_area,
            'skip_area_filtering': args.skip_area_filtering,
            'use_gentle_cleaning': args.use_gentle_cleaning,
            'fill_person_gaps': args.fill_person_gaps
        }
    elif args.model == 'gaussian_mixture':
        print("Using Gaussian Mixture Model")
        subtractor_params = {
            'history': args.history,
            'var_threshold': args.var_threshold,
            'detect_shadows': args.detect_shadows
        }
        processor_params = {
            'gaussian_blur_kernel': tuple(args.gaussian_blur_kernel),
            'morphology_kernel_size': args.morphology_kernel_size,
            'min_contour_area': args.min_contour_area,
            'skip_area_filtering': args.skip_area_filtering,
            'use_gentle_cleaning': args.use_gentle_cleaning,
            'fill_person_gaps': args.fill_person_gaps
        }
    elif args.model == 'median_filter':
        print("Using Median Filter background subtraction")
        subtractor_params = {
            'buffer_size': args.buffer_size,
            'threshold': args.mf_threshold
        }
        processor_params = {
            'gaussian_blur_kernel': tuple(args.gaussian_blur_kernel), 
            'morphology_kernel_size': args.morphology_kernel_size, 
            'min_contour_area': args.min_contour_area,
            'skip_area_filtering': args.skip_area_filtering,
            'use_gentle_cleaning': args.use_gentle_cleaning,
            'fill_person_gaps': args.fill_person_gaps
        }

    # Initialize motion detector with parameters from command line
    detector = MotionDetector(
        background_model_type=model_map[args.model],
        subtractor_params=subtractor_params,
        processor_params=processor_params
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
    frames_with_paths = []
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
        frame_with_paths = bbox_tracker.draw_combined_visualization(frame)
        frames_with_boxes.append(frame_with_boxes)
        frames_with_paths.append(frame_with_paths)


        frame_count += 1 
        if frame_count % 10 == 0:
            print(f"Processed {frame_count}/{len(image_files)} images")

        #display results (optional)
        if args.display:
            cv2.imshow('Original Image', frame)
            cv2.imshow('Generated Mask', mask)
            cv2.imshow('bounding boxes', frame_with_boxes)
            cv2.imshow('Paths & tracking', frame_with_paths)

            #quit out key 
            if cv2.waitKey(30) & 0xFF == ord('q'): #30ms delay between frames 
                break
        
    print("finalize tracking...") 
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
            print("saving paths.....")
            paths_output_dir = os.path.join(args.output, 'paths')
            save_path_frames(frames_with_paths, paths_output_dir)

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

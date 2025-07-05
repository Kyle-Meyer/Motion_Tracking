import cv2 
import os 
import argparse 
from mask_generatror import MotionDetector, BackgroundModelType 

def save_masks(masks, output_dir, prefix="mask_"):
    os.makedirs(output_dir, exist_ok=True)

    for i, mask in enumerate(masks):
        filename = f"{prefix}{i:04d}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, mask)

def main():
    parser = argparse.ArgumentParser(description='Motion Detection and Tracking')
    parser.add_argument('--input', required=True, help='Input video file path')
    parser.add_argument('--output', help='where to save the output')
    parser.add_argument('--model', default='running_average', 
                       choices=['running_average', 'gaussian_mixture'],
                       help='Background model type')
    parser.add_argument('--display', action='store_true', help='Display results in real-time')

    args = parser.parse_args()

    model_map = {
        'running_average': BackgroundModelType.RUNNING_AVERAGE,
        'gaussian_mixture': BackgroundModelType.GAUSSIAN_MIXTURE
    }

    #initialize motion detector 
    detector = MotionDetector(
        background_model_type=model_map[args.model],
        subtractor_params = {
            'learning_rate': 0.01,
            'threshold': 30.0
        },
        processor_params={
            'min_contour_area': 1000
        }
    )

    #process the video 
    cap = cv2.VideoCapture(args.input)

    if not cap.isOpened():
        print(f"Error: could not open video file {args.input}")
        return 
    
    print(f"Processing Video: {args.input}")
    print(f"Using Model: {args.model}")

    frame_count = 0 
    while True:
        ret, frame = cap.read()
        if not ret:
            break 

        mask = detector.process_frame(frame)

        frame_count += 1 
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames")

        if args.display:
            cv2.imshow('Original', frame)
            cv2.imshow('Motion Mask', mask)

            #exit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if args.display:
        cv2.destroyAllWindows()

    masks_output_dir = os.path.join(args.output, 'masks')
    save_masks(detector.get_masks(), masks_output_dir)

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



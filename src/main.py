import cv2 
import os 
import glob
import argparse 
import tkinter as tk 
from tkinter import ttk, filedialog, messagebox
import threading 
import numpy as np 
from mask_generatror import MotionDetector, BackgroundModelType
from bounding_box import BoundingBoxTracker, BoundingBox

class MotionDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title = ("Motion Detection and Tracking")
        self.root.geometry("900x700")

        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.model_type = tk.StringVar(value="running_average")
        self.file_pattern = tk.StringVar(value="*.jpg")

        #running average
        self.ra_learning_rate = tk.DoubeVar(value=0.01)
        self.ra_threshold = tk.DoubeVar(value=16.0)
        self.ra_gaussian_blur = tk.StringVar("3,3")
        self.ra_morph_kernel = tk.IntVar(value=3)
        self.ra_min_contour = tk.IntVar(value=200)
        self.ra_skip_filtering = tk.BooleanVar(value=False)
        self.ra_gentle_clearing = tk.BooleanVar(value=True)
        self.ra_fill_gaps = tk.BooleanVar(value=True)

        #gaussian method  
        self.gm_history = tk.IntVar(value=200)
        self.gm_var_threshold = tk.DoubleVar(value=8.0)
        self.gm_detect_shadows = tk.BooleanVar(value=True)
        self.gm_min_contour = tk.IntVar(value=1000)

        #bounding box parameters 
        self.bbox_max_distance = tk.DoubleVar(value=75.0)
        self.bbox_min_consecutive = tk.IntVar(value=30)
        self.bbox_padding = tk.IntVar(value=10)
        self.save_bbox_framess = tk.BooleanVar(value=True)

        #internal states 
        self.is_processing = False
        self.detector = None 
        self.bbox_tracker = None 
        self.image_files = []
        self.current_frame_idx = 0
        
        #call our setup 
        self.setup_ui()

    def setup_ui(self):
        canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollableregion=canvas.bbox("all"))
        )

        canvas.create_window((0,0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        main_frame = ttk.Frame(scrollable_frame, paddin="10")
        main_frame.pack(fill="both", expand=True)
        
        file_frame = ttk.LabelFrame(main_frame, text="File Selection", padding="5")
        file_frame.pack(fill="x", pady=5)
        file_frame.columnfigure(1, weight=1)

        ttk.Label(file_frame, text="Input Directory:").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Entry(file_frame, textvariable=self.output_dir, width=50).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_input_dir).grid(row=0, column=2, padx=5)

        ttk.Label(file_frame, text="Output Directory:").grid(row=1, column=0, sticky=tk.W, padx=5)
        ttk.Entry(file_frame, textvariable=self.output_dir, width=50).grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_output_dir).grid(row=1, column=2, padx=5)       

        ttk.Label(file_frame, text="File Pattern:").grid(row=2, column=0, sticky=tk.W, padx=5)
        ttk.Entry(file_frame, textvariable=self.file_pattern, widht=20).grid(row=2, column=1, sticky=tk.W, padx=5)

        #Model selection 
        model_frame = ttk.LabelFrame(main_frame, text="Model Selection", padding="5")
        model_frame.pack(fill="x", pady=5)

        ttk.RadioButton(model_frame, text="Running Average", variable=self.model_type,
                        value="running_average", command=self.on_model_change).pack(side=tk.LEFT, padx=5)
        ttk.RadioButton(model_frame, text="Gaussian Mixture", variable=self.model_type,
                        value="gaussian_mixture", command=self.on_model_change).pack(side=tk.LEFT, padx=5)

        #model params 
        self.params_frame = ttk.Label(main_frame, text="Model Parameters", padding="5")
        self.params_frame.pack(fill="x", pady=5)
        self.params_frame.columnconfigure(1, weight=1)

        self.create_model_parameter_widgets()

        bbox_frame = ttk.LabelFrame(main_frame, text="Bounding Box tracking parameters", padding="5")
        bbox_frame.pack(fill="x", pady=5)
        bbox_frame.columnconfigure(1, weight=1)

        ttk.Label(bbox_frame, text="Max Distance Threshold:").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Entry(bbox_frame, textvariable=self.bbox_max_distance, width=10).grid(row=0, column=1, sticky=tk.W, padx=5)

        ttk.Label(bbox_frame, text="Min Consecutive Frames:").grid(row=1, column=0, sticky=tk.W, padx=5)
        ttk.Entry(bbox_frame, textvariable=self.bbox_min_consecutive, width=10).grid(row=1, column=1, sticky=tk.W, padx=5)

        ttk.Label(bbox_frame, text="Padding:").grid(row=2, column=0, sticky=tk.W, padx=5)
        ttk.Entry(bbox_frame, textvariable=self.bbox_padding, width=10).grid(row=2, column=1, sticky=tk.W, padx=5)

        ttk.Checkbutton(bbox_frame, text="Save Bounding box frames:",
                        variable=self.save_bbox_frames).grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=5)

        control_frame = ttk.Frame(main_frame)
        control_frame.pack(pady=10)

        ttk.Button(control_frame, text="Start Processing", command=self.start_processing).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Stop Processing", command=self.stop_processing).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Reset", command=self.reset_processing).pack(side=tk.LEFT, padx=5)

        #progress bar 
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill="x", pady=5)

        self.status_label = ttk.Label(main_frame, text="Ready")
        self.status_label.pack(pady=5)

        stats_frame = ttk.LabelFrame(main_frame, text="Processing Statistics", padding="5")
        stats_frame.pack(fill="x", pady=5)
        
        self.stats_text = tk.Text(stats_frame, height=6, width=70)
        stats_scrollbar = ttk.Scrollbar(stats_frame, orient="vertical", command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=stats_scrollbar.set)
        self.stats_text.pack(side="left", fill="both", expand=True)
        self.stats_scrollbar.pack(side="right", fill="y")

        self.on_model_change()

    def create_model_parameter_widgets(self):
        self.ra_widgets = {}

        self.ra_labels = []

        #Learning rate 
        self.ra_labels.append(ttk.Label(self.params_frame, text="Learning Rate:"))
        self.ra_widgets['learning_rate'] = ttk.Entry(self.params_frame, textvariable=self.ra_learning_rate, width=10)

        #threshold 
        self.ra_labels.append(ttk.Label(self.params_frame, text="Threshold:"))
        self.ra_widgets['threshold'] = ttk.Entry(self.params_frame, textvariable=self.ra_learning_rate, width=10)

        #Gaussain blur kernel 
        self.ra_labels.append(ttk.Label(self.params_frame, text="Gaussian Blur Kernel:"))
        self.ra_widgets['gaussian_blur'] = ttk.Entry(self.params_frame, textvariable=self.ra_gaussian_blur, width = 10)

        #Morph kernel size 
        self.ra_labels.append(ttk.Lable(self.params_frame, text="Morphology Kernel Size:"))
        self.ra_widgets['morph_kernel'] = ttk.Entry(self.params_frame, textvariable=self.ra_morph_kernel, width = 10)

        #Min contour area
        self.ra_labels.append(ttk.Lable(self.params_frame, text="Minimum Contour Area:"))
        self.ra_widgets['min_contour'] = ttk.Entry(self.params_frame, textvariable=self.ra_min_contour, width = 10)

        #check boxes 
        self.ra_widgets['Skip Filtering'] = ttk.Checkbutton(self.params_frame, text="Skip Area Filtering",
                                                            variable=self.ra_skip_filtering)
        self.ra_widgets['gentle_cleaning'] = ttk.Checkbutton(self.params_frame, text="Use Gentle Cleaning",
                                                             variable=self.ra_gentle_clearing)
        self.ra_widgets['fill_gaps'] = ttk.Checkbutton(self.params_frame, text="Fill Gaps in Person",
                                                       variable=self.ra_fill_gaps)

        #gaussian mixture params 
        self.gm_widgets = {}
        self.gm_labels = []

        #History 
        self.gm_labels.append(ttk.Label(self.params_frame, text="History:"))
        self.gm_widgets['history'] = ttk.Entry(self.params_frame, textvariable=self.gm_history, width=10)

        #variance threshold 
        self.gm_labels.append(ttk.Label(self.params_frame, text="Variance Threshold:"))
        self.gm_widgets['var_threshold'] = ttk.Entry(self.params_frame, textvariable=self.gm_var_threshold, width=10)

        #Min contour area 
        self.gm_labels.append(ttk.Label(self.params_frame, text="Minimum Contour area:"))
        self.gm_widgets['min_contour'] = ttk.Entry(self.params_frame, textvariable=self.gm_min_contour, width=10)

        #Detect shadows 
        self.gm_widgets['detect_shadows'] = ttk.Checkbutton(self.params_frame, text="Detect Shadows",
                                                            varaible=self.gm_detect_shadows)

    def on_model_change(self):
        #clear all widgets from the grid
        for widget in self.params_frame.winfo_children():
            widget.grid_remove()

        if self.model_type.get() == "running_average":
            row = 0 
            #reposition all of our widgets
            for i, lable in enumerate(self.ra_labels):
                label.grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
                if i == 0: #learning rate 
                    self.ra_widgets['learning_rate'].grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
                elif i == 1: #threshold 
                    self.ra_widgets['threshold'].grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
                elif i == 2: #gaussian 
                    self.ra_widgets['gaussian_blur'].grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
                elif i == 3: #morph kernel 
                    self.ra_widgets['morph_kernel'].grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
                elif i == 4: #min contour 
                    self.ra_widgets['min_contour'].grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
                row += 1 

            #now add the checkboxes back 
            self.ra_widgets['skip_area_filtering'].grid(row=row, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
            row += 1

            self.ra_widgets['gentle_cleaning'].grid(row=row, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
            row += 1

            self.ra_widgets['fill_gaps'].grid(row=row, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
            row += 1
        #Gaussian 
        else:

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



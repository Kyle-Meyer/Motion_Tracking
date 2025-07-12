# Motion Detection and Tracking System

A Python-based motion detection and tracking system that processes sequences of JPEG images to detect moving objects, track their paths, and generate visualizations with bounding boxes and movement trajectories.

## Features

- **Multiple Background Subtraction Models**: Choose from three different background modeling approaches
- **Real-time Motion Detection**: Process image sequences to detect moving objects
- **Object Tracking**: Track detected objects across frames with bounding boxes
- **Path Visualization**: Generate movement trajectories and paths
- **Fully Configurable Parameters**: Customize all detection, tracking, and processing parameters via command line
- **Output Generation**: Save masks, bounding box frames, and path visualizations

## Requirements

- Python 3.7+
- OpenCV (cv2)
- NumPy
- Standard Python libraries (os, glob, argparse, collections, dataclasses)

## Installation

1. Clone or download the project
2. Install required dependencies:

```bash
pip install opencv-python numpy
```

## Project Structure

```
motion-detection/
├── README.md
├── frames/                          # Input JPEG images go here
│   ├── frame_0001.jpg
│   ├── frame_0002.jpg
│   └── ...
└── src/
    ├── main.py                      # Main execution script
    ├── bounding_box/                # Object tracking module
    │   ├── __init__.py
    │   └── bounding_box_tracker.py
    └── mask_generatror/             # Motion detection module
        ├── __init__.py
        ├── motion_detector.py       # Main motion detection class
        ├── enums.py                 # Background model types
        ├── background_subtractor.py # Abstract base class
        ├── running_average_subtractor.py
        ├── gaussian_mixture_subtractor.py
        ├── median_filter.py
        └── mask_processor.py        # Mask cleaning and processing
```

## Usage

### Basic Usage

Navigate to the `src` directory and run:

```bash
cd src
python main.py --input-dir ../frames --output ../results
```

### Command Line Arguments

#### Core Arguments
| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--input-dir` | Yes | - | Directory containing input JPEG images |
| `--pattern` | No | `*.jpg` | Image file pattern (e.g., "*.jpg", "*.png") |
| `--output` | No | - | Output directory for results |
| `--model` | No | `running_average` | Background model: `running_average`, `gaussian_mixture`, `median_filter` |
| `--display` | No | False | Show real-time visualization |
| `--save-bbox-frames` | No | False | Save frames with bounding boxes |

#### Background Subtractor Parameters

**Running Average Model:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--learning-rate` | 0.005 | How quickly the background model adapts (0.001-0.1) |
| `--ra-threshold` | 40.0 | Threshold for foreground detection (10.0-100.0) |

**Gaussian Mixture Model:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--history` | 700 | Number of frames for background modeling (100-1000) |
| `--var-threshold` | 75.0 | Variance threshold for classification (10.0-200.0) |
| `--detect-shadows` | True | Enable shadow detection |
| `--no-detect-shadows` | - | Disable shadow detection |

**Median Filter Model:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--buffer-size` | 20 | Number of frames in median buffer (5-50) |
| `--mf-threshold` | 40.0 | Threshold for foreground detection (10.0-100.0) |

#### Mask Processing Parameters
| Argument | Default | Description |
|----------|---------|-------------|
| `--gaussian-blur-kernel` | 1 1 | Gaussian blur kernel size (width height) |
| `--morphology-kernel-size` | 1 | Size of morphological operations kernel (1-10) |
| `--min-contour-area` | 1000 | Minimum area for contour filtering (100-5000) |
| `--skip-area-filtering` | False | Skip area-based filtering |
| `--use-gentle-cleaning` | False | Use gentle morphological cleaning |
| `--fill-person-gaps` | False | Fill gaps in person detection |

#### Tracking Parameters
| Argument | Default | Description |
|----------|---------|-------------|
| `--min-consecutive-frames` | 30 | Minimum frames for valid tracking (5-100) |
| `--max-distance-threshold` | 75.0 | Maximum distance for track matching (20.0-200.0) |
| `--bbox-padding` | 10 | Padding around bounding boxes (0-50) |

### Background Models

#### 1. Running Average (`running_average`)
- **Best for**: Stable lighting conditions, gradual background changes
- **Pros**: Fast, memory efficient, good for static backgrounds
- **Cons**: Struggles with dynamic backgrounds or lighting changes

**Basic usage:**
```bash
cd src
python main.py --input-dir ../frames --model running_average --output ../results
```

**Fine-tuned for high sensitivity:**
```bash
cd src
python main.py --input-dir ../frames --model running_average \
    --learning-rate 0.001 --ra-threshold 25.0 \
    --morphology-kernel-size 3 --output ../results
```

#### 2. Gaussian Mixture (`gaussian_mixture`)
- **Best for**: Dynamic backgrounds, multiple lighting conditions, outdoor scenes
- **Pros**: Handles complex backgrounds, good shadow detection
- **Cons**: Higher memory usage, more computationally intensive

**Basic usage:**
```bash
cd src
python main.py --input-dir ../frames --model gaussian_mixture --output ../results
```

**Optimized for outdoor scenes:**
```bash
cd src
python main.py --input-dir ../frames --model gaussian_mixture \
    --history 500 --var-threshold 100.0 --detect-shadows \
    --min-contour-area 500 --output ../results
```

#### 3. Median Filter (`median_filter`)
- **Best for**: Noisy environments, intermittent motion, scenes with periodic background changes
- **Pros**: Robust to noise, handles intermittent background objects
- **Cons**: Requires more memory for frame buffer

**Basic usage:**
```bash
cd src
python main.py --input-dir ../frames --model median_filter --output ../results
```

**Configured for noisy environments:**
```bash
cd src
python main.py --input-dir ../frames --model median_filter \
    --buffer-size 30 --mf-threshold 35.0 \
    --gaussian-blur-kernel 5 5 --fill-person-gaps --output ../results
```

## Quick Start Example

1. **Place your JPEG images in the `frames` directory**:
   ```
   frames/
   ├── image_001.jpg
   ├── image_002.jpg
   ├── image_003.jpg
   └── ...
   ```

2. **Run the basic detection**:
   ```bash
   cd src
   python main.py --input-dir ../frames --output ../results
   ```

3. **Run with real-time visualization**:
   ```bash
   cd src
   python main.py --input-dir ../frames --output ../results --display
   ```

4. **Run with full output and custom parameters**:
   ```bash
   cd src
   python main.py --input-dir ../frames --output ../results \
       --save-bbox-frames --min-contour-area 500 --morphology-kernel-size 3
   ```

## Advanced Usage Examples

### High-sensitivity detection for small movements:
```bash
cd src
python main.py --input-dir ../frames --model running_average \
    --learning-rate 0.002 --ra-threshold 20.0 \
    --min-consecutive-frames 15 --min-contour-area 300 \
    --output ../results --save-bbox-frames
```

### Robust detection for outdoor/variable lighting:
```bash
cd src
python main.py --input-dir ../frames --model gaussian_mixture \
    --history 800 --var-threshold 50.0 --detect-shadows \
    --morphology-kernel-size 5 --min-consecutive-frames 40 \
    --output ../results --display
```

### Noise-resistant detection with gap filling:
```bash
cd src
python main.py --input-dir ../frames --model median_filter \
    --buffer-size 25 --mf-threshold 30.0 \
    --gaussian-blur-kernel 7 7 --fill-person-gaps \
    --min-consecutive-frames 20 --output ../results --save-bbox-frames
```

### Fine-tuned tracking parameters:
```bash
cd src
python main.py --input-dir ../frames --model gaussian_mixture \
    --max-distance-threshold 100.0 --bbox-padding 15 \
    --min-consecutive-frames 25 --output ../results
```

## Parameter Tuning Guidelines

### For Different Scenarios:

**Indoor, controlled lighting:**
- Model: `running_average`
- Learning rate: 0.005-0.01
- Threshold: 30.0-50.0
- Morphology kernel: 1-3

**Outdoor, variable lighting:**
- Model: `gaussian_mixture`
- History: 500-1000
- Var threshold: 50.0-100.0
- Enable shadow detection

**Noisy/crowded scenes:**
- Model: `median_filter`
- Buffer size: 20-40
- Larger blur kernels: 5x5 or 7x7
- Enable gap filling

**Small object detection:**
- Reduce `min-contour-area`: 200-500
- Increase sensitivity (lower thresholds)
- Reduce `min-consecutive-frames`: 10-20

**Large object tracking:**
- Increase `min-contour-area`: 2000-5000
- Increase `max-distance-threshold`: 100.0-150.0
- Increase `bbox-padding`: 15-25

## Output Structure

When you specify an output directory, the system generates:

```
results/                      # Your specified output directory
├── masks/                    # Binary masks showing detected motion
│   ├── mask_0000.png
│   ├── mask_0001.png
│   └── ...
├── bounding_boxes/          # Frames with bounding boxes (if --save-bbox-frames)
│   ├── bbox_0000.png
│   ├── bbox_0001.png
│   └── ...
└── paths/                   # Frames with tracking paths
    ├── paths_0000.png
    ├── paths_0001.png
    └── ...
```

## Real-time Visualization

Use the `--display` flag to see real-time processing:

```bash
cd src
python main.py --input-dir ../frames --display
```

**Controls during display:**
- Press `q` to quit early
- Windows shown: Original Image, Generated Mask, Bounding Boxes, Paths & Tracking

## Success Indicators

The system will output processing information:
- **Frame processing progress**: Shows progress every 10 frames
- **Final statistics**: Total frames processed and masks generated
- **60+ frame indicator**: Confirms if sufficient frames were processed for reliable tracking

Example output:
```
Running average across pixel width
Processed 10/120 images
Processed 20/120 images
...
finalize tracking.....
saving paths.....

Processing complete!
Total frames processed: 120
Masks generated: 120
Masks saved to: ../results/masks
60+ frames generated
```

## Troubleshooting

### Common Issues

1. **No images found**: Ensure JPEG files are in the `frames` directory
2. **Import errors**: Make sure you're running from the `src` directory
3. **Low detection accuracy**: Try different background models or adjust sensitivity parameters
4. **Too many false positives**: Increase `min-consecutive-frames` or `min-contour-area`
5. **Missing small objects**: Decrease `min-contour-area` and detection thresholds
6. **Tracking breaks frequently**: Increase `max-distance-threshold` or decrease `min-consecutive-frames`
7. **Memory issues**: Process smaller batches or reduce `buffer-size` for median filter

### Debug Tips

- Start with `--display` flag to see real-time results
- Experiment with different background models for your specific use case
- Use parameter ranges provided in the tables above
- Check that your JPEG files are properly formatted and readable
- Monitor the mask output to verify detection quality before focusing on tracking

### Parameter Optimization Workflow

1. **Start with default parameters** and observe results
2. **Adjust background model parameters** to improve detection quality
3. **Fine-tune mask processing** to clean up noise or fill gaps
4. **Optimize tracking parameters** for your specific object movement patterns
5. **Use `--display`** to get immediate feedback during tuning

## Typical Workflow

1. **Place JPEG images in `frames/` directory**
2. **Navigate to src directory**: `cd src`
3. **Test with display**: `python main.py --input-dir ../frames --display`
4. **Choose optimal model and parameters**: Test different configurations
5. **Generate final output**: `python main.py --input-dir ../frames --output ../results --save-bbox-frames [custom parameters]`

## License

This project is provided as-is for educational and research purposes.

## References
### Running Average Background Subtraction
Herrero, Sergio, and Jesús Bescós. "Background Subtraction Techniques: Systematic Evaluation and Comparative Analysis." In Advanced Concepts for Intelligent Vision Systems, edited by J. Blanc-Talon, W. Philips, D. Popescu, and P. Scheunders, 33-42. Lecture Notes in Computer Science, vol. 5807. Berlin: Springer, 2009. https://doi.org/10.1007/978-3-642-04697-1_4.

Sukhavasi, Susrutha Babu, Suparshya Babu Sukhavasi, Habibulla Khan, and M. Kalpana Chowdary. "Implementation of Running Average Background Subtraction Algorithm in FPGA for Image Processing Applications." International Journal of Computer Applications 73, no. 21 (July 2013): 41-46. https://doi.org/10.5120/13022-0259.

### Gaussian Mixture Model Background Subtraction

Stauffer, Chris, and W. Eric L. Grimson. "Adaptive Background Mixture Models for Real-Time Tracking." In Proceedings of the 1999 IEEE Computer Society Conference on Computer Vision and Pattern Recognition, vol. 2, 246-252. Fort Collins, CO: IEEE, 1999. https://doi.org/10.1109/CVPR.1999.784637.

Zivkovic, Zoran, and Ferdinand van der Heijden. "Improved Adaptive Gaussian Mixture Model for Background Subtraction." In Proceedings of the 17th International Conference on Pattern Recognition, vol. 2, 28-31. Cambridge, UK: IEEE, 2004. https://doi.org/10.1109/ICPR.2004.1333992.

### Median Filter Background Subtraction

McFarlane, N. J. B., and C. P. Schofield. "Segmentation and Tracking of Piglets in Images." Machine Vision and Applications 8, no. 3 (May 1995): 187-193. https://doi.org/10.1007/BF01215814.

Cucchiara, Rita, Costantino Grana, Massimo Piccardi, and Andrea Prati. "Detecting Moving Objects, Ghosts, and Shadows in Video Streams." IEEE Transactions on Pattern Analysis and Machine Intelligence 25, no. 10 (October 2003): 1337-1342. https://doi.org/10.1109/TPAMI.2003.1233909.

### General Background Subtraction Review

Bouwmans, Thierry, and Bertrand Laugraud. "Background Subtraction in Real Applications: Challenges, Current Models and Future Directions." Computer Science Review 35 (February 2020): 100213. https://doi.org/10.1016/j.cosrev.2019.100213.

Yao, Guangle, Tao Lei, Jiandan Zhong, Ping Jiang, and Wenwu Jia. "Comparative Evaluation of Background Subtraction Algorithms in Remote Scene Videos Captured by MWIR Sensors." Sensors 17, no. 9 (September 2017): 1945. https://doi.org/10.3390/s17091945.

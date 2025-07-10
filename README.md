# Motion Detection and Tracking System

A Python-based motion detection and tracking system that processes sequences of JPEG images to detect moving objects, track their paths, and generate visualizations with bounding boxes and movement trajectories.

## Features

- **Multiple Background Subtraction Models**: Choose from three different background modeling approaches
- **Real-time Motion Detection**: Process image sequences to detect moving objects
- **Object Tracking**: Track detected objects across frames with bounding boxes
- **Path Visualization**: Generate movement trajectories and paths
- **Configurable Parameters**: Customize detection sensitivity and tracking behavior
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

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--input-dir` | Yes | - | Directory containing input JPEG images |
| `--pattern` | No | `*.jpg` | Image file pattern (typically leave as default) |
| `--output` | No | - | Output directory for results |
| `--model` | No | `running_average` | Background model type |
| `--display` | No | False | Show real-time visualization |
| `--min-consecutive-frames` | No | 30 | Minimum frames for valid tracking |
| `--save-bbox-frames` | No | False | Save frames with bounding boxes |

### Background Models

#### 1. Running Average (`running_average`)
- **Best for**: Stable lighting conditions, gradual background changes
- **Parameters**:
  - `learning_rate`: 0.005 (how quickly background adapts)
  - `threshold`: 40.0 (sensitivity to changes)

```bash
cd src
python main.py --input-dir ../frames --model running_average --output ../results
```

#### 2. Gaussian Mixture (`gaussian_mixture`)
- **Best for**: Dynamic backgrounds, multiple lighting conditions
- **Parameters**:
  - `history`: 700 (number of frames for background modeling)
  - `var_threshold`: 75.0 (threshold for foreground detection)
  - `detect_shadows`: True (shadow detection)

```bash
cd src
python main.py --input-dir ../frames --model gaussian_mixture --output ../results
```

#### 3. Median Filter (`median_filter`)
- **Best for**: Noisy environments, intermittent motion
- **Parameters**:
  - `buffer_size`: 20 (frames to keep in buffer)
  - `threshold`: 40.0 (threshold for foreground detection)

```bash
cd src
python main.py --input-dir ../frames --model median_filter --output ../results
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

4. **Run with full output (including bounding box frames)**:
   ```bash
   cd src
   python main.py --input-dir ../frames --output ../results --save-bbox-frames
   ```

## Advanced Usage Examples

### High-sensitivity detection for small movements:
```bash
cd src
python main.py --input-dir ../frames --model running_average --min-consecutive-frames 15 --output ../results --save-bbox-frames
```

### Robust detection for outdoor/variable lighting:
```bash
cd src
python main.py --input-dir ../frames --model gaussian_mixture --min-consecutive-frames 40 --output ../results --display
```

### Noise-resistant detection:
```bash
cd src
python main.py --input-dir ../frames --model median_filter --min-consecutive-frames 20 --output ../results --save-bbox-frames
```

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

## Configuration Guidelines

### Running Average
```bash
cd src
python main.py --input-dir ../frames --model running_average --output /output 
```

### Gaussian Mixture
```bash
cd src
python main.py --input-dir ../frames --model gaussian_mixture --output /output 
```

### Median Threshold
```bash
cd src
python main.py --input-dir ../frames --model median_filter --output /output 
```

## Tracking Parameters

The system includes configurable tracking parameters:

- **`min_contour_area`**: 1000 (minimum area for object detection)
- **`max_distance_threshold`**: 75.0 (maximum distance for track association)
- **`min_consecutive_frames`**: 30 (minimum frames for valid track)
- **`padding`**: 10 (bounding box padding)

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
3. **Low detection accuracy**: Try different background models or adjust parameters
4. **Too many false positives**: Increase `min_consecutive_frames` parameter
5. **Memory issues**: Process smaller batches of images

### Debug Tips

- Start with `--display` flag to see real-time results
- Try different background models for your specific use case
- Check that your JPEG files are properly formatted and readable

## Typical Workflow

1. **Place JPEG images in `frames/` directory**
2. **Navigate to src directory**: `cd src`
3. **Test with display**: `python main.py --input-dir ../frames --display`
4. **Choose optimal model**: Test different background models
5. **Generate final output**: `python main.py --input-dir ../frames --output ../results --save-bbox-frames`

## License

This project is provided as-is for educational and research purposes.

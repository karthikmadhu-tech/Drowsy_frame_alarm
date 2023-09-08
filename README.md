# Drowsy Frame Detection

Drowsy Frame is a Python project that uses computer vision and facial landmarks detection to monitor and detect drowsiness in a person's eyes while they are in front of a webcam. When the program detects signs of drowsiness, it triggers an alert to notify the user.

## Prerequisites

Before you can run the Drowsy Frame project, make sure you have the following dependencies installed:

- Python 3.10
- OpenCV (cv2)
- dlib
- imutils
- scipy
- pygame

You can install these libraries using pip:

```bash
pip install opencv-python dlib imutils scipy pygame
```

## Usage

1. Clone or download this repository to your local machine.

2. Navigate to the project directory.

3. Ensure you have a sound file named "music.wav" in the same directory as your project code. This sound file will be played as an alert when drowsiness is detected.

4. Run the Python script `drowsy_frame.py`:

```bash
python drowsy_frame.py
```

5. The webcam feed will open, and the program will monitor your eyes for signs of drowsiness.

6. If drowsiness is detected, an alert message will be displayed on the screen, and an alert sound will be played.

7. To exit the program, press the 'q' key.

## Configuration

You can adjust the following parameters in the `drowsy_frame.py` script to customize the drowsiness detection:

- `thresh`: The threshold for detecting drowsiness. Lower values make it more sensitive.

- `frame_check`: The number of consecutive frames below the threshold required to trigger an alert. Increase this value to reduce false alarms.

## Credits

This project uses the following libraries and resources:

- OpenCV (cv2) for webcam access and image processing.
- dlib for facial landmark detection.
- imutils for resizing and processing frames efficiently.
- scipy for distance calculations.
- pygame for playing the alert sound.

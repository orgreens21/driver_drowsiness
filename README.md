# Driver Monitoring System

## Project Description
This project is designed to monitor driver alertness using real-time video analysis. It combines eye status detection and head posture estimation to alert if a driver appears drowsy or distracted. The system uses computer vision techniques to detect closed eyes and non-forward head directions, signaling potential driver fatigue or distraction.

## Features
- Real-time eye status monitoring.
- Head direction detection.
- Audio and visual alerts to prevent driver drowsiness and distraction.

## Prerequisites
To run this project, you'll need:
- Python 3.11 or later.
- OpenCV for image processing.
- imutils for image manipulation.
- dlib
- winsound for alert sounds (Windows only).


## Installation

Follow these steps to get your development environment running:

1. **Clone the Repository**
   Clone this repository to your local machine using the following command:

   ```bash
   git clone https://github.com/orgreens21/driver_drowsiness.git
   cd driver_drowsiness

2. **Install Required Libraries**
   Install all the necessary libraries using the provided requirements.txt:

   pip install -r requirements.txt

3. **Download Required Data Files**
- Download the shape_predictor_68_face_landmarks.dat file:
  Go to http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 and download the .bz2 file.

- Extract the file using the following command, and move the /utils/calibration folder

  ```bash
  mv /path/to/shape_predictor_68_face_landmarks.dat /path/to/your/project/utils/calibration/

## Running the Project

To run the project, navigate to your project directory and execute the main script:

   ```bash
   python main.py

Enjoy our project, and don't fall asleep running it :)

---
title: AttendNet
emoji: 🛡️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
---

# Smart Face Attendance System

A Python-based face recognition attendance system using FaceNet deep learning model and OpenCV for real-time face detection and recognition.

## Features

- **Real-time Face Recognition**: Uses FaceNet model for high-accuracy face recognition
- **Web-based Registration**: Flask web interface for student registration with photo upload or camera capture
- **Automated Attendance Tracking**: Records attendance in Excel format with date-wise columns
- **Multi-face Detection**: Supports detecting multiple faces simultaneously
- **Smart Presence Detection**: Requires 20 seconds of continuous face presence to mark attendance
- **Accuracy Evaluation**: Built-in evaluation scripts with visualization charts

## Project Structure

```
.
├── attendance/
│   └── attendance.xlsx          # Attendance records (auto-generated, NOT committed)
├── dataset/
│   └── <USN>/img1.jpg...        # Student face images (PRIVATE, NOT committed)
├── templates/
│   ├── attendance.html          # Attendance dashboard UI
│   └── register.html            # Student registration UI
├── accuracy_evaluation.py       # Accuracy evaluation script
├── attendance.py                # Optional alternative (face_recognition)
├── face_attendance_run.py       # Main attendance system (VGG-Face model)
├── embeddings_cache.pkl         # Auto-generated cache (NOT committed)
├── login.py                     # Flask web server for registration
└── requirements.txt             # Python dependencies
```

## Prerequisites

- Python 3.10+
- Webcam/Camera
- Windows/Linux/MacOS

## Installation

1. Clone the repo
2. Create a virtual environment

```bash
python -m venv .venv
```

3. Activate it

Windows (PowerShell):

```bash
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

4. Install dependencies

```bash
python -m pip install -U pip
python -m pip install -r requirements.txt
```

## Usage

### 1. Student Registration

Start the Flask web server for student registration:

```bash
python login.py
```

Open your browser and navigate to `http://localhost:5000`

**Registration Options:**
- **Upload Photos**: Upload clear face photos (the app can accept multiple; it will guide you)
- **Camera Capture**: Press `C` to capture frames from webcam

The system will create a folder in `dataset/` with the student's USN and store their face images.

### 2. Mark Attendance

Run the main attendance system:

```bash
python face_attendance_run.py
```

**How it works:**
- The camera opens and detects faces in real-time
- Recognized students are labeled with their USN in green
- Keep your face in front of the camera for a few seconds to be marked "Present"
- Press `ESC` to finish early
- Attendance is automatically saved to `attendance/attendance.xlsx`

### 3. Alternative Attendance System

There's also a simpler version using the `face_recognition` library:

```bash
python attendance.py
```

This version runs for 20 seconds and marks attendance based on face distance matching.

### 4. Accuracy Evaluation

Evaluate the model's performance:

```bash
python accuracy_evaluation.py
```

This generates charts under `evaluation_results/` (ignored by git).

## Attendance Excel Format

The attendance file (`attendance/attendance.xlsx`) contains:

| Reg_No | Gmail | Phone | 2025-02-24 | 2025-02-25 | ... |
|--------|-------|-------|------------|------------|-----|
| 23BTRCL017 | student@email.com | 9876543210 | Present | Absent | ... |
| 23BTRCL046 | student2@email.com | 9876543211 | Present | Present | ... |

- New date columns are automatically added each day
- Default status is "Absent" for all students
- Status changes to "Present" when face is recognized

## Technical Details

### Face Recognition Models

1. **VGG-Face** (Primary): Deep learning model with ~89% accuracy
   - Used in `face_attendance_run.py`
   - Cosine similarity matching with 0.60 threshold

2. **face_recognition** (Alternative): dlib-based HOG/CNN model
   - Used in `attendance.py`
   - Euclidean distance matching with 0.38 threshold

### Detection Parameters

- **Detection Window**: 30 seconds
- **Required Presence**: 20 seconds for "Present" status
- **Frame Processing**: Every 5th frame for performance optimization
- **Face Detection**: Haar Cascade classifier (min size: 80x80)

## Troubleshooting

### TensorFlow / protobuf import error
If you see an error like `cannot import name 'runtime_version' from 'google.protobuf'`, upgrade protobuf:

```bash
python -m pip install -U --force-reinstall "protobuf>=5.26.0"
```

### Camera Not Opening
- Check if another application is using the camera
- Try changing the camera index in `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`

### Face Not Recognized
- Ensure good lighting conditions
- Face should be front-facing
- Check if student is registered in the dataset folder
- Try re-registering with clearer photos

### Model Loading Issues
- First run may take time to download VGG-Face model weights
- Ensure stable internet connection for initial setup

### Excel File Locked
- Close the attendance.xlsx file if it's open in Excel
- The script needs write access to update attendance

## Security Notes

- Face data is stored locally in the `dataset/` folder
- No data is sent to external servers
- Keep the dataset folder secure and backed up

## License

MIT License (see `LICENSE`).

## Honest Assessment

### Strengths
1. **Working Implementation**: The system successfully recognizes faces and marks attendance
2. **Dual Approach**: Two different face recognition libraries provide flexibility
3. **Smart Presence Logic**: 20-second continuous detection prevents quick spoofing
4. **Web Interface**: Clean registration UI with photo upload option
5. **Excel Integration**: Easy-to-use attendance tracking with date-wise columns

### Weaknesses & Limitations

1. **Small Dataset**: Only 2 students with 3 images each (6 total). Real-world use needs 30+ images minimum for reliable metrics.

2. **Hardcoded Thresholds**: 
   - 0.60 similarity threshold in VGG-Face may need tuning for different lighting/angles
   - 0.38 distance threshold in face_recognition is strict and may cause false negatives

3. **No Anti-Spoofing**: System can be fooled by:
   - Printed photos of registered students
   - Phone screens showing student photos
   - Video playback of registered faces

4. **Performance Issues**:
   - DeepFace represent() is called every 5th frame - still CPU intensive
   - No GPU acceleration support
   - Frame skipping causes choppy UI experience

5. **Code Quality Issues**:
   - Bare `except:` blocks hide errors (lines 63-64, 172-174)
   - No input validation on registration form
   - Hardcoded paths and magic numbers throughout
   - No logging system - only print statements

6. **Security Concerns**:
   - No authentication on web interface
   - Excel file has no access control
   - Face data stored as plain images (not encrypted)
   - No audit trail for attendance changes

7. **Scalability Problems**:
   - Linear search through all embeddings (O(n) complexity)
   - Will slow down significantly with 50+ students
   - No database - flat Excel file won't scale

8. **Robustness Issues**:
   - Haar Cascade is outdated (2010s technology)
   - Poor performance in low light, side angles, or with glasses/masks
   - No fallback if VGG-Face model download fails

### Recommendations for Production Use

1. **Add Liveness Detection**: Use blink detection or depth sensing to prevent photo attacks
2. **Upgrade Face Detector**: Replace Haar Cascade with MTCNN or RetinaFace
3. **Implement Database**: Use SQLite/PostgreSQL instead of Excel
4. **Add Authentication**: Login system for administrators
5. **Improve Error Handling**: Replace bare except blocks with specific exception handling
6. **Add Logging**: Implement proper logging instead of print statements
7. **Use Vector Database**: FAISS or Pinecone for fast similarity search with large datasets
8. **Add Unit Tests**: Currently no test coverage

## Credits

- VGG-Face model via DeepFace library
- OpenCV for computer vision operations
- Flask for web interface

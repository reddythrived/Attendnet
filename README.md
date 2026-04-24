AttendNet AI
Real-Time Face Recognition Attendance System

🔗 Live Deployment:
https://reddythrived-attendnet.hf.space

📖 Abstract

AttendNet AI is a real-time, AI-powered attendance management system that automates the process of identifying individuals and recording attendance using facial recognition. The system eliminates manual errors, reduces time consumption, and enhances reliability in institutional and organizational environments.

It leverages dlib-based 128-dimensional facial embeddings, combined with multi-face detection and temporal validation, to achieve robust and accurate recognition (~96%) under real-world conditions.

🎯 Objectives
Automate attendance recording using AI
Ensure high accuracy in face recognition
Handle multiple users simultaneously
Minimize false positives through validation techniques
Provide a scalable and accessible web-based solution
🧠 System Design

The system follows a modular pipeline:

Frame Acquisition – Captures real-time video input
Face Detection – Identifies faces using OpenCV
Feature Extraction – Generates 128D embeddings using dlib
Face Matching – Compares embeddings with stored dataset
Temporal Validation – Confirms identity across multiple frames
Attendance Logging – Stores verified entries in database
🏗️ Architecture
Video Stream → Face Detection → Feature Extraction → Face Recognition  
→ Temporal Validation → Attendance Storage → Web Interface
⚙️ Tech Stack

Frontend

HTML5
CSS3

Backend

Python
Flask

AI / ML

dlib (128D face embeddings)
OpenCV
NumPy

Deployment

Hugging Face Spaces
📂 Dataset
Custom dataset of registered individuals
Preprocessing steps include:
Face alignment
Image normalization
Noise reduction
🚀 Key Features
Real-time face recognition
Multi-face detection in a single frame
Temporal validation for improved reliability
Web-based interface (device-independent access)
Cloud deployment for scalability
High accuracy (~96%)
📊 Performance Metrics
Metric	Value
Accuracy	~96%
Processing	Real-time
Multi-user Support	Yes
False Positives	Reduced via temporal filtering
⚠️ Challenges and Solutions
Challenge	Approach
False recognition	Temporal validation across frames
Lighting variations	Robust embedding generation
Multiple faces	Parallel face detection
Processing latency	Frame optimization
🌐 Deployment Details

The application is deployed on a cloud platform using Hugging Face Spaces, ensuring:

Remote accessibility
Zero local installation requirement
Easy updates and maintenance

🔗 Access the system:
https://reddythrived-attendnet.hf.space

🔮 Future Scope
Integration with mobile platforms
Advanced anti-spoofing mechanisms
Cloud database integration (Firebase/AWS)
Real-time analytics dashboard
Upgrade to advanced models like FaceNet or DeepFace
👨‍💻 Author

K.P. Thrived Reddy
B.Tech – Artificial Intelligence & Machine Learning

# Face Recognition System

A Siamese Neural Network-based face recognition system implemented in Python using TensorFlow and OpenCV.

## Overview

This project implements a facial recognition system using a Siamese Neural Network architecture. The system can:
- Capture and process facial images
- Perform data augmentation
- Train a model to recognize faces
- Verify identities in real-time

## Requirements

tensorflow==2.0.0
opencv-python
numpy

project structure
├── data/
│   ├── anchor/       # Anchor images for training
│   ├── positive/     # Positive samples
│   ├── negative/     # Negative samples
├── face_recognition.py
└── requirements.txt


Features
Image Collection: Captures facial images through webcam

Data Augmentation: Implements various augmentation techniques including:
  Random brightness adjustment
  Random contrast
  Random horizontal flips
  JPEG quality variation
  Random saturation
  
Model Architecture: Uses Siamese Network with:
  Convolutional layers for feature extraction
  L1 distance layer for similarity measurement
  Binary classification output
  
Usage
1-Install dependencies:
  pip install -r requirements.txt

2-Run the main script:
  python face_recognition.py

Controls
  Press 'a' to capture anchor images
  Press 'p' to capture positive images
  Press 'q' to quit
  
Model Performance
The model's performance is evaluated using:
  Precision
  Recall
  Binary Cross-Entropy Loss
  
Verification System
The system includes a verification pipeline that:
  Captures validation images
  Processes input images in real-time
  Compares against stored verification images
  Uses detection and verification thresholds for accurate recognition

  


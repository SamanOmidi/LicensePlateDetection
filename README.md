# Persian License Plate Detection & Recognition

This project implements a Persian license plate detection and character recognition system using two different approaches and compares their performance.

## Overview
The goal of this project is to detect vehicle license plates and recognize Persian characters from images. Two fundamentally different pipelines were implemented and evaluated.

## Implemented Approaches

### 1. Classical Computer Vision + CNN
- Edge detection and image processing techniques to localize license plates
- A CNN-based model for Persian character recognition after plate extraction

### 2. YOLOv8-Based End-to-End Model
- A YOLOv8 model trained to directly detect license plates and characters
- Single-stage detection without manual feature engineering

## Results
The YOLOv8-based approach significantly outperformed the classical edge detection + CNN pipeline in terms of accuracy and robustness, especially under varying lighting conditions and complex backgrounds.

## Tech Stack
- Python
- OpenCV
- Keras
- YOLOv8
- NumPy


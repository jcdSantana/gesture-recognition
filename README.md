# Gesture Recognition Project

This project aims to recognize static hand gestures using computer vision techniques. It leverages classical image processing methods (without deep learning frameworks like TensorFlow or PyTorch) to binarize images, extract features from hand contours, and classify gestures.

## Table of Contents
- [Project Overview](#project-overview)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Dataset](#dataset)
- [Usage](#usage)
  - [Running the Project](#running-the-project)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Project Overview

The goal of this project is to develop a simple yet effective static hand gesture recognition system using classical computer vision techniques. We utilize libraries like OpenCV for image processing, and basic machine learning models (e.g., KNN, SVM) for classification.

### Features:
- Binarization of hand gesture images to separate the hand from the background.
- Contour detection to extract the shape of the hand.
- Convex Hull and Convexity Defects analysis to count fingers.
- Gesture classification using traditional machine learning models.
- Option to use the model for real-time hand gesture recognition using a webcam.

## Getting Started

### Installation

1. **Clone the Repository**
   Start by cloning this repository to your local machine:
   ```bash
   git clone https://github.com/your-username/gesture-recognition.git
   cd gesture-recognition

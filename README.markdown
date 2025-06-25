# Self-Driving Traffic Sign Detection

## Project Overview

This project focuses on developing and evaluating deep learning convolutional neural network (CNN) models to detect traffic light signs using Udacity's Self-Driving dataset. The dataset includes images of pedestrians, bikers, cars, and traffic lights, with the analysis specifically targeting traffic light images (Red, Yellow, Green). The goal is to compare various CNN architectures, including a pre-trained ResNet50 model and custom shallow CNNs, to optimize accuracy and efficiency for traffic light detection.

## Dataset

The dataset is sourced from Udacity's Self-Driving dataset, consisting of 15,000 images with 97,942 labels across 11 classes, including 1,720 null examples. This project focuses on three traffic light classes:
- **trafficLight-Red**: 13,673 samples
- **trafficLight-Yellow**: 541 samples (upsampled to 13,673 to balance the dataset)
- **trafficLight-Green**: 10,838 samples

### Preprocessing
- Extracted traffic light images to improve computational efficiency.
- Resized images to 224x224 pixels for standardization.
- Normalized pixel values to the 0-1 range for better generalization.
- Upsampled the underrepresented Yellow class to match the largest class size (Red).
- Split data into 80% training and 20% testing sets, with stratified sampling to maintain class balance.

## Models

Two main models were developed, each with three variations:
1. **ResNet50 (Pre-trained)**
   - **Variation 1**: Used GlobalAveragePooling2D, followed by Dense layers (1024 units and 3 units for classification).
   - Other variations are not detailed in the provided notebook but were tested with different configurations.
   - Results: Overfitting observed due to model complexity and limited training buffer (1,000 images).

2. **Custom Shallow CNN**
   - **Variation 3**:
     - Architecture: Two Conv2D layers (32 and 64 filters), two MaxPooling2D layers, Flatten, Dense (128 units), Dropout (0.6), Dense (64 units), Dropout (0.4), and Dense (3 units, softmax).
     - Removed BatchNormalization to reduce complexity.
     - Added an additional Dropout layer (0.4) and increased the first Dropout to 0.6 to combat overfitting.
     - Used EarlyStopping to monitor validation accuracy and restore the best weights.
   - Results: Achieved balanced performance with ~85% training and validation accuracy, avoiding overfitting.

## Key Findings

- **ResNet50**: Overfitted due to its complexity, which is better suited for larger datasets. The small training buffer (1,000 images) limited its effectiveness.
- **Custom CNN**: The simpler architecture of the custom CNN was better suited for the dataset size. Variation 3, with increased dropout and early stopping, achieved the best balance, with training and validation accuracies aligning at ~85%.
- **Lesson**: Starting with a simpler model and gradually increasing complexity is more effective for smaller datasets.

## Installation

To run this project, ensure you have the following dependencies installed:

```bash
pip install tensorflow pandas numpy scikit-learn
```

Additionally, download the Udacity Self-Driving dataset and place it in the `data/export` directory, with the annotations file `_annotations.csv` in the same folder.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/self-driving-traffic-sign-detection.git
   cd self-driving-traffic-sign-detection
   ```

2. Set up the dataset:
   - Unzip the dataset to `data/export`.
   - Ensure `_annotations.csv` is in the `data/export` directory.

3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook SelfDrivingTrafficSignDetection.ipynb
   ```

4. Follow the notebook to preprocess the data, train the models, and evaluate their performance.

## Next Steps

- Explore regularization techniques (e.g., L1 Lasso, L2 Ridge) to further improve the custom CNN.
- Investigate other pre-trained models like MobileNet, which is designed for smaller datasets.
- Increase the training buffer size to potentially improve ResNet50 performance.
- Experiment with data augmentation techniques to enhance model robustness against variations in lighting and occlusions.
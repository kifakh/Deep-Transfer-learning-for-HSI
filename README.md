# Hyperspectral Image Classification using Deep Learning and Machine Learning

This repository contains Python code for the classification of Hyperspectral Images (HSI) using deep learning techniques such as transfer learning with ResNet, Xception, etc., and traditional machine learning methods like Support Vector Machine (SVM) and Random Forest (RF). The project applies various techniques including Principal Component Analysis (PCA), 5-fold cross-validation, and fine-tuning. The dataset used for experimentation is the Pavia University dataset.

## Dataset

The Pavia University dataset consists of hyperspectral images captured by remote sensing devices. It contains high-dimensional spectral data with spatial information. The images are labeled with different classes representing different land cover types.

## Deep Learning Models

The following transfer learning models are implemented for classification:

- **ResNet**: A deep residual neural network architecture known for its ability to train very deep networks effectively.
- **Xception**: A deep convolutional neural network architecture with a large number of layers designed for image classification tasks.
- **EffiecientNet**

## Machine Learning Methods

The following traditional machine learning methods are employed for classification:

- **Support Vector Machine (SVM)**: A supervised learning algorithm used for classification tasks by finding the hyperplane that best divides a dataset into classes.
- **Random Forest (RF)**: An ensemble learning method that builds multiple decision trees during training and outputs the mode of the classes for classification.

## Techniques Applied

The project applies the following techniques to enhance classification performance:

- **Principal Component Analysis (PCA)**: Dimensionality reduction technique used to reduce the number of features while preserving the variance in the data.
- **5-Fold Cross-Validation**: Data is split into five folds, and the model is trained and evaluated on each fold separately to assess its generalization performance.
- **Fine-Tuning**: Pre-trained deep learning models are fine-tuned on the Pavia University dataset to adapt them to the specific task of HSI classification.

## Usage

1. Clone this repository to your local machine.
2. Download the Pavia University dataset and place it in the appropriate directory.
3. Open the Python scripts for the desired model or method
4. Execute the script to train and evaluate the model or method on the Pavia University dataset.

## Results
The classification results, including accuracy, precision, recall, F1-score, and confusion matrices, are generated and stored in the `results` directory for each model or method applied.

## Conclusion

This project demonstrates the application of deep learning and machine learning techniques for the classification of Hyperspectral Images (HSI) using Python. The use of transfer learning, PCA, and fine-tuning contributes to improved classification accuracy and robustness in comparision with traditional machine learning methods.

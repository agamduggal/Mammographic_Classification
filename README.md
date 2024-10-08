# Mammographic Analysis for Breast Cancer Detection

This repository contains two comprehensive projects focused on detecting breast cancer through mammographic data analysis. Leveraging machine learning and deep learning techniques, I’ve applied a variety of models and approaches to achieve robust predictions.

## Project 1: Machine Learning Models for Cancer Prediction

In this project, I have been provided with a labeled dataset containing mammographic masses with relevant features. The goal was to apply different machine learning models to predict whether a mass is benign or malignant.

### Key Steps:
1. **Data Preprocessing**:
   - Cleaned and normalized the dataset to ensure high-quality input.
   - Handled missing values and performed feature selection for optimal results.

2. **Modeling**:
   - Applied several models including:
     - Logistic Regression
     - Support Vector Machine (SVM)
     - Random Forest
     - K-Nearest Neighbors (KNN)

3. **Evaluation**:
   - Identified the best-performing model to maximize prediction accuracy.

### Results:
- Achieved high accuracy with almost all the models except the **[Decision Trees]**, demonstrating reliable performance for cancer prediction based on mammographic data.

## Project 2: Deep Learning on Mammogram Images

The second project involves detecting cancerous masses using actual mammographic images. Here, I’ve employed Convolutional Neural Networks (CNNs) to process the images and determine whether a mass is benign or malignant.

### Key Steps:
1. **Image Processing**:
   - Preprocessed mammogram images by resizing, normalizing, and augmenting to improve model generalization.

2. **Neural Network Architecture**:
   - Developed a CNN architecture tailored to detect features from mammographic images.
   - Integrated layers such as convolution, pooling, and fully connected layers to extract and classify features.

3. **Training & Optimization**:
   - Utilized techniques like batch normalization and dropout to prevent overfitting.
   - Trained the model using a labeled dataset and optimized it with Adam optimizer.

4. **Evaluation**:
   - Assessed the performance using metrics like accuracy, AUC-ROC, and confusion matrix.
   - Fine-tuned hyperparameters to achieve optimal model performance.

### Results:
- Achieved high accuracy in identifying malignant masses, with the neural network effectively learning from the image features.

## Conclusion

These projects showcase the power of both traditional machine learning models and deep learning architectures in detecting breast cancer from mammographic data. The combination of data-driven insights and image-based analysis provides a comprehensive approach to tackling this critical healthcare challenge.



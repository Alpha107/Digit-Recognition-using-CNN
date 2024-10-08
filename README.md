# Digit Recognition using Convolutional Neural Networks (CNN)

This project is a digit recognition system implemented using Convolutional Neural Networks (CNN) and the MNIST dataset. It predicts handwritten digits (0-9) and visualizes the model performance through a confusion matrix heatmap and learning curves.

## Introduction

This project aims to develop a robust deep learning model that can recognize digits from the famous MNIST dataset using CNNs. The model is trained using TensorFlow and Keras libraries, and achieves high accuracy on the test set. Performance metrics include accuracy and a confusion matrix.

## Dataset

The MNIST dataset is used, which contains 70,000 images of handwritten digits (0-9) in grayscale. The dataset is split as follows:

• Training set: 60,000 images

• Test set: 10,000 images

Each image is a 28x28 pixel grayscale image. The data is loaded directly from (tensorflow.keras.datasets.)

## Model Architecture

The Convolutional Neural Network (CNN) model has the following architecture:

• Conv2D Layer: 32 filters, kernel size 3x3, activation function ReLU

• Conv2D Layer: 64 filters, kernel size 3x3, activation function ReLU

• Conv2D Layer: 128 filters, kernel size 3x3, activation function ReLU

• MaxPooling2D Layer: Pool size 2x2

• Dropout Layer: Dropout rate of 0.5 to prevent overfitting

• Flatten Layer: Converts the 2D matrix data into a 1D vector

• Dense Layer: 128 neurons, activation function ReLU

• Output Layer: 10 neurons, activation function Softmax (for multi-class classification)

The model is compiled using the Adam optimizer and the sparse_categorical_crossentropy loss function, with accuracy as the evaluation metric.

## Training and Evaluation

The model is trained on the MNIST training set for 10 epochs with a batch size of 128. During training, validation is done on the test set.

After training, the test set is used to make predictions, and the accuracy and confusion matrix are computed to assess the model's performance.

## Visualization

1. Confusion Matrix Heatmap
A heatmap is used to visualize the confusion matrix, which shows how well the model predicted each class (digit). This allows for a deeper analysis of model performance.

![ConfusionMatrix](https://github.com/user-attachments/assets/d1578cbf-8c63-41ed-8edb-e378f54069c8)

2. Learning Curves
The learning curves for training and validation accuracy and loss are plotted to monitor how the model's performance evolves over epochs.

![LearningCurve](https://github.com/user-attachments/assets/5ed07a27-5c7b-4471-bd05-99498205bd71)

Results

• Accuracy: 99.27%

## Conclusion
This project successfully demonstrates how Convolutional Neural Networks (CNNs) can be applied to digit recognition tasks using the MNIST dataset. The model achieves high accuracy and performs well in classifying handwritten digits. Key performance metrics, such as the accuracy score and confusion matrix, indicate that the CNN is highly effective at recognizing digit patterns from images.

Additionally, the use of visualization techniques like the confusion matrix heatmap and learning curves provides insights into the model’s behavior, highlighting areas for further improvement and tuning. This project serves as a foundational example of using deep learning techniques for image classification tasks, and can be further extended to more complex datasets or deeper architectures.

Future work could include experimenting with different architectures, applying data augmentation, or implementing hyperparameter tuning to further enhance performance.

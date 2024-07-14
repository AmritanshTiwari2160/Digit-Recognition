# Digit Recognition Using Neural Networks on the MNIST Dataset
This project focuses on recognizing handwritten digits using a neural network trained on the MNIST dataset. The MNIST dataset is a widely-used benchmark for this task, containing 60,000 training images and 10,000 testing images of handwritten digits from 0 to 9.

**Key Highlights:**
1. Dataset Overview:<br/>
Training Set: 60,000 images.<br/>
Testing Set: 10,000 images.<br/>
Images are grayscale and have a resolution of 28x28 pixels.

2. Data Preprocessing:
Loaded the MNIST dataset directly into NumPy arrays, which simplifies the processing.
Visualized sample images from the dataset to understand the data structure.
Scaled the pixel values to the range [0, 1] for better model performance.

3. Neural Network Architecture:<br/>
Model Type: Sequential<br/>
Layers:<br/>
Flatten Layer: Converts each 28x28 image into a 784-dimensional vector.<br/>
Dense Layer 1: 50 neurons, ReLU activation.<br/>
Dense Layer 2: 50 neurons, ReLU activation.<br/>
Output Layer: 10 neurons, Sigmoid activation.

4. Model Compilation:<br/>
Optimizer: Adam<br/>
Loss Function: Sparse Categorical Crossentropy
Metrics: Accuracy

5. Training the Model:
Trained the model for 10 epochs, achieving high accuracy on the training set.
Achieved an impressive training accuracy of 96%.

6. Model Evaluation:
Evaluated the model on the test dataset, achieving an accuracy of 96.6%.
Visualized the predictions and compared them with actual labels.

7. Prediction Example:
Visualized a test image and displayed its predicted label.
Generated predictions for all test images and calculated the confusion matrix to analyze the model's performance.

8. Image Saving and Prediction:
Saved training images as PNG files for further analysis.
Implemented a functionality to predict the label of a custom input image using the trained model.

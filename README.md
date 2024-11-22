# Image Classification Using Multilayer Perceptron (MLP)

## Project Overview
This project involves building a Multilayer Perceptron (MLP) model for classifying images from the MNIST dataset. The model is designed to recognize handwritten digits (0-9) with high accuracy by training on 60,000 training samples and evaluating on 10,000 test samples. The MLP achieved an impressive accuracy of 97% on the test set.

## Table of Contents
- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)



ruby
Copy code

## Installation
To run this project, ensure you have Python installed and the required dependencies. Use the following command to install the dependencies:

```bash
pip install -r requirements.txt
Key Libraries
numpy
tensorflow
scikit-learn
matplotlib
Data Preparation
The project uses the MNIST dataset, a widely-used benchmark dataset for image classification tasks. The dataset is loaded and preprocessed as follows:

Normalization: Pixel values are scaled to the range [0, 1].
Flattening: 28x28 images are reshaped to 1D arrays of length 784.
Model Architecture
The MLP model consists of:

Input Layer: 784 nodes (one for each pixel).
Hidden Layers:
First hidden layer: 128 neurons with ReLU activation.
Second hidden layer: 64 neurons with ReLU activation.
Third hidden layer: 32 neurons with ReLU activation.
Output Layer: 10 neurons (softmax activation) for multi-class classification.
python
Copy code
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax')
])
Training
The model was trained using:

Optimizer: Adam
Loss Function: Sparse Categorical Crossentropy
Metrics: Accuracy
Epochs: 10
Batch Size: 32
python
Copy code
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
Evaluation
The model was evaluated on the test set, achieving an accuracy of 97%. The confusion matrix and classification report provided insight into per-class performance.

Key Metrics:
Overall Accuracy: 97%
Precision, Recall, F1-Score: Detailed per-class metrics available in the report.
Results
The model demonstrates strong performance with high precision and recall for most classes. Some confusion was observed between specific digits, such as '3' and '5'.

Confusion Matrix (Sample):
css
Copy code
[[ 974    1    1    0    0    0    2    1    1    0]
 [   0 1115    3    1    0    1    7    1    7    0]
 ...
]
Classification Report:
Class	Precision	Recall	F1-Score	Support
0	0.97	0.99	0.98	980
1	0.99	0.98	0.99	1135
...	...	...	...	...
Future Work
Hyperparameter Tuning: Experiment with different learning rates, optimizers, and architectures.
Data Augmentation: Apply data augmentation to increase robustness.
Ensemble Methods: Combine models to further improve accuracy.
Contributing
Contributions are welcome! Please submit a pull request or report any issues you find.

License
This project is licensed under the MIT License. See the LICENSE file for more details.

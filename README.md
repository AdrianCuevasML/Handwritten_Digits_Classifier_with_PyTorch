# Handwritten Digits Classifier with PyTorch

## Project Overview

This project demonstrates the development of a neural network using PyTorch to classify handwritten digits from the MNIST dataset. The primary goal was to achieve an accuracy of at least 90% on the test dataset, which was successfully exceeded. The project encompasses all steps from data preprocessing to model evaluation and saving, ensuring a robust and reusable solution.

---

## Features
- Dataset loading and preprocessing using PyTorch's torchvision.
- Exploration of dataset size, shape, and sample visualizations.
- Neural network construction with two or more hidden layers and a softmax output.
- Model training and hyperparameter tuning.
- Evaluation using accuracy metrics, and performance visualizations.
- Saving and loading the trained model for future use.

---

## Project Summary

### Section 1: Data Loading and Exploration

#### Dataset Loading and Preprocessing
- The MNIST dataset was loaded using `torchvision.datasets`.
- Data preprocessing steps included:
  - Converting images to tensors with `.ToTensor()` from `torchvision.transforms`.
  - Normalization to standardize pixel values.
  - Flattening images for network compatibility.
- Efficient batching was handled using `DataLoader` objects for both training and testing datasets.

#### Dataset Exploration
- The dataset's size and shape were analyzed to understand the input format.
- Sample images were visualized using `plt.imshow()` to verify preprocessing.
- A justification of preprocessing steps, such as normalization and flattening, is provided in the notebook.

### Section 2: Model Design and Training

#### Neural Network Construction
A custom feedforward neural network, `EnhancedNet`, was implemented with:
- Five layers of increasing depth to optimize feature extraction.
- ReLU activation functions for non-linearity.
- A softmax output layer to classify digits into 10 categories.

#### Training
- The network was trained using the Adam optimizer to minimize `CrossEntropyLoss`.
- Hyperparameters, including learning rate, were tuned to achieve high accuracy.
- The architecture:
  ```python
  class EnhancedNet(nn.Module):
      def __init__(self):
          super().__init__()
          self.activation = F.relu
          self.layer1 = nn.Linear(28 * 28, 512)
          self.layer2 = nn.Linear(512, 256)
          self.layer3 = nn.Linear(256, 128)
          self.layer4 = nn.Linear(128, 64)
          self.layer5 = nn.Linear(64, 10)

      def forward(self, x):
          x = torch.flatten(x, 1)
          x = self.activation(self.layer1(x))
          x = self.activation(self.layer2(x))
          x = self.activation(self.layer3(x))
          x = self.activation(self.layer4(x))
          x = self.layer5(x)
          return x
  ```

### Section 3: Model Testing and Evaluation

#### Testing
- The trained model was evaluated using the test set.
- Predictions were generated and compared against true labels using the test `DataLoader`.

#### Evaluation Metrics
- The model achieved an outstanding test accuracy of **98.21%**.
- Loss and accuracy metrics were visualized per epoch.
- A confusion matrix was generated to analyze misclassifications and overall performance.

#### Model Saving
- The trained model parameters were saved using `torch.save()` to allow reuse without retraining.

---

## Key Achievements
- Successfully achieved an accuracy of **98.21%** on the test set.
- Implemented and optimized a robust neural network architecture.
- Gained practical experience with PyTorch for building and evaluating classification models.

---

## Files in this Repository
- **Jupyter Notebook**: `Handwritten-Digits-Classifier.ipynb` 
  - Contains the full implementation with detailed explanations and visualizations.
- **Model File**: `enhanced_model.pth`
  - The saved PyTorch model ready for reuse.
- **Visualization File**: `training_metrics.png`
  - Graphs of training and validation loss and accuracy over epochs.

---

## Prerequisites
To run this project, ensure the following:
- Python 3.8+
- PyTorch 1.11+
- torchvision 0.12+
- Jupyter Notebook

Install dependencies using:
```bash
pip install torch torchvision notebook
```

---

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/AdrianCuevasML/Handwritten_Digits_Classifier_with_PyTorch.git
   ```
2. Open the Jupyter Notebook:
   ```bash
   jupyter notebook MNIST_Handwritten_Digits_Classifier.ipynb
   ```
3. Follow the notebook steps to train, evaluate, or reuse the saved model.

---

## Acknowledgments
This project was developed as part of the Udacity Machine Learning Fundamentals Nanodegree offered by AWS as part of the "AWS AI & ML Scholarship". It represents a significant milestone in my journey to mastering machine learning with PyTorch.

---

## Future Work
- Implementing CNNs for improved performance.
- Extending the project to include custom datasets.
- Exploring deployment options for real-world applications.

---

For any questions or feedback, feel free to contact me!

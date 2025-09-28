# Vision-Based Handwritten Math Symbol Recognition

## 📖 Project Overview

This project is a complete pipeline for recognizing individual handwritten mathematical symbols. Using a custom-built Convolutional Neural Network (CNN), this model takes 32x32 pixel images of symbols and classifies them into one of over 900 classes. The primary goal was to build, train, and evaluate a deep learning model capable of automated digitization of handwritten symbols, which is a foundational step towards recognizing full mathematical expressions.

After experimenting with various architectures, including transfer learning with MobileNetV2, the final and most effective model was a custom CNN architecture that achieved a **Test Accuracy of 68.04%** on a stratified test set.

## ✨ Features

-   **Data Loading & Preprocessing**: Scripts to load and preprocess the HASYv2 dataset, including image normalization.
-   **Model Architecture**: A custom CNN built with TensorFlow/Keras, featuring Convolutional layers, Batch Normalization for stability, and Dropout for regularization.
-   **Model Training**: Includes a complete training loop with a stratified data split to prevent data leakage and ensure a reliable validation score.
-   **Evaluation**: The model's performance is measured on a completely unseen test set.
-   **Prediction**: Ability to load the saved model and make predictions on new, single-symbol images.

## 🛠️ Tech Stack

-   **Language**: Python
-   **Core Libraries**:
    -   TensorFlow & Keras (for building and training the neural network)
    -   Scikit-learn (for the stratified data split)
    -   Pandas (for data manipulation)
    -   NumPy (for numerical operations)
    -   Matplotlib (for data visualization)
-   **Environment**: Jupyter Notebook in VS Code

## 🚀 Getting Started

### Prerequisites

-   Python 3.10+
-   Homebrew (for macOS users)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/harshgalba/handwritten-math-recognition.git](https://github.com/harshgalba/handwritten-math-recognition.git)
    cd handwritten-math-recognition
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required packages:**
    ```bash
    pip3 install -r requirements.txt
    ```
    *(Note: You may need to update the `requirements.txt` file with the libraries we installed: `pandas`, `matplotlib`, `tensorflow`, `scikit-learn`, `tqdm`)*

## Usage

The main workflow and all experiments are documented in the `notebooks/EDA.ipynb` Jupyter Notebook. To run the project:

1.  Launch VS Code and open the project folder.
2.  Open the `notebooks/EDA.ipynb` notebook.
3.  Ensure you select the `venv` kernel.
4.  Run the cells sequentially to load data, build the model, train it, and evaluate the results.

## Results

The final model, a custom CNN with Batch Normalization and a stratified data split, achieved the following performance:

-   **Training Accuracy**: ~69.9%
-   **Validation Accuracy**: ~68.0%
-   **Final Test Accuracy**: **68.04%**

The close alignment between validation and test accuracy indicates that the model is generalizing well and is not significantly overfit.

---

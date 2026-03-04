# IMDB Sentiment Analysis using TensorFlow
## Project Overview

This project performs **sentiment analysis on movie reviews** using the **IMDB dataset**.
The model is built using **TensorFlow**, **TensorFlow Hub**, and **TensorFlow Datasets**.

The goal of this project is to classify movie reviews as:

* **Positive Review**
* **Negative Review**

The model uses a **pre-trained word embedding layer** from TensorFlow Hub and a simple **neural network classifier**.

---

## Technologies Used

* Python
* TensorFlow
* TensorFlow Hub
* TensorFlow Datasets
* NumPy
* Matplotlib

---

## Dataset

The dataset used in this project is the **IMDB Movie Reviews Dataset**.

Dataset details:

* 50,000 movie reviews
* 25,000 training reviews
* 25,000 testing reviews
* Binary labels:

  * `1` → Positive
  * `0` → Negative

The dataset is automatically downloaded using TensorFlow Datasets.

---

## Model Architecture

Text Input
↓
TensorFlow Hub Embedding Layer (20-dimensional vector)
↓
Dense Layer (16 neurons, ReLU activation)
↓
Output Layer (1 neuron)
↓
Sentiment Prediction

---

## Features

* Loads IMDB movie reviews dataset
* Uses pre-trained word embeddings
* Trains a neural network for sentiment classification
* Evaluates model accuracy
* Plots **training and validation accuracy/loss graphs**
* Saves the trained model
* Allows prediction on custom movie reviews

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Project

Run the training script:

```bash
python train_model.py
```

The program will:

1. Download the dataset
2. Train the model
3. Show accuracy and loss graphs
4. Save the trained model

---

## Example Prediction

Example input:

```
This movie is fantastic!
```

Example output:

```
Prediction: Positive Review
```

---

## Output Visualization

The program generates two graphs:

* **Training vs Validation Loss**
* **Training vs Validation Accuracy**

These graphs help evaluate the model performance.

---

## Model Saving

The trained model is saved in:

```
models/sentiment_model
```

This model can later be loaded for predictions without retraining.

---

## Future Improvements

* Improve accuracy using deeper neural networks
* Add a web interface using Flask
* Deploy the model as an API
* Support real-time sentiment prediction

---

## Author

Student Project – Sentiment Analysis using TensorFlow

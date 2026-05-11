# Machine Learning Project — Sentiment Analysis on Amazon Reviews

This repository contains a Machine Learning project focused on **binary sentiment analysis** of Amazon Fashion product reviews.  
The goal is to classify each review as either **negative** or **positive** using Natural Language Processing techniques and deep learning models implemented in a Jupyter Notebook.

## Project Overview

The project analyzes a dataset of Amazon Fashion reviews organized into two classes:

- `0` → Negative review
- `1` → Positive review

The workflow includes:

1. Loading the dataset from a `.zip` archive
2. Building a Pandas DataFrame containing review texts and labels
3. Checking class distribution and dataset balance
4. Cleaning and preprocessing textual data
5. Tokenizing and padding text sequences
6. Splitting the dataset into training, validation and test sets
7. Training deep learning models for sentiment classification
8. Evaluating the models using accuracy curves and confusion matrices

## Repository Structure

```text
Machine-Learning-Project/
│
└── project.ipynb
```

The entire project is contained in the notebook `project.ipynb`.

## Dataset

The project expects the dataset to be available as a compressed archive named:

```text
AmazonFashionBinary.zip
```

In the original notebook, the dataset is loaded from Google Drive using the following path:

```text
/content/drive/MyDrive/MLProject/AmazonFashionBinary.zip
```

The expected archive structure is:

```text
AmazonFashionBinary/
├── 0/
│   └── negative review files
└── 1/
    └── positive review files
```

where folder `0` contains negative reviews and folder `1` contains positive reviews.

## Preprocessing

The preprocessing phase includes:

- Removal of empty reviews
- Conversion of text to lowercase
- Removal of numbers
- Removal of punctuation
- Tokenization using Keras `Tokenizer`
- Padding of sequences using Keras `pad_sequences`

The notebook uses the following main preprocessing parameters:

```python
NUM_WORDS = 5000
MAX_LENGTH = 50
```

Several experiments were performed by changing these values. In particular, the notebook highlights that the `MAX_LENGTH` parameter has a strong influence on model performance.

## Train / Validation / Test Split

The dataset is manually divided as follows:

- 80% Training set
- 10% Validation set
- 10% Test set

This is done using `train_test_split` from Scikit-learn.

## Models

The notebook focuses on deep learning models for text classification, including:

- LSTM
- Bidirectional LSTM
- Deeper neural architectures

### LSTM Architecture

The LSTM model includes:

- An `Embedding` layer to convert tokenized words into dense vectors
- An `LSTM` layer with 64 units
- A final `Dense` layer with sigmoid activation for binary classification

The model is trained using:

- Loss function: `binary_crossentropy`
- Optimizer: `Adam`
- Metric: `accuracy`
- Early stopping on validation loss

## Evaluation

Model performance is evaluated through:

- Test accuracy
- Training and validation loss curves
- Training and validation accuracy curves
- Confusion matrix

The confusion matrix is used to compare predicted labels against the true sentiment labels.

## Technologies Used

The project uses the following Python libraries:

- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Pi3raldoSanturro/Machine-Learning-Project.git
cd Machine-Learning-Project
```

### 2. Install the required dependencies

You can install the main dependencies with:

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib seaborn
```

### 3. Open the notebook

You can run the project locally using Jupyter Notebook:

```bash
jupyter notebook project.ipynb
```

or open it in Google Colab.

### 4. Set the dataset path

Before running the notebook, make sure the dataset path is correct.

If you are using Google Colab, upload the dataset to Google Drive and update:

```python
zip_path = "/content/drive/MyDrive/MLProject/AmazonFashionBinary.zip"
```

If you are running locally, replace it with the local path to your dataset, for example:

```python
zip_path = "AmazonFashionBinary.zip"
```

## Results

The notebook compares model behavior through accuracy, loss and confusion matrix analysis.  
The basic LSTM model already provides acceptable results for the sentiment analysis task, while larger vocabulary sizes increase training time without necessarily producing significantly better performance.

## Notes

This project was developed as an academic Machine Learning project and is intended to demonstrate a complete text classification pipeline, from dataset loading to model evaluation.

## Author

**Pieraldo Santurro**

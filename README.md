# California Housing Prices Prediction - Machine Learning Project

## Overview

This repository contains an end-to-end machine learning project focused on predicting housing prices in California. The dataset used for this project is sourced from Statlib and consists of housing data from the year 1990. Please note that due to the dataset's age, the model's predictions may not be proficient in forecasting current housing prices. However, this project serves as my learning experience for various aspects of building a machine learning project.

## Project Structure

The repository is organized as follows:

- **`data/`**: Contains the dataset (`california_housing.csv`) used for training and testing the machine learning model.
- **`notebooks/`**: Jupyter notebooks illustrating different stages of the machine learning pipeline, including data exploration, preprocessing, model training, and evaluation.
  - `01_data_splitting.ipynb`: Quick Look at the data and split into train and test sets
  - `02_data_exploration.ipynb`: Explore the train set to gain insights into its structure and characteristics.
  - `03_data_preprocessing.ipynb`: Prepare the data for training by handling missing values, scaling features, and encoding categorical variables.
  - `04_model_training.ipynb`: Train machine learning models using various algorithms.
  - `05_model_evaluation.ipynb`: Evaluate the trained models and analyze their performance.

- **`modules/`**: Custom Python modules to modularize the codebase.
  - `preprocessing.py`: Contains functions for handling data preprocessing tasks. Used in `04_model_training.ipynb`.

- **`models/`**: Contains the final trained model.
  - `my_california_housing_model.pkl.gz`: The serialized and compressed form of the trained machine learning model.

- **`requirements.txt`**: List of Python packages and their versions required to run the code.
- **`README.md`**: This document providing an overview of the project, its purpose, and instructions for running the code.

## How to test the project

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/your-username/california-housing-prediction.git
   ```

2. Explore the Jupyter notebooks in the notebooks/ directory to understand each step of the machine learning pipeline.

**Note**: Notebooks generate data necessary for subsequent notebooks. However, the required datasets have been pre-generated, and you can run the notebooks in any order that suits your preferences.

## Bibligraphy

Géron, A. (2019). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems. O'Reilly Media.

## Disclaimer

This project is primarily intended for educational purposes. Due to the outdated nature of the dataset, the machine learning model's predictions may not accurately reflect current housing prices in California. Use the code and insights gained from this project as a foundation for further exploration and learning in the field of data science and machine learning.




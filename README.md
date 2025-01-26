# IPL Winning Team Prediction with Machine Learning

This project predicts the winning team of an IPL match using machine learning algorithms. The goal is to analyze historical match data and other relevant features to build a model capable of forecasting the winner based on current match statistics. The model was developed as part of my internship at **Internpe**.

## Table of Contents

- [Overview](#overview)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Installation](#installation)
- [Model Training and Evaluation](#model-training-and-evaluation)

## Overview

The IPL (Indian Premier League) is one of the most popular cricket tournaments, and predicting the outcome of its matches can be a challenging yet exciting task. This project uses machine learning to predict the winner based on various features like team statistics, player performance, and match venue. The project incorporates data preprocessing, feature engineering, and model training to deliver accurate predictions.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib (for visualizations)
- Jupyter Notebooks

## Dataset

The dataset used for training the model includes historical match data from previous IPL seasons. It contains features such as:
- Teams playing in a match
- Venue details
- Player statistics
- Toss winner
- Match result

You can find the dataset in the `data/` folder.

## Model Training and Evaluation

The model has been trained using the following steps:
- Data Preprocessing: Missing values were handled, and categorical features were encoded.
- Feature Engineering: New features, such as the average player performance and match venue effects, were derived.
- Model Selection: Various machine learning algorithms were evaluated, including Logistic Regression, Decision Trees, and Random Forest.
- Model Evaluation: The model's accuracy was tested using cross-validation and evaluation metrics like accuracy score and confusion matrix.


## Installation

To get started with this project, clone the repository to your local machine and install the necessary dependencies.

```bash
git clone https://github.com/ramcharanpeesapati/IPL_Winning_Team_Prediction_with_ML.git
cd IPL_Winning_Team_Prediction_with_ML
pip install -r requirements.txt 


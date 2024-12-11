# Machine Learning Model

## Project Overview

**Objective**:  
The goal of this project is to build a machine learning model to recommend places destination for itinerary generator. 
Specifically, we aim to predict tourism destination based on inputted category, city, and budget range. This model can be used in Voyageur application or travel recommendation.

**Dataset Description**:  
- **Source**: The dataset is sourced from kaggle and google maps.
- **places_dataset**: Data of the 1015 tourism destination from 12 region (Bali, Bandung, Bogor, Jakarta, Lombok, Malang, Manado, Raja Ampat, Semarang, Solo, Surabaya, Yogyakarta).
  **Features**: Place_Id, Place_Name, Description, Category, City, Price, Rating, Link, Address
- **places_ratings**: User Reviews about the tourism destination from Google Maps Reviews, consisting 26372 data from 13173 users. 
- **Data Type**: Tabular

## Environment Setup

To run the project, make sure you have the following:

### Prerequisites
- Python 3.x (Recommended: [3.13])
- Tensorflow 2.15.0
- Tensorflow-recommenders

### Libraries
This project relies on the following libraries:
- `numpy`
- `pandas`
- `matplotlib`
- `tensorflow`
- `keras`
- `tensorflow_recommenders`
- `tempfile`
- `os`
- `collections`
- `json`

### Installing Dependencies
   ```python
  !pip install tensorflow==2.15.0 --quiet
  !pip install tensorflow-recommenders --no-deps
  ```

## Data Preprocessing
The following steps were taken to preprocess the data before feeding it into the model:

* **Data Loading:** The data was loaded from [[dataset](https://github.com/Voyageur-Team/Machine-Learning/tree/main/dataset)].
* **Missing Data Handling:** Missing values were handled using removal method.
* **Feature Engineering:** New features were created by binning the price feature to create price_category data. This feature needed to know the place category based on their price.
* **Convert data to tensorflow:** Convert data from pandas dataframe form to tensorflow form. 
* **Data Splitting:** The data was split into training and test sets using an [80% training, 20% test] split.

## Model Selection

This section describes the chosen model(s):

* **Model Type:** User Collaborative Recommendation System
* **Algorithm(s):** TFRS Retrieval model

## Model Training

The model was trained using the following steps:
* **Training Process:** The model was trained on the training set for 100 epochs with a batch size of 512. The optimizer used was tf.keras.optimizers.Adagrad() with learning rate 0.1..
* **Metrics:** The model's performance was evaluated using FactorizedTopK metric 

## Model Evaluation

The final model's performance on the test set is presented below:

* **factorized_top_k/top_1_categorical_accuracy:** [15%]
* **factorized_top_k/top_5_categorical_accuracy:** [53%]
* **factorized_top_k/top_10_categorical_accuracy:** [70%]
* **F1-factorized_top_k/top_50_categorical_accuracy:** [95%]
* **factorized_top_k/top_100_categorical_accuracy**:** [100%]

## Model Save
The model than saved to json file

## Model Load and Predict
To load the model, use 'serving_default' mode
``` python
    infer = loaded.signatures['serving_default']
```
to get the recommendation from loaded model, we need City, Category, and Price_Category inputs
```python
    inputs = {
        "City": tf.constant(['Surabaya']),
        "Category": tf.constant(['Edukasi']),
        "Price_Category": tf.constant(['0-25'])
    }
    
    results = infer(**inputs)
    place = results['output_2']
    recommended_places = [place.decode('utf-8') for place in place.numpy()[0]]
```

## Challenges & Limitations
* **Challenges:** The tourism destination always growth also the reviews data. 
* **Limitations:** The model only can doing user collaborative recommendation.

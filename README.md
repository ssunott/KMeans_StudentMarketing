# Car Price Prediction App
This app has been built using Streamlit and deployed with Streamlit community cloud

[Visit the app here](https://loan-application-app.streamlit.app/)

password - streamlit

This project is a machine learning web application built with Streamlit that predicts used car prices based on user-provided details such as model, mileage, fuel type, gearbox type, and more.

The model is trained on a cleaned car dataset using a Random Forest Regressor, and the `Model` column is encoded using Leave-One-Out Encoding to handle high cardinality. The app includes dynamic dropdowns populated from the dataset and displays real-time predictions.

---

## Features

* User-friendly interface powered by Streamlit.
* Real-time prediction of used car sales price based on user input.
* Accessible via Streamlit Community Cloud.

---
## Dataset

* Data sourced from Kaggle: https://www.kaggle.com/datasets/deepcontractor/car-price-prediction-challenge 

## Input Features

* Model (encoded with Leave-One-Out)
* Mileage
* Car Age
* Manufacturer
* Fuel type
* Engine volume
* Gear box type
* Drive wheels

---

## Machine Learning Model

* Model: RandomForestRegressor
* Preprocessing:

  * Remove outliers
  * Handle missing values
  * Leave-One-Out Encoding on `Model`
  * One-hot encoding for other categorical features

---

## Technologies Used
- **Streamlit**: For building the web application.
- **Scikit-learn**: For model training and evaluation.
- **Pandas** and **NumPy**: For data preprocessing and manipulation.
- **Matplotlib** and **Seaborn**: For exploratory data analysis and visualization (if applicable).

---

## Future Enhancements
* Adding support for multiple datasets.
* Incorporating explainability tools like SHAP to provide insights into predictions.
* Adding visualizations to better represent user input and model predictions.

## Installation (for local deployment)
If you want to run the application locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/credit_eligibility_application.git
   cd credit_eligibility_application

2. Create and activate a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows, use `env\\Scripts\\activate`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt

4. Run the Streamlit application:
   ```bash
   streamlit run app.py

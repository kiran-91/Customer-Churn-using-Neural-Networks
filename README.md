# Customer Churn Prediction using Artificial Neural Networks

## Project Overview
This repository contains a machine learning project for predicting customer churn using Artificial Neural Networks (ANN). The project includes a Streamlit web app that allows users to input customer data and predict whether the customer is likely to churn. The model is trained on historical customer data, preprocessed, and visualized using TensorBoard to track the learning process.

## Key Features
1. Data Preprocessing: Dropping unnecessary columns, Encoding categorical variables using different techniques, feature scaling.
2. Artificial Neural Network: Built an Artificial Neural Network using TensorFlow/Keras for accurate churn predictions.
3. Model Training & Visualization: Monitor the model’s performance using TensorBoard.
4. Streamlit Web App: User-friendly interface for making real-time churn predictions based on user input.

## Requirements 
1. Python
2. Pandas
3. Scikit-learn
4. Streamlit
5. Tensorflow/keras
6. Plotly


## Installation
To run the project locally, follow these steps:

1. Clone the repository:

```cmd
git clone https://https://github.com/kiran-91/Customer-Churn-using-Neural-Nets.git
cd Customer-Churn-using-Neural-Nets
```

2. Setup a virtual environment (optional but recommended)
```cmd
python -m venv venv
source venv/bin/activate  # For Linux/MacOS
venv\Scripts\activate  # For Windows
```

3. Install required dependencies
```cmd
pip install -r requirements.txt
```

4. Run the streamlit app
```cmd
streamlit run app.py
```

## Results 
If you're on Team Lazy like me and would rather skip all the tasks, no worries—just kick back and check out the Streamlit app right here!
```
cuspre.streamlit.app
```
## Usage 

### Customer Demographics:
1. Country: Select the customer's country from the dropdown list.
2. Gender: Choose the customer’s gender (Male or Female) by selecting the appropriate radio button.
3. Age: Adjust the slider to set the customer’s age between 18 and 92.

### Account Information:
1. Account Balance: Enter the customer’s account balance.
2. Credit Score: Use the slider to select the customer’s credit score (between 350 and 850).

### Customer Engagement:
1. Monthly Salary: Adjust the slider to set the customer’s monthly salary (from 10 to 200,000).
2. Credit Tenure (years): Use the slider to specify the number of years the customer has been using credit (from 0 to 10).
3. Number of Products: Adjust the slider to set the number of products the customer is using (from 1 to 4).
4. Has a Credit Card: Check this box if the customer has a credit card.
5. Is an Active Member: Check this box if the customer is an active member.

### Prediction:
Once all fields are filled, click the "Predict Churn Probability" button to get the prediction and the app will output the probability that the customer will churn based on the inputs provided

### Prediction Explanation:
There is a generic explanation on how the model works and how the prediction is made 

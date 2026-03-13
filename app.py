from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

# load the model and scaler
with open('model.pkl', 'rb') as f:
    kmeans = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# create FastAPI app
app = FastAPI()

# define the input data model
class Customer(BaseModel):
    gender: int
    age: int
    income: int
    spending_score: int

# Cluster names
cluster_names = {
    0: 'Mature Low Spenders',
    1: 'Rich but Stingy',
    2: 'Young Active Buyers',
    3: 'VIP Customers',
    4: 'Impulsive Spenders'
}

# Root endpoint
@app.get('/')
def home():
    return {'message': 'Customer Segmentation API is running!'}

# Prediction endpoint
@app.post('/predict_cluster')
def predict_cluster(customer: Customer):
    
    data = pd.DataFrame([[
        customer.gender,
        customer.age,
        customer.income,
        customer.spending_score
    ]], columns=['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)'])
    
    scaled = scaler.transform(data)
    cluster = kmeans.predict(scaled)[0]
    
    return {
        'cluster': int(cluster),
        'segment': cluster_names[cluster]
    }
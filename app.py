
import pickle
import pandas as pd

with open('model.pkl', 'rb') as f:
    kmeans = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

class Customer(BaseModel):
    income: float
    spending_score: float

cluster_names = {
    0: 'Moderate Customers',
    1: 'High Income High Spending (VIP)',
    2: 'Low Income High Spending',
    3: 'High Income Low Spending',
    4: 'Low Income Low Spending'
}

@app.get('/')
def home():
    return {'message': 'Customer Segmentation API is running'}

@app.post('/predict')
def predict(customer: Customer):

    data = pd.DataFrame([[customer.income, customer.spending_score]],
        columns=['Annual Income (k$)', 'Spending Score (1-100)']
    )

    scaled = scaler.transform(data)
    cluster = kmeans.predict(scaled)[0]

    return {
        'cluster': int(cluster),
        'segment': cluster_names[cluster]
    }
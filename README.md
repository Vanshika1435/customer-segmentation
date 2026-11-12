# Customer Segmentation

An interactive machine learning application that groups customers into distinct segments based on their purchasing behavior and spending patterns. The project uses K-Means clustering and an interactive Streamlit dashboard to explore customer groups and predict the segment of a new customer.

## Live Demo

**[Launch Customer Segmentation Dashboard](https://customer-segmentation-rkd6.onrender.com/)**

Explore customer clusters, interactive visualizations, and predict customer segments based on annual income and spending score.

## Overview

Customer segmentation is an important technique in data analytics and marketing. It helps businesses understand customer behavior, identify different customer groups, and make data-driven decisions.

This project applies unsupervised machine learning to customer data to identify meaningful segments based on features such as annual income and spending score.

The application provides an interactive dashboard where users can explore customer clusters, visualize patterns, and predict the segment of a customer using the trained model.

## Features

- **Customer Data Analysis:** Explore customer information and identify patterns in the dataset.
- **K-Means Clustering:** Group customers into segments using an unsupervised machine learning algorithm.
- **Cluster Distribution:** Visualize the distribution of customers across different clusters.
- **Interactive Visualizations:** Explore relationships between annual income, spending score, and customer segments.
- **3D Cluster Visualization:** View customer groups through an interactive three-dimensional visualization.
- **Cluster Summary:** Examine characteristics and statistics associated with each customer segment.
- **Customer Segment Prediction:** Enter annual income and spending score to predict the corresponding customer segment using the saved model.
- **Interactive Dashboard:** Access the analysis and prediction functionality through a Streamlit web application.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| Algorithm | K-Means Clustering |
| Data Processing | Pandas, NumPy |
| Data Visualization | Plotly |
| Web Application | Streamlit |
| Model Persistence | Pickle |
| Development Environment | Jupyter Notebook, VS Code |

## Machine Learning Approach

The project uses K-Means, an unsupervised machine learning algorithm that groups data points into clusters based on similarity.

### Workflow

1. **Data Collection:** Load the customer dataset containing customer-related attributes.
2. **Data Preprocessing:** Select relevant numerical features and prepare the data for clustering.
3. **Feature Scaling:** Apply feature scaling to ensure that differences in feature ranges do not disproportionately influence the clustering process.
4. **Model Training:** Train a K-Means clustering model to identify customer groups.
5. **Model Persistence:** Save the trained model and scaler using Pickle for reuse in the dashboard.
6. **Visualization:** Display the identified customer groups through interactive charts.
7. **Prediction:** Accept new customer information, apply the saved scaler, and use the trained model to assign a cluster.

### Algorithm: K-Means Clustering

K-Means is an unsupervised learning algorithm that partitions data into a predefined number of clusters.

The algorithm works by:

- Initializing cluster centroids.
- Assigning each data point to its nearest centroid.
- Recalculating centroids based on the assigned data points.
- Repeating the process until the cluster assignments stabilize.

The resulting clusters help identify groups of customers with similar characteristics.

## Application Workflow

```text
             Customer Dataset
                    |
                    v
             Data Preprocessing
                    |
                    v
              Feature Scaling
                    |
                    v
            K-Means Clustering
                    |
                    v
             Trained Model
                    |
             +------+------+
             |             |
             v             v
       Saved Model     Saved Scaler
             |             |
             +------+------+
                    |
                    v
           Streamlit Dashboard
                    |
          +---------+---------+
          |                   |
          v                   v
    Data Visualization   Customer Input
                              |
                              v
                       Feature Scaling
                              |
                              v
                      Cluster Prediction
                              |
                              v
                      Predicted Segment
```

## Dashboard Functionality

The Streamlit dashboard provides two main areas of functionality.

### 1. Customer Segmentation Analysis

- Customer cluster distribution.
- Annual income versus spending score visualization.
- Interactive three-dimensional cluster visualization.
- Cluster-level summaries and statistics.

### 2. Customer Segment Prediction

Users can provide:

- Annual Income
- Spending Score

The application processes the input using the saved feature scaler and passes the transformed data to the trained K-Means model.

The predicted cluster represents the segment assigned to the customer.

## Project Structure

```text
customer-segmentation/
│
├── analysis.ipynb
├── dashboard.py
├── app.py
│
├── Mall_Customers.csv
├── model.pkl
├── scaler.pkl
│
├── requirements.txt
└── README.md
```

### Important Files

| File | Description |
|---|---|
| `analysis.ipynb` | Data exploration, preprocessing, and model development. |
| `dashboard.py` | Streamlit application for customer analysis and segment prediction. |
| `app.py` | Separate FastAPI application entry point. |
| `Mall_Customers.csv` | Customer dataset used in the project. |
| `model.pkl` | Serialized trained K-Means model. |
| `scaler.pkl` | Serialized feature scaler. |
| `requirements.txt` | Python dependencies required to run the application. |

## Run Locally

### Prerequisites

- Python 3.10 or a compatible Python version.
- Git.
- pip.

### 1. Clone the Repository

```bash
git clone https://github.com/Vanshika1435/customer-segmentation.git

cd customer-segmentation
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application

```bash
streamlit run dashboard.py
```

The application will be available at:

```text
http://localhost:8501
```

## Deployment

The interactive Streamlit dashboard is deployed on **Render**.

**Live Application:** https://customer-segmentation-rkd6.onrender.com/

**Source Code:** https://github.com/Vanshika1435/customer-segmentation

## Key Learning Outcomes

- Understanding unsupervised machine learning and clustering techniques.
- Implementing K-Means clustering using Scikit-learn.
- Performing data preprocessing and feature scaling.
- Saving and reusing trained machine learning models.
- Creating interactive data visualizations using Plotly.
- Developing machine learning applications using Streamlit.
- Connecting trained models with user inputs for interactive predictions.
- Deploying a Python-based machine learning application.

## Future Improvements

- Add additional customer attributes for more detailed segmentation.
- Improve cluster interpretation and generate business-oriented recommendations.
- Compare K-Means with alternative clustering algorithms.
- Add more detailed customer analytics and downloadable reports.
- Extend the application with additional prediction and analysis capabilities.

---

**Author:** Vanshika Sharma

**GitHub:** https://github.com/Vanshika1435

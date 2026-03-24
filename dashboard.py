import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle

# Model load karo
with open('model.pkl', 'rb') as f:
    kmeans = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Data load karo
df = pd.read_csv('Mall_Customers.csv')
df = df.drop('CustomerID', axis=1)
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# Cluster names
cluster_names = {
    0: 'Mature Low Spenders ',
    1: 'Rich but Stingy ',
    2: 'Young Active Buyers ',
    3: 'VIP Customers ',
    4: 'Impulsive Spenders '
}

# Clusters add karo
X = scaler.transform(df[['Annual Income (k$)', 'Spending Score (1-100)']])
df['Cluster'] = kmeans.predict(X)
df['Segment'] = df['Cluster'].map(cluster_names)

# ─── Dashboard UI ───

st.title('Customer Segmentation Dashboard')
st.markdown('K-Means Clustering on Mall Customers Dataset')

# ─── Section 1: Cluster Distribution ───
st.header('Cluster Distribution')

count_df = df['Segment'].value_counts().reset_index()
count_df.columns = ['Segment', 'Count']

fig1 = px.bar(
    count_df,
    x='Segment',
    y='Count',
    color='Segment',
    title='Customers per Segment'
)
st.plotly_chart(fig1, use_container_width=True)

# ─── Section 2: 2D Scatter Plot ───
st.header(' Income vs Spending Score')

fig2 = px.scatter(
    df,
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    color='Segment',
    title='Customer Segments — 2D View',
    size_max=10
)
st.plotly_chart(fig2, use_container_width=True)

# ─── Section 3: 3D Plot ───
st.header(' 3D Visualization')

fig3 = px.scatter_3d(
    df,
    x='Age',
    y='Annual Income (k$)',
    z='Spending Score (1-100)',
    color='Segment',
    title='Customer Segments — 3D View'
)
st.plotly_chart(fig3, use_container_width=True)

# ─── Section 4: Cluster Summary ───
st.header('Cluster Summary')

summary = df.groupby('Segment').mean(numeric_only=True).round(1)
summary = summary.drop('Cluster', axis=1)
st.dataframe(summary)

# ─── Section 5: Predict New Customer ───
st.header('Predict New Customer Segment')

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox('Gender', ['Male', 'Female'])
    age = st.slider('Age', 18, 70, 25)

with col2:
    income = st.slider('Annual Income (k$)', 15, 140, 60)
    spending = st.slider('Spending Score', 1, 100, 50)

if st.button('Predict Segment'):
    gender_val = 0 if gender == 'Male' else 1
    
    new_data = pd.DataFrame([[income, spending]],
            columns=['Annual Income (k$)', 
                     'Spending Score (1-100)'])
    scaled = scaler.transform(new_data)
    cluster = kmeans.predict(scaled)[0]
    
    st.success(f'Customer belongs to: **{cluster_names[cluster]}**')
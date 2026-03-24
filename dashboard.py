
import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

with open('model.pkl', 'rb') as f:
    kmeans = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

df = pd.read_csv('Mall_Customers.csv')

data = df[['Annual Income (k$)', 'Spending Score (1-100)']]

scaled_data = scaler.transform(data)
df['Cluster'] = kmeans.predict(scaled_data)

cluster_names = {
    0: 'Moderate Customers',
    1: 'High Income High Spending (VIP)',
    2: 'Low Income High Spending',
    3: 'High Income Low Spending',
    4: 'Low Income Low Spending'
}

df['Segment'] = df['Cluster'].map(cluster_names)

# Title
st.title('Customer Segmentation Dashboard')

st.header('Cluster Distribution')

count_df = df['Segment'].value_counts().reset_index()
count_df.columns = ['Segment', 'Count']

fig1 = px.bar(count_df, x='Segment', y='Count', color='Segment')
st.plotly_chart(fig1, use_container_width=True)

st.header('Income vs Spending Score')

fig2 = px.scatter(
    df,
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    color='Segment'
)
st.plotly_chart(fig2, use_container_width=True)

st.header('3D Visualization')

fig3 = px.scatter_3d(
    df,
    x='Annual Income (k$)',
    y='Spending Score (1-100)',
    z='Cluster',
    color='Segment'
)
st.plotly_chart(fig3, use_container_width=True)

st.header('Cluster Summary')

summary = df.groupby('Segment').mean(numeric_only=True).round(1)
st.dataframe(summary)

st.header('Predict Customer Segment')

col1, col2 = st.columns(2)

with col1:
    income = st.slider('Annual Income (k$)', 15, 140, 60)

with col2:
    spending = st.slider('Spending Score', 1, 100, 50)

if st.button('Predict'):
    
    input_df = pd.DataFrame([[income, spending]],
        columns=['Annual Income (k$)', 'Spending Score (1-100)']
    )

    scaled_input = scaler.transform(input_df)
    cluster = kmeans.predict(scaled_input)[0]

    st.success(f'Segment: {cluster_names[cluster]}')
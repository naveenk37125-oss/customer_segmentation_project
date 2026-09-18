import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

def generate_synthetic_data(num_records=1000, num_customers=200):
    np.random.seed(42)
    customer_ids = [f"CUST{1000 + i}" for i in range(num_customers)]
    
    dates = [datetime.now() - timedelta(days=int(np.random.exponential(scale=60))) for _ in range(num_records)]
    cust_choices = np.random.choice(customer_ids, size=num_records)
    amounts = np.random.exponential(scale=150, size=num_records) + 10.0
    quantities = np.random.randint(1, 10, size=num_records)
    
    df = pd.DataFrame({
        'InvoiceNo': [f"INV{10000 + i}" for i in range(num_records)],
        'CustomerID': cust_choices,
        'InvoiceDate': dates,
        'Quantity': quantities,
        'UnitPrice': np.round(amounts / quantities, 2)
    })
    df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
    return df

def calculate_rfm(df):
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    max_date = df['InvoiceDate'].max() + timedelta(days=1)
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (max_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalAmount': 'sum'
    }).reset_index()
    
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    return rfm

def transform_features(rfm):
    rfm_log = rfm[['Recency', 'Frequency', 'Monetary']].apply(np.log1p)
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)
    return rfm_scaled, scaler

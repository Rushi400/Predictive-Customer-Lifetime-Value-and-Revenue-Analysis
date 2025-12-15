import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv('transactions.csv')
df['OrderDate'] = pd.to_datetime(df['OrderDate'])

clv = df.groupby('CustomerID').agg(
    total_orders=('OrderID', 'count'),
    total_revenue=('Revenue', 'sum'),
    avg_order_value=('Revenue', 'mean'),
    first_purchase=('OrderDate', 'min'),
    last_purchase=('OrderDate', 'max')
).reset_index()

clv['customer_lifespan'] = (clv['last_purchase'] - clv['first_purchase']).dt.days / 365
clv['CLV'] = clv['avg_order_value'] * clv['total_orders'] * clv['customer_lifespan']

clv['CLV_Segment'] = pd.qcut(clv['CLV'], 3, labels=['Low Value', 'Medium Value', 'High Value'])

df['Month'] = df['OrderDate'].dt.to_period('M').astype(str)
monthly = df.groupby('Month')['Revenue'].sum().reset_index()
monthly['index'] = range(len(monthly))

model = LinearRegression()
model.fit(monthly[['index']], monthly['Revenue'])
monthly['Forecasted_Revenue'] = model.predict(monthly[['index']])

clv.to_csv('clv_output.csv', index=False)
monthly.to_csv('monthly_revenue_forecast.csv', index=False)

print('Files generated successfully')

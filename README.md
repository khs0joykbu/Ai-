## 🧠 AdventureWorks Data Pipeline Project

## 📊 개요
이 프로젝트는 Microsoft AdventureWorks 데이터를 기반으로 한 **ETL → 전처리 → EDA → 모델링 → API 배포** 파이프라인입니다.

## 📁 프로젝트 구조


import pandas as pd
import sqlalchemy
import os

sales = pd.read_csv('Sales.csv')
customer = pd.read_csv('Customer.csv')

sales['OrderDate'] = pd.to_datetime(sales['OrderDate'])
sales = sales.drop_duplicates()

engine = sqlalchemy.create_engine('sqlite:///adventure_sales.db')
sales.to_sql('Sales', engine, index=False, if_exists='replace')
customer.to_sql('Customer', engine, index=False, if_exists='replace')

print(pd.read_sql('SELECT COUNT(*) FROM Sales', engine))

-------------------------------------------------------------------------------

import pandas as pd
from sklearn.preprocessing import StandardScaler

engine = sqlalchemy.create_engine('sqlite:///adventure_sales.db')
sales = pd.read_sql('SELECT * FROM Sales', engine)

sales = sales.dropna(subset=['CustomerID', 'TotalDue'])

sales = sales[sales['TotalDue'] < sales['TotalDue'].quantile(0.99)]

scaler = StandardScaler()
sales[['TotalDue']] = scaler.fit_transform(sales[['TotalDue']])

sales.to_sql('Sales_preprocessed', engine, index=False, if_exists='replace')

-------------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlalchemy

engine = sqlalchemy.create_engine('sqlite:///adventure_sales.db')
sales = pd.read_sql('SELECT * FROM Sales_preprocessed', engine)

print(sales.describe())

plt.figure(figsize=(8,6))
sns.boxplot(x='TerritoryID', y='TotalDue', data=sales)
plt.title('지역별 TotalDue 분포')
plt.show()

plt.figure(figsize=(6,5))
sns.heatmap(sales.corr(), annot=True, cmap='coolwarm')
plt.title('변수 간 상관관계')
plt.show()

-------------------------------------------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib
import sqlalchemy

engine = sqlalchemy.create_engine('sqlite:///adventure_sales.db')
sales = pd.read_sql('SELECT * FROM Sales_preprocessed', engine)

X = sales[['TerritoryID', 'SubTotal']]
y = sales['TotalDue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("R² Score:", r2_score(y_test, y_pred))

joblib.dump(model, 'adventure_model.pkl')

-------------------------------------------------------------------------------
from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load('adventure_model.pkl')

@app.post("/predict/")
def predict(data: dict):
    df = pd.DataFrame([data])
    pred = model.predict(df)[0]
    return {"prediction": pred}

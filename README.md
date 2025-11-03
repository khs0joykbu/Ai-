## 🧠 AdventureWorks Data Pipeline Project

## 📊 개요
이 프로젝트는 Microsoft AdventureWorks 데이터를 기반으로 한 **ETL → 전처리 → EDA → 모델링 → API 배포** 파이프라인입니다.

## 📁 프로젝트 구조


# 1. 라이브러리 import
import pandas as pd
import sqlalchemy
import os

# 2. 데이터 추출 (Extract)
sales = pd.read_csv('Sales.csv')
customer = pd.read_csv('Customer.csv')

# 3. 데이터 변환 (Transform)
sales['OrderDate'] = pd.to_datetime(sales['OrderDate'])
sales = sales.drop_duplicates()

# 4. 데이터 적재 (Load)
engine = sqlalchemy.create_engine('sqlite:///adventure_sales.db')
sales.to_sql('Sales', engine, index=False, if_exists='replace')
customer.to_sql('Customer', engine, index=False, if_exists='replace')

# 5. 로드 확인
print(pd.read_sql('SELECT COUNT(*) FROM Sales', engine))

-------------------------------------------------------------------------------

# 1. 라이브러리 import
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 2. DB 불러오기
engine = sqlalchemy.create_engine('sqlite:///adventure_sales.db')
sales = pd.read_sql('SELECT * FROM Sales', engine)

# 3. 결측치 처리
sales = sales.dropna(subset=['CustomerID', 'TotalDue'])

# 4. 이상치 처리
sales = sales[sales['TotalDue'] < sales['TotalDue'].quantile(0.99)]

# 5. 스케일링
scaler = StandardScaler()
sales[['TotalDue']] = scaler.fit_transform(sales[['TotalDue']])

# 6. 전처리 결과 저장
sales.to_sql('Sales_preprocessed', engine, index=False, if_exists='replace')

-------------------------------------------------------------------------------

# 1. 라이브러리 import
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlalchemy

# 2. 데이터 불러오기
engine = sqlalchemy.create_engine('sqlite:///adventure_sales.db')
sales = pd.read_sql('SELECT * FROM Sales_preprocessed', engine)

# 3. 기본 통계
print(sales.describe())

# 4. 시각화
plt.figure(figsize=(8,6))
sns.boxplot(x='TerritoryID', y='TotalDue', data=sales)
plt.title('지역별 TotalDue 분포')
plt.show()

# 5. 상관관계 히트맵
plt.figure(figsize=(6,5))
sns.heatmap(sales.corr(), annot=True, cmap='coolwarm')
plt.title('변수 간 상관관계')
plt.show()

-------------------------------------------------------------------------------
# 1. 라이브러리 import
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import joblib
import sqlalchemy

# 2. 데이터 불러오기
engine = sqlalchemy.create_engine('sqlite:///adventure_sales.db')
sales = pd.read_sql('SELECT * FROM Sales_preprocessed', engine)

# 3. 학습 데이터 구성
X = sales[['TerritoryID', 'SubTotal']]
y = sales['TotalDue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 모델 학습
model = RandomForestRegressor()
model.fit(X_train, y_train)

# 5. 평가
y_pred = model.predict(X_test)
print("R² Score:", r2_score(y_test, y_pred))

# 6. 모델 저장
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

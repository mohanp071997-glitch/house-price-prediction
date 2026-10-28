import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

df=pd.read_csv("house_prices_1000_records.csv")

from sklearn.impute import SimpleImputer

import joblib

print("First 5 Records : ")
print(df.head())

print("\nDataset Shape: ")
print(df.shape)

print("\nColumns Informations: ")
df.info()

print("\nMissing Value: ")
print(df.isnull().sum())

print("\nStatistical Summary: ")
print(df.describe())

print("Total Columns : ",len(df.columns))
print("Columns name : ",df.columns.tolist())

print("Missing values count : ")
print(df.isnull().sum())

print("\nMissing values percentage : ")
print((df.isnull().sum()/len(df))*100)


# Create X and y AFTER cleaning

print("\n--- Features (X) ---")

X = df[
    [
        "Area_sqft",
        "Bedrooms",
        "Bathrooms",
        "Age_years",
        "Location",
        "Parking",
        "Furnishing"
    ]
].copy()

y = df["Price_Lakhs"].copy()

print("\n--- Features Before Encoding ---")
print(X.head())

print("Target y Shape:", y.shape)

print("\n--- Missing Values in X ---")
print(X.isnull().sum())



numeric_columns = [
    "Area_sqft",
    "Bedrooms",
    "Bathrooms",
    "Age_years",
    "Price_Lakhs"
]

for column in numeric_columns:
    plt.figure(figsize=(7, 3))
    sns.boxplot(x=df[column])
    plt.title(f"Box Plot - {column}")
    plt.show()


plt.figure(figsize=(8,5))
sns.boxplot(x=df["Price_Lakhs"])

plt.title("House Price - Box Plot")

plt.xlabel("Price (Lakhs)")
plt.show()


plt.figure(figsize=(8,5))

sns.histplot(data=df,x="Area_sqft",bins=20,kde=True)

plt.title("Distribution of House Area")

plt.xlabel("Area(Square feet)")

plt.ylabel("Number of Houses")

plt.show()

print("\n---Area Statistics---")

print("Mean Area : ",df["Area_sqft"].mean())
print("Median Area : ",df["Area_sqft"].median())

print("\n---Price Statistics---")
print("Mean Price : ",df["Price_Lakhs"].mean())
print("Median Price : ",df["Price_Lakhs"].median())

print("\n----Skewness Analysis---")
print("Area Skewness : ",df["Area_sqft"].skew())
print("Price Skewness : ",df["Price_Lakhs"].skew())

plt.figure(figsize=(8,5))

sns.scatterplot(data=df,x="Area_sqft",y="Price_Lakhs")

plt.title("House Area vs House Price")
plt.xlabel("Area (square feet)")
plt.ylabel("Price (Lakhs)")

plt.show()

print("\n---Correlation Analysis---")

correlation =df["Area_sqft"].corr(df["Price_Lakhs"])
print("Area vs Price Correlation : ",correlation)

numeric_columns=["Area_sqft","Bedrooms","Bathrooms","Age_years","Price_Lakhs"]

correlation_matrix=df[numeric_columns].corr()

plt.figure(figsize=(9,6))

sns.heatmap(correlation_matrix,annot=True,cmap="coolwarm",fmt=".2f",vmin=-1,vmax=1)

plt.title("Correlation Heatmap")

plt.show()

print("\n---Feature and Target Shape---")

print("X shape : ",X.shape)
print("y shape : ",y.shape)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

print("\n---Train/Test split---")

print("X_train : ",X_train.shape)

print("X_test : ",X_test.shape)

print("y.train : ",y_train.shape)

print("y_test : ",y_test.shape)

# model =LinearRegression()

# model.fit(X_train,y_train)

# print("\n---Model Training---")
# print("Linear Regression model training successfully")

# y_pred=model.predict(X_test)

numeric_features=["Area_sqft","Bedrooms","Bathrooms","Age_years"]
categorical_features=["Location","Parking","Furnishing"]

print("Numeric Features : ",numeric_features)
print("Categorical Features : ",categorical_features)

numeric_tranformer=Pipeline(steps=[("imputer",SimpleImputer(strategy='median'))])
categorical_tranformer=Pipeline(steps=[("imputer",SimpleImputer(strategy="most_frequent")),
                                       ("encoder",OneHotEncoder(handle_unknown="ignore"))])

preprocessor=ColumnTransformer(transformers=[("num",numeric_tranformer,numeric_features),
                                             ("cat",categorical_tranformer,categorical_features)])

print("\nPreProcessor Created Successfully!")

model_pipeline=Pipeline(steps=[("preprocessor",preprocessor),
                               ("model",LinearRegression())])
print("\nModel pipeline created successfully!")

model_pipeline.fit(X_train,y_train)

print("\nPipeline Model Training Successfully!")

y_pred=model_pipeline.predict(X_test)


print("\n---First 10 Actual Price---")
print(y_test.head(10).to_list())
print("\n---First 10 Predicted Price---")
print(y_pred[:10])

mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)

r2=r2_score(y_test,y_pred)

print("\n---Model Evaluation---")
print("MAE : ",round(mae,2))
print("MSE : ",round(mse,2))
print("RMSE : ",round(rmse,2))
print("R2 score : ",round(r2,4))


plt.figure(figsize=(8,5))
plt.scatter(y_test,y_pred)

plt.plot([y_test.min(),y_test.max()],
         [y_test.min(),y_test.max()],"r--")

plt.title("Actual vs Predicted House Prices")
plt.xlabel("Actual Price(Lakhs)")
plt.ylabel("Predicted Price(Lakhs)")
plt.show()

y_train_pred=model_pipeline.predict(X_train)
train_r2=r2_score(y_train,y_train_pred)

test_r2=r2_score(y_test,y_pred)
print("\nTraining R2 Score : ",round(train_r2,4))
print("Testing R2 score : ",round(test_r2,4))

# Predict price for a new house

new_house = pd.DataFrame([{
    "Area_sqft": 1500,
    "Bedrooms": 3,
    "Bathrooms": 2,
    "Age_years": 5,
    "Location": "Coimbatore",
    "Parking": "Yes",
    "Furnishing": "Semi-Furnished"
}])

predicted_price = model_pipeline.predict(new_house)

print("\n--- New House Price Prediction ---")
print("Predicted Price:", round(predicted_price[0], 2), "Lakhs")

joblib.dump(model_pipeline,"house_price_model.pkl")
print("Model Saved SuccessFully!")

# Load the saved model
loaded_model = joblib.load("house_price_model.pkl")

# New house details
new_house = pd.DataFrame([{
    "Area_sqft": 1500,
    "Bedrooms": 3,
    "Bathrooms": 2,
    "Age_years": 5,
    "Location": "Coimbatore",
    "Parking": "Yes",
    "Furnishing": "Semi-Furnished"
}])

# Predict house price using the loaded model
predicted_price = loaded_model.predict(new_house)

print("\n--- Prediction Using Saved Model ---")
print("Predicted Price:", round(predicted_price[0], 2), "Lakhs")
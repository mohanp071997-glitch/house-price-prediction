import pandas as pd
import joblib

loaded_model=joblib.load("house_price_model.pkl")
area=float(input("Enter Area (sqft): "))
bedrooms=int(input("Enter Bedrooms: "))
bathrooms=int(int(input("Enter Bathrooms: ")))
age=int(input("Enter House Age (Years): "))
location=input("Enter location: ")
parking=input("Parking (Yes/No): ")
furnishing=input("Furnishing: ")


new_house=pd.DataFrame([{
    "Area_sqft":area,
    "Bedrooms":bedrooms,
    "Bathrooms":bathrooms,
    "Age_years":age,
    "Location":location,
    "Parking":parking,
    "Furnishing":furnishing}])

predicted_price=loaded_model.predict(new_house)

print("House Price Prediction")
print("----------------------")


print(f"Predicted Price : {predicted_price[0]:.2f} Lakhs")

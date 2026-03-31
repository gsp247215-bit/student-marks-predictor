import pandas as pd
from sklearn.linear_model import LinearRegression

# Create dataset
data = {
    'Hours': [1, 2, 3, 4, 5, 6, 7, 8],
    'Attendance': [50, 55, 60, 65, 70, 75, 80, 85],
    'Marks': [30, 35, 40, 50, 55, 60, 70, 75]
}

df = pd.DataFrame(data)

# Features (input)
X = df[['Hours', 'Attendance']]

# Target (output)
y = df['Marks']

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict
hours = float(input("Enter study hours: "))
attendance = float(input("Enter attendance: "))
new_data = pd.DataFrame({
    'Hours': [hours],
    'Attendance': [attendance]
})


prediction = model.predict(new_data)

print("Predicted Marks:", prediction[0])

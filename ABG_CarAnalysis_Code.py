import pandas as pd

japan_df = pd.read_excel("JPN Data.xlsx")
india_df = pd.read_excel("IN_Data.xlsx")


print("Japanese Dataset:")
print(japan_df.head())

print("\nIndian Dataset:")
print(india_df.head())


print("\nColumns in Japanese Dataset:", japan_df.columns.tolist())
print("Columns in Indian Dataset:", india_df.columns.tolist())



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Copy original data
data = japan_df.copy()

# Step 2: Encode Gender column (M=1, F=0)
data['GENDER'] = LabelEncoder().fit_transform(data['GENDER'])

# Step 3: Define X (features) and y (target)
X = data[['CURR_AGE', 'GENDER', 'ANN_INCOME', 'AGE_CAR']]
y = data['PURCHASE']

# Step 4: Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 6: Evaluate model
y_pred = model.predict(X_test)

print("\n Accuracy:", accuracy_score(y_test, y_pred))
print("\n Classification Report:\n", classification_report(y_test, y_pred))
print("\n Confusion Matrix:\n", confusion_matrix(y_test, y_pred))



# Step 1: Prepare Indian dataset for prediction
india = india_df.copy()

# Encode Gender like before (M=1, F=0)
india['GENDER'] = LabelEncoder().fit_transform(india['GENDER'])

# Step 2: Add a dummy AGE_CAR column (assume avg = 400 months from Japan data)
india['AGE_CAR'] = 400

# Step 3: Select features
X_india = india[['CURR_AGE', 'GENDER', 'ANN_INCOME', 'AGE_CAR']]

# Step 4: Predict
india['PREDICTED_BUY'] = model.predict(X_india)

# Step 5: Count how many are predicted to buy
buyers_count = india['PREDICTED_BUY'].sum()
total_people = len(india)

print(f"\nTotal people in Indian data: {total_people}")
print(f"Predicted buyers: {buyers_count}")




avg_age_car = japan_df[japan_df['PURCHASE'] == 1]['AGE_CAR'].mean()
india['AGE_CAR'] = avg_age_car

X_india = india[['CURR_AGE', 'GENDER', 'ANN_INCOME', 'AGE_CAR']]
india['PREDICTED_BUY'] = model.predict(X_india)
buyers_count = india['PREDICTED_BUY'].sum()
print(f"Updated predicted buyers: {buyers_count}")

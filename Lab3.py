import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier

# Step 1 Data Collection
titanic_data = sns.load_dataset('titanic')
print(titanic_data.head())

# Step 2 Data Cleaning
missing_values = titanic_data.isnull().sum() # to sum all missing values in the diferent colums
print("Missing values in each column:\n", missing_values)

# Drop columns that have too many missing values (like 'deck')
titanic_data_cleaned = titanic_data.drop(columns=['deck'])

# Impute missing values in 'age' with the median
titanic_data_cleaned['age'] = titanic_data_cleaned['age'].fillna(titanic_data_cleaned['age'].median())

# Impute 'embark_town' with the most frequent value (mode)
titanic_data_cleaned['embark_town'] = titanic_data_cleaned['embark_town'].fillna(titanic_data_cleaned['embark_town'].mode()[0])

# Dropping rows with missing 'embarked' since they are not many
titanic_data_cleaned = titanic_data_cleaned.dropna(subset=['embarked'])

# Verify if all missing values have been remove or replace with approprite values
print("Missing values after cleaning:\n", titanic_data_cleaned.isnull().sum())

# Step 3 Handling Outliers
plt.figure(figsize=(10, 5))
# we plot both box plot on the same figeur and args define it position and size with respect to and aready existing plot
plt.subplot(1, 2, 1)
# Box plot to identify outlier for the column age
sns.boxplot(x=titanic_data_cleaned['age'])
plt.title('Age Boxplot')

plt.subplot(1, 2, 2)
# boc plot to identify outlier for the culum fare
sns.boxplot(x=titanic_data_cleaned['fare'])
plt.title('Fare Boxplot')

plt.show()

# Capping the outliers (for 'age' and 'fare')
# Define upper and lower caps for 'age' and 'fare'
upper_age = titanic_data_cleaned['age'].quantile(0.95)
lower_age = titanic_data_cleaned['age'].quantile(0.05)
upper_fare = titanic_data_cleaned['fare'].quantile(0.95)

# Cap 'age' and 'fare'
titanic_data_cleaned['age'] = titanic_data_cleaned['age'].apply(lambda x: upper_age if x > upper_age else (lower_age if x < lower_age else x))
titanic_data_cleaned['fare'] = titanic_data_cleaned['fare'].apply(lambda x: upper_fare if x > upper_fare else x)

# STep 4 Data Normalization

scaler = MinMaxScaler() # Min-Max scaling brings 'age' and 'fare' values into a range between 0 and 1 for normalizing it scale

# Normalize 'age' and 'fare'
titanic_data_cleaned[['age', 'fare']] = scaler.fit_transform(titanic_data_cleaned[['age', 'fare']])

# Check normalized values
print(titanic_data_cleaned[['age', 'fare']].head())

# Step 5 Feature Engineering

titanic_data_cleaned['family_size'] = titanic_data_cleaned['sibsp'] + titanic_data_cleaned['parch'] + 1 # create family_size column
# Check the new feature by printing it values
print(titanic_data_cleaned[['family_size']].head())

print(titanic_data_cleaned.info()) # to get info and which dtat to hot encode for the next step

# Step 6 Feature Selection
# One-hot encoding for categorical variables since corellation only work with numeric values
titanic_data_encoded = pd.get_dummies(titanic_data_cleaned,
                                      columns=['sex', 'embarked', 'class', 'who', 'embark_town', 'alive'],
                                      drop_first=True)
# Correlation analysis to select important features
plt.figure(figsize=(10, 6))
sns.heatmap(titanic_data_encoded.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show() # show corellation between the features

# Selecting features knowing ssurvived is our target
X = titanic_data_cleaned[['pclass', 'sex', 'age', 'fare', 'family_size', 'embarked']]
X = pd.get_dummies(X, drop_first=True)  # Convert categorical features to numerical
y = titanic_data_cleaned['survived']

# Fit a Random Forest to get feature importances
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Display feature importances
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh')
plt.title('Feature Importances')
plt.show()

# Step 7 Model Building

# Split the data into train and test sets (same for both models)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ----- Logistic Regression -----
# Initialize and fit Logistic Regression model
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

# Make predictions
y_pred_logreg = logreg.predict(X_test)

# Evaluate the Logistic Regression model
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
precision_logreg = precision_score(y_test, y_pred_logreg)
recall_logreg = recall_score(y_test, y_pred_logreg)
f1_logreg = f1_score(y_test, y_pred_logreg)

print("Logistic Regression Model Results:")
print(f"Accuracy: {accuracy_logreg:.2f}")
print(f"Precision: {precision_logreg:.2f}")
print(f"Recall: {recall_logreg:.2f}")
print(f"F1 Score: {f1_logreg:.2f}")
print("-" * 40)

# ----- Random Forest Classifier -----
# Initialize the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model on the training data
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluate the Random Forest model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

print("Random Forest Model Results:")
print(f"Accuracy: {accuracy_rf:.2f}")
print(f"Precision: {precision_rf:.2f}")
print(f"Recall: {recall_rf:.2f}")
print(f"F1 Score: {f1_rf:.2f}")
print("-" * 40)

# ----- Compare Precision Difference -----
precision_diff = precision_logreg - precision_rf
print(f"Difference in Precision (Logistic Regression - Random Forest): {precision_diff:.2f}")
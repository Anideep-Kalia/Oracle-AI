import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


iris_data=pd.read_csv('iris.csv')
iris_data.head()

X=iris_data.drop(columns=['Id','Species'])    # dropping columns Id and Species 
Y=iris_data['Species']                        # Selecting column Species only so these ar ethe only labels

# Split data into testing and training
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Standardised the features
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled= scaler.transform(X_test)

X.head()

# creating the model
model = LogisticRegression()

# train the model
model.fit(X_train_scaled,y_train)

# Evaluate the model on testing set
y_pred=model.predict(X_test_scaled)
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)

# Sample new data for predictions
new_data=mp.array([[5.1, 3.5, 1.4, 0.2],
                    [6.3, 2.9, 5.6, 1.8,],
                    [4.9, 3.0, 1.4, 0.2]])


# Standardised the new data
new_data_scaled=scaler.transform(new_data)

predictions=model.predict(new_data_scaled)


# print the result
print("Predictiosn: "predictions)


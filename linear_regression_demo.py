import pandas as pd
from sklearn.linear_model import LogisticRegression

iris_data=pd.read_csv('iris.csv')
iris_data.head()

X=iris_data.drop(columns=['Id','Species'])    # dropping columns Id and Species 
Y=iris_data['Species']                        # Selecting column Species only so these ar ethe only labels

X.head()

model = LogisticRegression()


# predict using the trained model
predictions=model.predict([[4.6, 3.5, 1.5, 0.2]])

# print the result
print(predictions)


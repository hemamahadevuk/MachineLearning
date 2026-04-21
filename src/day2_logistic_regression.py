import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/diabetes.csv")
y = df['Outcome']
df.drop(['Outcome'], axis=1, inplace=True)

X_train,X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(accuracy_score(y_test, predictions))

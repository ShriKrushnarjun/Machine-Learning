import pandas as pd
from sklearn.cluster import KMeans
from sklearn import cross_validation, preprocessing
import numpy as np

'''
Pclass Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
survival Survival (0 = No; 1 = Yes)
name Name
sex Sex
age Age
sibsp Number of Siblings/Spouses Aboard
parch Number of Parents/Children Aboard
ticket Ticket Number
fare Passenger Fare (British pound)
cabin Cabin
embarked Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
boat Lifeboat
body Body Identification Number
home.dest Home/Destination

Index(['pclass', 'survived', 'sex', 'age', 'sibsp', 'parch', 'ticket', 'fare',
       'cabin', 'embarked', 'boat', 'home.dest'],
'''

df = pd.read_excel("titanic.xls")
df.drop(["name", "body"], 1, inplace=True)
df.fillna(0, inplace=True)


# print(df.head())

def handle_non_neric_data(df):
    for column in df.columns.values:

        text_to_numeric = {}
        def convert_text_to_numeric(val):
            return text_to_numeric[val]

        column_content = set(df[column])

        x = 0
        for category in column_content:
            text_to_numeric[category] = x
            x += 1

        df[column] = list(map(convert_text_to_numeric, df[column]))

    return df


df = handle_non_neric_data(df)

df.drop(["boat"],1 ,inplace=True)

X = np.array(df.drop(["survived"], 1).astype(float))
y = np.array(df["survived"])

clf = KMeans(n_clusters=2)
clf.fit(X)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    for p in prediction:
        if p == y[i]:
            correct += 1

print("Accuracy:", correct / len(X))

actually_survived = 0
for i in y:
    if i == 1:
        actually_survived += 1
print("original survival rate:",actually_survived/len(y))
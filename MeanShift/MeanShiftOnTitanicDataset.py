import numpy as np
from sklearn.cluster import MeanShift
import pandas as pd

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


def handle_non_numeric_data(df):
    for column in df.columns.values:
        text_to_data = {}

        def text_to_data_conversion(val):
            return text_to_data[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:

            column_contents = df[column].values.tolist()

            unique_elements = set(column_contents)

            x = 0
            for unique in unique_elements:
                if unique not in text_to_data:
                    text_to_data[unique] = x
                    x += 1

            df[column] = (list(map(text_to_data_conversion, df[column])))
    return df


original_df = pd.read_excel("../titanic.xls")
# original_df.drop(["survived"], 1, inplace=True)
original_df.fillna(0, inplace=True)
new_df = handle_non_numeric_data(original_df)

X = np.array(new_df.drop(['body', 'name'], 1)).astype(float)

clf = MeanShift()
clf.fit(X)
centers = clf.cluster_centers_
labels = clf.labels_
n_clusters = len(set(labels))

original_df["cluster_group"] = np.nan

for i in range(len(original_df)):
    original_df["cluster_group"].iloc[i] = labels[i]

survivale_rates = {}

for i in range(n_clusters):
    temp_df = original_df[(original_df["cluster_group"] == i)]
    survival_cluster = temp_df[(temp_df["survived"] == 1)]
    survival_rate = len(survival_cluster) / len(temp_df)
    survivale_rates[i] = survival_rate

print(survivale_rates)

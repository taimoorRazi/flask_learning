import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

df = pd.read_excel("hiring_data.xlsx")

df["experience"].fillna(0,inplace=True)

df["test_score"].fillna(df["test_score"].mean(), inplace=True)

X= df.iloc[:,:3]
y=df.iloc[:, -1]

#Converting words to integer values
def convert_to_int(word):
    word_dict = {"one":1, "two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,
    "eight":8,"nine":9,"ten":10, "eleven":11, "twelve":12, "zero":0, 0:0}
    return word_dict[word]

X["experience"] = X["experience"].apply(lambda x : convert_to_int(x))

# splitting training and testing set
# Since we have a very small dataset, we will tarain our model with all available data.

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

# fitting model with training data
regressor.fit(X,y)

# saving the model to disk
pickle.dump(regressor, open("model.pkl", "wb"))

# loading the model to compare the results
model = pickle.load(open("model.pkl", "rb"))
print(model.predict([[2,9,6]]))
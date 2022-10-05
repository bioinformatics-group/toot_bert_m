from sklearn.linear_model import LogisticRegression
import pandas as pd
import pycm
import pyperclip
from matplotlib import pyplot as plt
import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from methods.latex import *
from constants import constants

path_representation = "./toot_bert_m/representations/"

# Load the data
df_test = pd.read_csv(path_representation + "test.csv")
df_train = pd.read_csv(path_representation + "train.csv")


# Print the shape of the dataframes
print("Shape of the train dataframe:", df_train.shape)
print("Shape of the test dataframe:", df_test.shape)

# Train the model with all the columns except the last one
X_train = df_train.iloc[:, :-1]
y_train = df_train.iloc[:, -1]

X_test = df_test.iloc[:, :-1]
y_test = df_test.iloc[:, -1].to_list()

model = LogisticRegression(random_state=1)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Make table with the results of tp,tn,fp,fn,f1,mcc from pycm
cm = pycm.ConfusionMatrix(y_test, y_pred, classes=["transporter", "nontransporter"])
from sklearn.metrics import roc_curve, auc
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import os
import pdb
import matplotlib.pyplot as plt

outputs = pd.read_csv('validation_outputs.csv')

mean_out = np.mean(outputs.values)
print(mean_out)
print(max(outputs.values.flatten()))
print(min(outputs.values.flatten()))
print(np.median(outputs.values.flatten()))

data_train = json.load(open('/Users/jiayi/1008 deep learning/data/train.json'))
df_train = pd.DataFrame.from_records(data_train["annotations"])
data_val = json.load(open('/Users/jiayi/1008 deep learning/data/validation.json'))
df_val = pd.DataFrame.from_records(data_val["annotations"])

val_size = 1984
val_ = df_val[:val_size]

mlb = MultiLabelBinarizer()
mlb = mlb.fit(df_train['labelId'])

labels = mlb.transform(val_['labelId'])

def compute_f1(endpoint,num_pts):
    thresholds = np.linspace(0,endpoint,num=num_pts)
    f1_all = []

    for t in thresholds:
        pred = outputs.gt(t)
        pred = pred.astype(int)
        tp = (pred + labels).eq(2).values.sum()
        fp = (pred - labels).eq(1).values.sum()
        fn = (pred - labels).eq(-1).values.sum()
        tn = (pred + labels).eq(0).values.sum()
        acc = (tp + tn) / (tp + tn + fp + fn)
        try:
            prec = tp / (tp + fp)
        except ZeroDivisionError:
            prec = 0.0
        try:
            rec = tp / (tp + fn)
        except ZeroDivisionError:
            rec = 0.0
        try:
            f1 = 2*(rec * prec) / (rec + prec)
        except ZeroDivisionError:
            f1 = 0.0
        f1_all.append(f1)
    return thresholds,f1_all


def plot_f1(thresholds,f1s):
    fig = plt.figure(1,figsize=(11,7))
    ax = plt.subplot(111)

    ax.grid()
    ax.set_title("F1 score under different thresholds")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("F1 score")

    plt.plot(thresholds,f1s)

    plt.show()



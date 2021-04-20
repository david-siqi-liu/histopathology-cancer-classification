from src.config import *

import random
import numpy as np
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt


def get_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Current device: {:s}".format(device.type))
    return device


def set_seed():
    seed = args['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def plot_statistics(measure, values):
    fig = plt.figure(figsize=(10, 5))
    plt.title("Measure: {}".format(measure))
    for label, value in values.items():
        plt.plot(value, label=label)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(measure)
    plt.legend(loc='best')


def plot_cm(df):
    fig = plt.figure(figsize=(6, 6))
    cm = pd.crosstab(df['label'], df['pred'],
                     rownames=['Actual'], colnames=['Predicted'])
    sns.heatmap(cm, annot=True)
    plt.show()

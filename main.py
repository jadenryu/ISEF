from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier, XGBRegressor
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv("data.csv")
print(data.columns)
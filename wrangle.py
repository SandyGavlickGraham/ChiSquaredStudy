import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler
import math

def wrangle_grades():
    grades = pd.read_csv("student_grades.csv")
    grades.drop(columns='student_id', inplace=True)
    grades.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    df = grades.dropna().astype('int')
    return df

def plot_grade_distributions(df):
    # histogram with subplots in matplot lib
    plt.figure(figsize=(16, 3))

    for i, col in enumerate(['exam1', 'exam2', 'exam3', 'final_grade']):  
        plot_number = i + 1 # i starts at 0, but plot nos should start at 1
        series = df[col]  
        plt.subplot(1,4, plot_number)
        plt.title(col)
        series.hist(bins=5, density=False, cumulative=False, log=False)
    
    # we don't want to plot the `student_id` column.
    plt.figure(figsize=(8,4))
    sns.boxplot(data=df)
    plt.show()

def split_my_data(data):
    '''the function will take a dataframe and returns train and test dataframe split 
    where 80% is in train, and 20% in test. '''
    return train_test_split(data, train_size = .80, random_state = 123)

def standard_scaler(train, test):
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(train) # fit the object
    train = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    test = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return scaler, train, test


# If we wanted to return to original values:
def my_inv_transform(scaler, train, test):
    train = pd.DataFrame(scaler.inverse_transform(train), columns=train.columns.values).set_index([train.index.values])
    test = pd.DataFrame(scaler.inverse_transform(test), columns=test.columns.values).set_index([test.index.values])
    return scaler, train, test


def uniform_scaler(train, test):
    scaler = QuantileTransformer(n_quantiles=100, output_distribution='uniform', random_state=123, copy=True).fit(train)
    train = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    test = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return scaler, train, test


def gaussian_scaler(train, test, method='yeo-johnson'):
    scaler = PowerTransformer(method, standardize=False, copy=True).fit(train)
    train = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    test = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return scaler, train, test


def my_minmax_scaler(train, test, minmax_range=(0,1)):
    scaler = MinMaxScaler(copy=True, feature_range=minmax_range).fit(train)
    train = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    test = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return scaler, train, test


def iqr_robust_scaler(train, test):
    scaler = RobustScaler(quantile_range=(25.0,75.0), copy=True, with_centering=True, with_scaling=True).fit(train)
    train = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
    test = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])
    return scaler, train, test


if __name__ == '__main__':
    df = wrangle_grades()
    plot_grade_distributions(df)
    train, test = split_my_data(data=df)
    print('train shape:', train.shape)
    print('test shape:', test.shape)
    scaler, train_scaled, test_scaled = standard_scaler(train, test)
    print("Mean:") 
    print(scaler.mean_)
    print("Standard Deviation:")
    print([math.sqrt(i) for i in scaler.var_])


# train.head()

# scaler, train, test = my_inv_transform(scaler, train_scaled, test_scaled)
# train.head()

# scaler, train_scaled, test_scaled = uniform_scaler(train, test)
# train_scaled.head()

# scaler, train_scaled, test_scaled = gaussian_scaler(train, test, method='yeo-johnson')
# train_scaled.head()

# scaler, train_scaled, test_scaled = my_minmax_scaler(train, test)
# train_scaled.head()

# scaler, train_scaled, test_scaled = iqr_robust_scaler(train, test)
# train_scaled.head()
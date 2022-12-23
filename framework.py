import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import warnings
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error as mae
warnings.filterwarnings("ignore")

def read_csv(path):
    df = pd.read_csv(path)
    # removing the duplicate values from the df 
    print(df[df.duplicated()])
    df = df.drop_duplicates(keep='first')
    df = df.reset_index(drop = True)
    print('Number of rows',df.shape[0])
    print('number of columns ',df.shape[1])
    return df

def univariate_categorial_analysis(df, categorical_columns):
     for col in categorical_columns:
        sns.countplot(data=df, x=col, order = np.sort(df[col].unique()))
        plt.show()
        
def bivariate_categorial_analysis(df, categorical_columns, hue):
     for col in categorical_columns:
        sns.countplot(data=df, x=col, order = np.sort(df[col].unique()), hue = hue, palette = ['green', 'red'])
        plt.show()
        
        
def bivariate_analysis_numerical(df, columns,output, output_1, output_2):
    for col in columns:
        sns.distplot(df[df[output] == output_1][col], color='green',kde=True) 
        sns.distplot(df[df[output] == output_2][col], color='red',kde=True)
        plt.legend([output_1, output_2])
        plt.show()


def univariate_numerical_analysis(df, continous_columns):
    for col in continous_columns:
        fig, (ax1, ax2) = plt.subplots(1,2, figsize = (15,7))
        ax1.hist(df[col])
        sns.boxplot(ax = ax2, x = col, data =df)
        ax1.set_xlabel(col)
        plt.show()
        
# statistical tests for classification based problems

def chi_squared_test(column, df, output, alpha = 0.05):
    probability = 1- alpha
    contingency = pd.crosstab(df[column], df[output])
    stat, p, dof, expected = scipy.stats.chi2_contingency(contingency)
    critical = scipy.stats.chi2.ppf(probability, dof)
    if abs(stat) >= critical:
        None
    else:
        print(column, ':Independent (fail to reject H0)', stat, critical)
        
def t_tests(column, df, output,output_1, output_2,alpha = 0.05):
    value = scipy.stats.ttest_ind(df[column][df[output] == output_1],
                df[column][df[output] == output_2])
    p_value = value[1]
    
    if abs(p_value) <= alpha:
        None
    else:
        print(column, ':similar means between the two groups (fail to reject H0)',p_value)
        
        
def wilcoxon(column, df, output,output_1, output_2,alpha = 0.05):
    res = scipy.stats.wilcoxon(df[column][df[output] == output_1],
                df[column][df[output] == output_2])
    p_value = res.pvalue
    
    if abs(p_value) <= alpha:
        None
    else:
        print(column, ':similar means between the two groups (fail to reject H0)',p_value)
        
        
def mannwhitneyu(column, df, output,output_1, output_2,alpha = 0.05):
    value = scipy.stats.mannwhitneyu(df[column][df[output] == output_1],
                df[column][df[output] == output_2])
    p_value = value[1]
    
    if abs(p_value) <= alpha:
        None
    else:
        print(column, ':similar means between the two groups (fail to reject H0)',p_value)
        
def one_hot_encoding(df, columns):
    df = pd.get_dummies(df, columns = columns)
    return df

def standard_scaler(df, columns):
    df[columns] =  StandardScaler().fit_transform(df[columns])
    return df

from sklearn.preprocessing import PowerTransformer

def power_transform(df, columns):
    df[columns] =  PowerTransformer().fit_transform(df[columns])
    return df

def train_test_split(X,y, test_ratio = 0.3, stratify = True):
    X_train, X_test, y_train, y_test = tts(X,y, test_size=test_ratio, random_state=42, stratify = y)
    return (X_train, X_test, y_train, y_test)

def score(name, ClassifierClass,X_train, X_test, y_train, y_test, names, classifier_scores, acc = True):
    clf = ClassifierClass()
    clf.fit(X_train,y_train)
    if acc:
        score = accuracy_score(y_test,clf.predict(X_test))
    else:
        score =  mae(y_test,clf.predict(X_test))
    names.append(name)
    classifier_scores.append(score)
    return (names, classifier_scores)
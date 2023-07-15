# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 06:08:06 2022

@author: User
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv(
    "E:\\BITS_STUDY\\Section 1\\Data Mining\\Assignment\\cc_fraud\\card_transdata.csv")
df.shape
df.head()

# Checking any null/missing value is available or not
df.isnull().any()

# Counting how many ecentage of fraud transaction is posted
# As number of fraud tansaction are more  so this data set is not imbalance dataset
df['fraud'].value_counts()
total_fraud = df['fraud'].value_counts(normalize=True)
print(total_fraud)
# finding mean,std,min,25%,75% and max value for each feature.
df.describe()

# from above result it can be obseve that outlier may be available in 'distance_from_home','distance_from_last_transaction' and 'ratio_to_median_purchase_price'
# Plotting scatter plot for above features
plt.scatter(df['distance_from_home'], df['distance_from_home'])
plt.xlabel("distance_from_home")
plt.ylabel("distance_from_home")
plt.show()

plt.scatter(df['distance_from_last_transaction'],
            df['distance_from_last_transaction'])
plt.xlabel("distance_from_last_transaction")
plt.ylabel("distance_from_last_transaction")
plt.show()

plt.scatter(df['ratio_to_median_purchase_price'],
            df['ratio_to_median_purchase_price'])
plt.xlabel("ratio_to_median_purchase_price")
plt.ylabel("ratio_to_median_purchase_price")
plt.show()

# from above graph it can be observed that outlier present in above feature let's remove outlier
sns.boxplot(df['distance_from_home'])
sns.boxplot(df['distance_from_last_transaction'])
sns.boxplot(df['ratio_to_median_purchase_price'])


# from above graph it can be observed that outlier present in above feature let's remove outlier

Q1 = np.percentile(df['distance_from_last_transaction'], 25,
                   interpolation='midpoint')

Q3 = np.percentile(df['distance_from_last_transaction'], 75,
                   interpolation='midpoint')
IQR = Q3 - Q1

print("Old Shape: ", df.shape)

# Upper bound
upper = np.where(df['distance_from_last_transaction'] >= (Q3+1.5*IQR))
# Lower bound
lower = np.where(df['distance_from_last_transaction'] <= (Q1-1.5*IQR))

''' Removing the Outliers '''
df.drop(upper[0], inplace=True)
df.drop(lower[0], inplace=True)

print("New Shape: ", df.shape)


X = df.iloc[:, :-1].values
y = df.iloc[:, 7].values


# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)


print("************************Analysis using Linear Regession **************************************")
# predicting values
#from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred_linear = regressor.predict(X_test)
print(y_pred_linear)

cutoff = 0.7
y_pred_classes = np.zeros_like(y_pred_linear)
y_pred_classes[y_pred_linear > cutoff] = 1

# you have to do the same for the actual values too:

y_test_classes = np.zeros_like(y_pred_linear)
y_test_classes[y_test > cutoff] = 1


confusion_matrix = confusion_matrix(y_test_classes, y_pred_classes)
print(confusion_matrix)

print('Precision: %.3f' % precision_score(y_test_classes, y_pred_classes))
print('Recall: %.3f' % recall_score(y_test_classes, y_pred_classes))
print('Accuracy: %.3f' % accuracy_score(y_test_classes, y_pred_classes))
print('F1 Score: %.3f' % f1_score(y_test_classes, y_pred_classes))


print("************************Analysis using Logistic Regession **************************************")
# predicting values
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()
print(logmodel.fit(X_train, y_train))


# Predicting
y_pred_logistic = logmodel.predict(X_test)
print("sau")
print(y_pred_logistic)


confusion_matrix = confusion_matrix(y_test, y_pred_logistic)
print(confusion_matrix)


print('Precision: %.3f' % precision_score(y_test, y_pred_logistic))
print('Recall: %.3f' % recall_score(y_test, y_pred_logistic))
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred_logistic))
print('F1 Score: %.3f' % f1_score(y_test, y_pred_logistic))


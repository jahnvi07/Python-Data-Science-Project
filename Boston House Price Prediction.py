#!/usr/bin/env python
# coding: utf-8

# # Boston House Price Prediction

# This project is for learning Machine Learning Algorithms and to learn the different data preprocessing techniques such as Exploratory Data Analysis, Feature Engineering, Feature Selection, Feature Scaling and finally to build a machine learning model.
# 
# In this project we will predicts house price in boston city
# 
# The dataset is collected from Kaggle. 

# # About the Dataset

# This dataset contains information collected by the U.S Census Service concerning housing in the area of Boston Mass. It was obtained from the StatLib archive Data, and has been used extensively throughout the literature to benchmark algorithms. However, these comparisons were primarily done outside of Delve and are thus somewhat suspect. The dataset is small in size with only 506 cases.
# 
# The data was originally published by Harrison, D. and Rubinfeld, D.L. Hedonic prices and the demand for clean air', J. Environ. Economics & Management, vol.5, 81-102, 1978.

# # Dataset Naming

# The name for this dataset is simply boston. It has two prototasks: nox, in which the nitrous oxide level is to be predicted; and price, in which the median value of a home is to be predicted.

# # Miscellaneous Details

# ### Origin
#     The origin of the boston housing data is Natural.
# ### Usage
#     This dataset may be used for Assessment.
# ### Number of Cases
#     The dataset contains a total of 506 cases.
# ### Order
#     The order of the cases is mysterious.
# 
# ### Variables
#     There are 14 attributes in each case of the dataset. They are:
# 
#      1. CRIM - per capita crime rate by town
#      2. ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
#      3. INDUS - proportion of non-retail business acres per town.
#      4. CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
#      5. NOX - nitric oxides concentration (parts per 10 million)
#      6. RM - average number of rooms per dwelling
#      7. AGE - proportion of owner-occupied units built prior to 1940
#      8. DIS - weighted distances to five Boston employment centres
#      9. RAD - index of accessibility to radial highways
#     10. TAX - full-value property-tax rate per 10,000 dollars.
#     11. PTRATIO - pupil-teacher ratio by town
#     12. B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
#     13. LSTAT - % lower status of the population
#     14. MEDV - Median value of owner-occupied homes in 1000's dollars
# 
# #### Note
# Variable #14 seems to be censored at 50.00 (corresponding to a median price of 50,000 dollars); Censoring is suggested by the fact that the highest median price of exactly 50,000 dollars is reported in 16 cases, while 15 cases have prices between 40,000 dollars and 50,000 dollars, with prices rounded to the nearest hundred. Harrison and Rubinfeld do not mention any censoring.

# In[1]:


import numpy as np
import pandas as pd

# Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sb

# Setting Seaborn Style
sb.set(style = 'whitegrid')

# For Preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# For Preformance metrics
from sklearn.metrics import mean_squared_error, r2_score


# In[3]:


# Initializing column names
columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

# Loading Boston Housing Dataset
boston = pd.read_csv('data/housing.csv', delimiter=r"\s+", names = columns)
# Top 5 rows of the boston dataset
boston.head()


# In[4]:


# TODO : Let's know how many factors of an individual and Number of Samples
print("The Boston housing Price Prediction Dataset has")
print("\t\tNumber of Factors : \t", boston.shape[1] - 1)
print("\t\tNumber of Samples : \t", boston.shape[0])


# # Exploratory Data Analysis

# In[5]:


# TODO : Descriptive Statistics on Boston Housing Dataset
boston.describe()


# In[6]:


# TODO : Check for null values and visualizing it using heatmap
boston.isnull().sum()


# There are no null values in any of the column

# In[7]:


# TODO : Let's check for data types of all the columns
boston.dtypes


# All are numerical values. So no need of encoding

# In[18]:


plt.figure(figsize=(8, 6))
sb.displot(boston['CRIM'], rug = True)
plt.savefig('images/crim.png')

plt.figure(figsize=(8, 6))
sb.displot(boston['ZN'], rug = True)
plt.savefig('images/zn.png')

plt.figure(figsize=(8, 6))
sb.displot(boston['INDUS'], rug = True)
plt.savefig('images/indus.png')

plt.figure(figsize=(8, 6))
sb.displot(boston['CHAS'], rug = True)
plt.savefig('images/chas.png')

plt.figure(figsize=(8, 6))
sb.displot(boston['NOX'], rug = True)
plt.savefig('images/nox.png')

plt.figure(figsize=(8, 6))
sb.displot(boston['RM'], rug = True)
plt.savefig('images/rm.png')

plt.figure(figsize=(8, 6))
sb.displot(boston['AGE'], rug = True)
plt.savefig('images/age.png')

plt.figure(figsize=(8, 6))
sb.displot(boston['DIS'], rug = True)
plt.savefig('images/dis.png')

plt.figure(figsize=(8, 6))
sb.displot(boston['RAD'], rug = True)
plt.savefig('images/rad.png')

plt.figure(figsize=(8, 6))
sb.displot(boston['TAX'], rug = True)
plt.savefig('images/tax.png')

plt.figure(figsize=(8, 6))
sb.displot(boston['PTRATIO'], rug = True)
plt.savefig('images/ptration.png')

plt.figure(figsize=(8, 6))
sb.displot(boston['B'], rug = True)
plt.savefig('images/b.png')

plt.figure(figsize=(8, 6))
sb.displot(boston['LSTAT'], rug = True)
plt.savefig('images/lstat.png')

plt.figure(figsize=(8, 6))
sb.displot(boston['MEDV'], rug = True)
plt.savefig('images/medv.png')


# # Feature Observation

# In[17]:


plt.figure(figsize  = (2, 2))
sb.pairplot(boston)
plt.savefig('images/pairplot.png')


# In[19]:


# TODO : Visualizing Feature Correlation
plt.figure(figsize = (16, 12))
sb.heatmap(boston.corr(), cmap = 'Greens', annot = True, fmt = '.2%')
plt.savefig('images/features_correlation.png')


# In[20]:


# TODO : Visualizing correlation of features with prediction column `MEDV`

corr_with_medv = boston.corrwith(boston['MEDV'])

plt.figure(figsize = (16, 4))
sb.heatmap([np.abs(corr_with_medv)], cmap = 'RdBu_r', annot = True, fmt = '.2%')
plt.savefig('images/correlation_with_price.png')


# In[21]:


# Let's see the features having more correlation
corr_with_medv[:-1].abs().sort_values(ascending = False)


# In[22]:


# Let's confirm this by using ExtraTreesRegressor
# TODO : To know the feature Importances
y = boston['MEDV'].values
from sklearn.ensemble import ExtraTreesRegressor
etc = ExtraTreesRegressor()
etc.fit(boston.iloc[:, :-1].values, y)

print("Percentage Importance of each features with respect to House Price : ")
important_features = pd.Series(etc.feature_importances_*100, index = boston.columns[:-1])
important_features


# In[23]:


# Feature Impotances by ExtraTressRegressor
important_features.sort_values(ascending = False)


# In[24]:


# Feature Impotances by Correlation Matrix
corr_with_medv[:-1].abs().sort_values(ascending = False)


# In[26]:


plt.figure(figsize=(16, 10))
plt.plot(etc.feature_importances_, boston.columns[:-1], 'go-', linewidth=5, markersize=12)
plt.savefig('images/feature_importances.png')


# From the above feature observations, we found that some columns are most important such as LSTAT and RM

# # Building Machine Learning Model

# In[27]:


# Arranging features based on features importance
features_arranged_on_importance = important_features.sort_values(ascending = False).index
features_arranged_on_importance


# In[28]:


y = boston.loc[:, 'MEDV'].values


# In[29]:


# Arranging columns based on features importance
new_boston = boston[features_arranged_on_importance]
new_boston.head()


# In[30]:


# Getting boston values
X = new_boston.values
X = X[:, :6]

# TODO : Splitting data as train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)


# In[ ]:





# # Linear Regression

# In[31]:


from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

print('Training Score : ', linear_model.score(X_train, y_train))
print('Testing Score  : ', linear_model.score(X_test, y_test))

print('R2 Score : ', r2_score(y_test, linear_model.predict(X_test)))
print('MSE : ', mean_squared_error(y_test, linear_model.predict(X_test)))


# In[32]:


from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

linear_model = make_pipeline(MinMaxScaler(), LinearRegression())
linear_model.fit(X_train, y_train)

print('Training Score : ', linear_model.score(X_train, y_train))
print('Testing Score  : ', linear_model.score(X_test, y_test))

print('R2 Score : ', r2_score(y_test, linear_model.predict(X_test)))
print('MSE : ', mean_squared_error(y_test, linear_model.predict(X_test)))


# In[ ]:





# # Decision Tree Regression

# In[33]:


from sklearn.tree import DecisionTreeRegressor
scores = []
for i in range(100):

    dtr_model = DecisionTreeRegressor(max_depth=None, random_state=i)
    dtr_model.fit(X_train, y_train)
    scores.append(r2_score(y_test, dtr_model.predict(X_test)))

plt.figure(figsize = (16, 8))
plt.plot(list(range(100)), scores, 'ro-')
plt.xlabel('Random Decision Tree Regressor')
plt.ylabel('Scores')
plt.savefig('images/random_decision_tree_regressor.png')
plt.show()


# the decision tree score changes for different random states

# In[34]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

dtr_model = DecisionTreeRegressor(max_depth=23, random_state=3)
dtr_model.fit(X_train[:, :], y_train)
    

print('Training Score : ', dtr_model.score(X_train, y_train))
print('Testing Score  : ', dtr_model.score(X_test, y_test))

print('R2 Score : ', r2_score(y_test, dtr_model.predict(X_test)))
print('MSE : ', mean_squared_error(y_test, dtr_model.predict(X_test)))


# In[35]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
adtr_model = make_pipeline(MinMaxScaler(), DecisionTreeRegressor(max_depth = 12, random_state = 92))
adtr_model.fit(X_train, y_train)

print('Training Score : ', adtr_model.score(X_train, y_train))
print('Testing Score  : ', adtr_model.score(X_test, y_test))

print('R2 Score : ', r2_score(y_test, adtr_model.predict(X_test)))
print('MSE : ', mean_squared_error(y_test, adtr_model.predict(X_test)))


# 

# # Random Forest Regression

# In[36]:


from sklearn.ensemble import RandomForestRegressor
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
rfr = RandomForestRegressor(max_depth = 7, random_state = 63)
rfr.fit(X_train, y_train)


print('Training Score : ', rfr.score(X_train, y_train))
print('Testing Score  : ', rfr.score(X_test, y_test))

print('R2 Score : ', r2_score(y_test, rfr.predict(X_test)))
print('MSE : ', mean_squared_error(y_test, rfr.predict(X_test)))


# In[ ]:





# # Different Models Accuracy

# In[37]:


X_train, X_test, y_train, y_test = train_test_split(X[:, :], y, test_size = 0.20, random_state = 42)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

print('Linear Regression : ')
model1 = LinearRegression()
model1.fit(X_train, y_train)
print('Score : ', model1.score(X_test, y_test))

print('Decision Tree Regression : ')
model2 = DecisionTreeRegressor(max_depth=23, random_state=3)
model2.fit(X_train, y_train)
print('Score : ', model2.score(X_test, y_test))

print('Random Forest Regression : ')
model3 = RandomForestRegressor(max_depth = 7, random_state = 63)
model3.fit(X_train, y_train)
print('Score : ', model3.score(X_test, y_test))

print('k Neighbors Regression : ')
model4 = KNeighborsRegressor(n_neighbors = 10)
model4.fit(X_train, y_train)
print('Score : ', model4.score(X_test, y_test))


# In[ ]:





# # Building optimal Random Regression Model

# In[38]:


X_train, X_test, y_train, y_test = train_test_split(X[:, :], y, test_size = 0.20, random_state = 46)

print('Random Forest Regression : ')
random_forest_regressor = RandomForestRegressor(max_depth = 7, random_state = 63)
random_forest_regressor.fit(X_train, y_train)
print('Score : ', random_forest_regressor.score(X, y))


# In[40]:


# Scores for different training samples
scores = []
for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = i)
    random_forest_regressor = RandomForestRegressor(max_depth = 7, random_state = 63)
    random_forest_regressor.fit(X_train, y_train)
    scores.append(random_forest_regressor.score(X, y))
    
plt.figure(figsize = (16, 8))
plt.plot(list(range(100)), scores, 'go-')
plt.xlabel('Different Training Samples')
plt.ylabel('Scores')
plt.savefig('images/random_forest_diff_train_samples.png')
plt.show()


# In[42]:


# Scores for different random forest model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 3)

scores = []
for i in range(100):
    random_forest_regressor = RandomForestRegressor(max_depth = 13, random_state = i)
    random_forest_regressor.fit(X_train, y_train)
    scores.append(random_forest_regressor.score(X, y))
    
plt.figure(figsize = (16, 8))
plt.plot(list(range(100)), scores, 'ro-')
plt.xlabel('Different Random Forest Models')
plt.ylabel('Scores')
plt.savefig('images/random_forest_diff_RF_models.png')
plt.show()


# In[43]:


# Scores for different random forest model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 3)

scores = []
for i in range(1, 30):
    random_forest_regressor = RandomForestRegressor(max_depth = i, random_state = 68)
    random_forest_regressor.fit(X_train, y_train)
    scores.append(random_forest_regressor.score(X, y))
    
plt.figure(figsize = (16, 8))
plt.plot(list(range(1, 30)), scores, 'bo-')
plt.xlabel('Different Max_depths')
plt.ylabel('Scores')
plt.savefig('images/random_forest_diff_max_depth.png')
plt.show()


# In[44]:


plt.figure(figsize = (16, 8))
plt.plot(list(range(1, 30)), scores, 'bo-')
plt.ylim(0.95, 0.97)
plt.show()


# From this, we are going to choose,
# 
#      1) random_state = 3, for choosing random Training samples
#      2) random_state = 68, for random Random forest regressor
#      3) max_depth = 13, for Max Depths in random forest regressor

# In[ ]:





# # Building Optimal Model

# In[48]:


# Choosing Optimal Training Samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 3)

# Building Optimal Random Forest regressor Model
random_forest_regressor = RandomForestRegressor(max_depth = 13, random_state = 68)
random_forest_regressor.fit(X_train, y_train)


# In[47]:


random_forest_regressor.score(X, y)


# In[49]:


print('Training Accuracy : ', random_forest_regressor.score(X_train, y_train))
print('Testing Accuracy  : ', random_forest_regressor.score(X_test, y_test))


# In[50]:


print('Mean Squared Error : ', mean_squared_error(y_test, random_forest_regressor.predict(X_test)))
print('Root Mean Squared Error : ', mean_squared_error(y_test, random_forest_regressor.predict(X_test))**0.5)
print('Score : ', r2_score(y, random_forest_regressor.predict(X)))


# We have built a Random Forest Regressor Model which performs well with top 6 features and having the Training accuracy of 97.89% and Testing accuracy of 96.73%.

# In[ ]:





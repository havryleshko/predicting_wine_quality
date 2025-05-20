import pandas as pd
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

#loading the data
df_red = pd.read_csv('/Users/ohavryleshko/Documents/GitHub/WineQuality/wine+quality/winequality-red.csv', sep=';')
df_white = pd.read_csv('/Users/ohavryleshko/Documents/GitHub/WineQuality/wine+quality/winequality-white.csv', sep=';')
#with open("/Users/ohavryleshko/Documents/GitHub/WineQuality/wine+quality/winequality.names", 'r') as f:
#  names_text = f.read()
#print(names_text)

#first I am building a workflow for red wine
#inspecting the data
warnings.filterwarnings('ignore')
print('Printing first 10 columns of the data for red wine...')
print(df_red.head(10))
print('Describing shape of red wine dataset...')
print(df_red.shape)

#EDA (red)
print(df_red.describe().transpose())
print(df_red.corr()['quality'].sort_values(ascending=False))
print(df_red.isnull().sum())

#skews computation and visualisation
skews = df_red.drop('quality', axis=1).skew()
print(f'Sorted skewed values:\n', skews.sort_values(ascending=False))

skewed_features = ['chlorides', 'residual sugar', 'sulphates', 'total sulfur dioxide', 'free sulfur dioxide']
for f in skewed_features:
    pos_neg = (df_red[f].sum())
    if pos_neg > 0:
        df_red[f + '_l'] = np.log(df_red[f] + 1)
    else:
        df_red[f] = np.log(df_red[f])

print(df_red[[f + '_l' for f in skewed_features]].head(10))
df_red['chlorides'].hist(bins=30, color='blue', edgecolor='green') #histogram for top-skewed feature: chlorides
plt.title('Histogram of chlorides')
plt.xlabel('Chlorides')
plt.ylabel('Frequency')
plt.show()

df_red['residual sugar'].hist(bins=20, color='blue', edgecolor='green') #histogram for second skewed feature: residual sugar
plt.title('Histogram of residual sugar')
plt.xlabel('Residual sugar')
plt.ylabel('Frequency')
plt.show()

df_red['sulphates'].hist(bins=25, color='blue', edgecolor='green')
plt.title('Histogram of sulphates')
plt.xlabel('Sulphates')
plt.ylabel('Frequency')
plt.show()

df_red['total sulfur dioxide'].hist(bins=20, color='blue', edgecolor='green') #histogram for third skewed feature: total sulfur dioxide
plt.title('Histogram of total sulfur dioxide')
plt.xlabel('Total sulfur dioxide')
plt.ylabel('Frequency')
plt.show()

df_red['free sulfur dioxide'].hist(bins=20, color='blue', edgecolor='green') #histogram for fourth skewed feature: free sulfur dioxide
plt.title('Histogram of total free sulfur dioxide')
plt.xlabel('Free sulfur dioxide')
plt.ylabel('Frequency')
plt.show()


#visualising the data
top_5 = ['alcohol', 'sulphates', 'citric acid', 'fixed acidity', 'residual sugar'] #histogram
df_red[top_5].hist(bins=5, figsize=(10, 6))
plt.suptitle('Distributions of the top 5 features')
plt.show()

plt.figure(figsize=(8,4)) #boxplot
sns.boxplot(x='quality', y='alcohol', data=df_red)
plt.title('Comparison top feature to target')
plt.xlabel('Quality score')
plt.ylabel('Alcohol amount')
plt.show()

corr = df_red.corr() #heatmap
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.xlabel('Alcohol percentage')
plt.ylabel('Quality')
plt.show()

print('EDA complete')

#feature engineering
feature_interactions = [
    ('alcohol', 'sulphates'),
    ('pH', 'total sulfur dioxide'),
    ('sulphates', 'fixed acidity'),
    ('alcohol', 'citric acid'),
    ('pH', 'free sulfur dioxide')
] # feature interactions I think have the strongest correlation with regards to quality

for f1, f2 in feature_interactions:
    df_red[f'{f1}_and_{f2}'] = df_red[f1] * df_red[f2] #feature interactions loop to iterate over every pair of features and then create a new, possibly correlated one

corr_mat = df_red.corr().abs() # creating correlation matrix to remove correlation between the most similar feature
upper_triangle = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool)) # I try to take only upper part of matrix to avoid duplication
dropping = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.9)] # checking high correlation and if so, dropping it
df_red_dropped = df_red.drop(columns=dropping)
print("Dropping highly correlated features:", dropping)

#model building
features = df_red_dropped.drop(columns=['quality'])
target = df_red_dropped['quality']

X = features
y = target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#linear regression - first try
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred = lr.predict(X_test_scaled)

mse_lr = mean_squared_error(y_test, y_pred)
r2_lr = r2_score(y_test, y_pred)

print('Mean squared error for LR:', mse_lr)
print('R2 score for LR:', r2_lr)

#random forest regressor - second try
rfr = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 3, 5]
}
gs = GridSearchCV(rfr, param_grid, scoring='r2', cv=5)
gs.fit(X_train_scaled, y_train)
best_model = gs.best_estimator_
y_pred = best_model.predict(X_test_scaled)

mse_rfr = mean_squared_error(y_test, y_pred)
r2_rfr = r2_score(y_test, y_pred)
print('Best params:', gs.best_params_)
print('Mean squared error for RFR with gs:', mse_rfr)
print('R2 score for RFR with gs:', r2_rfr)

#visual model comparison table
results_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest'],
    'MSE': [mse_lr, mse_rfr],
    'R2': [r2_lr, r2_rfr]
})
print('Model performance comparison:')
print(results_df)

plt.figure(figsize=(10, 6)) #visual for R2 comparison
sns.barplot(x='Model', y='R2', data=results_df, palette='viridis')
plt.title('Model performance comparison - R2')
plt.xlabel('Model type')
plt.ylabel('R2 score')
plt.show()

plt.figure(figsize=(10, 6)) #visual for MSE comparison
sns.barplot(x='Model', y='MSE', data=results_df, palette='viridis')
plt.title('Model performance comparison - MSE')
plt.xlabel('Model type')
plt.ylabel('MSE')
plt.show()


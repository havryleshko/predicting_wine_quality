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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

#loading the data
df_red = pd.read_csv('/Users/ohavryleshko/Documents/GitHub/WineQuality/wine+quality/winequality-red.csv', sep=';')
df_white = pd.read_csv('/Users/ohavryleshko/Documents/GitHub/WineQuality/wine+quality/winequality-white.csv', sep=';')
#with open("/Users/ohavryleshko/Documents/GitHub/WineQuality/wine+quality/winequality.names", 'r') as f:
#  names_text = f.read()
#print(names_text)

#first I am building a workflow for RED WINE
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

#skews computation and visualisation (red wine)
skews = df_red.drop('quality', axis=1).skew()
print(f'Sorted skewed values (red wine):\n', skews.sort_values(ascending=False))

skewed_features = ['chlorides', 'residual sugar', 'sulphates', 'total sulfur dioxide', 'free sulfur dioxide']
for f in skewed_features:
    pos_neg = (df_red[f].sum())
    if pos_neg > 0:
        df_red[f + '_l'] = np.log(df_red[f] + 1)
    else:
        df_red[f] = np.log(df_red[f])

print(df_red[[f + '_l' for f in skewed_features]].head(10))
df_red['chlorides'].hist(bins=30, color='blue', edgecolor='green') #histogram for top-skewed feature: chlorides
plt.title('Histogram of chlorides (red wine)')
plt.xlabel('Chlorides')
plt.ylabel('Frequency')
plt.show()

df_red['residual sugar'].hist(bins=20, color='blue', edgecolor='green') #histogram for second skewed feature: residual sugar
plt.title('Histogram of residual sugar (red wine)')
plt.xlabel('Residual sugar')
plt.ylabel('Frequency')
plt.show()

df_red['sulphates'].hist(bins=25, color='blue', edgecolor='green')
plt.title('Histogram of sulphates (red wine)')
plt.xlabel('Sulphates')
plt.ylabel('Frequency')
plt.show()

df_red['total sulfur dioxide'].hist(bins=20, color='blue', edgecolor='green') #histogram for third skewed feature: total sulfur dioxide
plt.title('Histogram of total sulfur dioxide (red wine)')
plt.xlabel('Total sulfur dioxide')
plt.ylabel('Frequency')
plt.show()

df_red['free sulfur dioxide'].hist(bins=20, color='blue', edgecolor='green') #histogram for fourth skewed feature: free sulfur dioxide
plt.title('Histogram of total free sulfur dioxide (red wine)')
plt.xlabel('Free sulfur dioxide')
plt.ylabel('Frequency')
plt.show()

#visualising the data
top_5 = ['alcohol', 'sulphates', 'citric acid', 'fixed acidity', 'residual sugar'] #histogram
df_red[top_5].hist(bins=5, figsize=(10, 6))
plt.suptitle('Distributions of the top 5 features (red wine)')
plt.show()

plt.figure(figsize=(8,4)) #boxplot
sns.boxplot(x='quality', y='alcohol', data=df_red)
plt.title('Comparison top feature to target (red wine)')
plt.xlabel('Quality score')
plt.ylabel('Alcohol amount')
plt.show()

corr = df_red.corr() #heatmap
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.xlabel('Alcohol percentage (red wine)')
plt.ylabel('Quality')
plt.show()

print('EDA for red wine complete')

#then EDA for WHITE WINE

warnings.filterwarnings('ignore')
print('Printing first 10 columns of the data for white wine...')
print(df_white.head(10))
print('Describing shape of white wine dataset...')
print(df_white.shape)

#EDA (white)
print(df_white.describe().transpose())
print(df_white.corr()['quality'].sort_values(ascending=False))
print(df_white.isnull().sum())

#skews computation and visualisation (white wine)
skews_white = df_white.drop('quality', axis=1).skew()
print(f'Sorted skewed values (white wine):\n', skews_white.sort_values(ascending=False))

skewed_features = ['chlorides', 'residual sugar', 'sulphates', 'total sulfur dioxide', 'free sulfur dioxide']
for f in skewed_features:
    pos_neg = (df_white[f].sum())
    if pos_neg > 0:
        df_white[f + '_l'] = np.log(df_white[f] + 1)
    else:
        df_white[f] = np.log(df_white[f])

print(df_white[[f + '_l' for f in skewed_features]].head(10))
df_white['chlorides'].hist(bins=30, color='blue', edgecolor='green') #histogram for top-skewed feature: chlorides
plt.title('Histogram of chlorides (white wine)')
plt.xlabel('Chlorides')
plt.ylabel('Frequency')
plt.show()

df_white['residual sugar'].hist(bins=20, color='blue', edgecolor='green') #histogram for second skewed feature: residual sugar
plt.title('Histogram of residual sugar (white wine)')
plt.xlabel('Residual sugar')
plt.ylabel('Frequency')
plt.show()

df_white['sulphates'].hist(bins=25, color='blue', edgecolor='green')
plt.title('Histogram of sulphates (white wine)')
plt.xlabel('Sulphates')
plt.ylabel('Frequency')
plt.show()

df_white['total sulfur dioxide'].hist(bins=20, color='blue', edgecolor='green') #histogram for third skewed feature: total sulfur dioxide
plt.title('Histogram of total sulfur dioxide (white wine)')
plt.xlabel('Total sulfur dioxide')
plt.ylabel('Frequency')
plt.show()

df_white['free sulfur dioxide'].hist(bins=20, color='blue', edgecolor='green') #histogram for fourth skewed feature: free sulfur dioxide
plt.title('Histogram of total free sulfur dioxide (white)')
plt.xlabel('Free sulfur dioxide')
plt.ylabel('Frequency')
plt.show()


#visualising the data (white wine)
top_5 = ['alcohol', 'sulphates', 'citric acid', 'fixed acidity', 'residual sugar'] #histogram
df_white[top_5].hist(bins=5, figsize=(10, 6))
plt.suptitle('Distributions of the top 5 features')
plt.show()

plt.figure(figsize=(8,4)) #boxplot
sns.boxplot(x='quality', y='alcohol', data=df_white)
plt.title('Comparison top feature to target (white)')
plt.xlabel('Quality score')
plt.ylabel('Alcohol amount')
plt.show()

corr = df_white.corr() #heatmap
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.xlabel('Alcohol percentage')
plt.ylabel('Quality')
plt.show()

print('EDA for white wine complete')

#feature engineering
feature_interactions = [
    ('alcohol', 'sulphates'),
    ('pH', 'total sulfur dioxide'),
    ('sulphates', 'fixed acidity'),
    ('alcohol', 'citric acid'),
    ('pH', 'free sulfur dioxide')
] # feature interactions I think have the strongest correlation with regards to quality (white wine)

def add_interactions(df, feature_interactions):
  new_columns = []
  for f1, f2 in feature_interactions:
    new_column = f'{f1}_&_{f2}'
    df[new_column] = df[f1] * df[f2]
  return df

df_red = add_interactions(df_red, feature_interactions)
df_white = add_interactions(df_white, feature_interactions)

#function for calculating high correlation between features for both dataframes
def highly_corr(df, threshold=0.9):
  corr_mat = df.corr().abs() # creating correlation matrix to remove correlation between the most similar feature
  upper_triangle = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool)) # I try to take only upper part of matrix to avoid duplication
  dropping = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)] # checking high correlation and if so, dropping it
  print("Dropping highly correlated features (white wine):", dropping)
  return df.drop(columns=dropping)

df_red_dropped = highly_corr(df_red)
df_white_dropped = highly_corr(df_white)

#model building for red wine
print('Start modeling for RED wine...')
features_red = df_red_dropped.drop(columns=['quality'])
target_red = df_red_dropped['quality']
X = features_red
y = target_red
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#linear regression - first try
lr_pipeline = make_pipeline(StandardScaler(), LinearRegression())
neg_mse_lr = cross_val_score(lr_pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
pos_mse_lr = np.mean(-neg_mse_lr) #convert negative MSE to positive
r2_lr = np.mean(cross_val_score(lr_pipeline, X_train, y_train, cv=5, scoring='r2'))

lr_pipeline.fit(X_train, y_train)

print('Mean squared error for LR:', pos_mse_lr)
print('R2 score for LR:', r2_lr)

#random forest regressor - second try

rfr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rfr', RandomForestRegressor(random_state=42))
])
param_grid_rfr = {
    'rfr__n_estimators': [100, 200],
    'rfr__max_depth': [None, 10, 20, 30],
    'rfr__min_samples_split': [2, 3, 5]
}
gs_rfr = GridSearchCV(rfr_pipeline, param_grid_rfr, scoring='r2', cv=5)
gs_rfr.fit(X_train, y_train)
best_model_rfr_red = gs_rfr.best_estimator_
y_pred_rfr = best_model_rfr_red.predict(X_test)

mse_rfr = mean_squared_error(y_test, y_pred_rfr)
r2_rfr = r2_score(y_test, y_pred_rfr)
print('Best params:', gs_rfr.best_params_)
print('Mean squared error for RFR with gs:', mse_rfr)
print('R2 score for RFR with gs:', r2_rfr)

#gradient boosting - third try
gbr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('gbr', GradientBoostingRegressor(random_state=34))
])
param_grid_gbr = {
    'gbr__n_estimators': [50, 100, 150, 200],
    'gbr__learning_rate': [0.01, 0.1],
    'gbr__max_depth': [5, 10, 20],
    'gbr__subsample': [0.5, 0.8, 1.0]
}
gs_gbr = GridSearchCV(gbr_pipeline, param_grid_gbr, cv=5, scoring='r2')
gs_gbr.fit(X_train, y_train)
best_model_gbr_red = gs_gbr.best_estimator_
y_pred = best_model_gbr_red.predict(X_test)

mse_gbr = mean_squared_error(y_test, y_pred)
r2_gbr = r2_score(y_test, y_pred)
print('Best params for Gradient Boosting Regressor:', gs_gbr.best_params_)
print('Mean squared error for Gradient Boosting Resgressor with gs:', mse_gbr)
print('R2 score for Gradient Boosting Regressor with gs:', r2_gbr)


# WHITE WINE modeling (trying)
print('Start modeling for WHITE wine...')
features_white = df_white_dropped.drop(columns=['quality'])
target_white = df_white_dropped['quality']
X_white = features_white
y_white = target_white
Xw_train, Xw_test, yw_train, yw_test = train_test_split(X_white, y_white, random_state=42, test_size=0.2)

# Linear regression
lr_pipeline_white = make_pipeline(StandardScaler(), LinearRegression())
lr_pipeline_white.fit(X_white, y_white)
neg_mse_lr_white = cross_val_score(lr_pipeline_white, X_white, y_white, cv=5, scoring='neg_mean_squared_error')
pos_mse_lr_white = np.mean(-neg_mse_lr_white) #convert negative MSE to positive (white wine)
r2_lr_white = np.mean(cross_val_score(lr_pipeline_white, X_white, y_white, cv=5, scoring='r2'))

print('Mean squared error for LR (white wine):', pos_mse_lr_white)
print('R2 score for LR (white wine):', r2_lr_white)

#random forest for white wine

rfr_pipeline_white = Pipeline([
    ('scaler', StandardScaler()),
    ('rfr_white', RandomForestRegressor(random_state=42))
])
param_grid_rfr_white = {
    'rfr_white__n_estimators': [100, 200],
    'rfr_white__max_depth': [5, 10, 20, 30],
    'rfr_white__min_samples_split': [2, 3, 5]
}
gs_rfr_white = GridSearchCV(rfr_pipeline_white, param_grid_rfr_white, scoring='r2', cv=5)
gs_rfr_white.fit(Xw_train, yw_train)
best_model_rfr_white = gs_rfr_white.best_estimator_
y_pred_rfr_white = best_model_rfr_white.predict(Xw_test)

mse_rfr_white = mean_squared_error(yw_test, y_pred_rfr_white)
r2_rfr_white = r2_score(yw_test, y_pred_rfr_white)
print('Best params (white wine):', gs_rfr_white.best_params_)
print('Mean squared error for RFR with gs (white wine):', mse_rfr_white)
print('R2 score for RFR with gs (white wine):', r2_rfr_white)

#gradient boosting for white
gbr_pipeline_white = Pipeline([
    ('scaler', StandardScaler()),
    ('gbr_white', GradientBoostingRegressor(random_state=34))
])
param_grid_gbr_white = {
    'gbr_white__n_estimators': [50, 100, 150, 200],
    'gbr_white__learning_rate': [0.01, 0.1],
    'gbr_white__max_depth': [3, 10, 20],
    'gbr_white__subsample': [0.5, 0.8, 1.0]
}
gs_gbr_white = GridSearchCV(gbr_pipeline_white, param_grid_gbr_white, cv=5, scoring='r2')
gs_gbr_white.fit(Xw_train, yw_train)
best_model_gbr_white = gs_gbr_white.best_estimator_
y_pred_gbr_white = best_model_gbr_white.predict(Xw_test)

mse_gbr_white = mean_squared_error(yw_test, y_pred_gbr_white)
r2_gbr_white = r2_score(yw_test, y_pred_gbr_white)
print('Best params for Gradient Boosting Regressor (white wine):', gs_gbr_white.best_params_)
print('Mean squared error for Gradient Boosting Resgressor with gs (white wine):', mse_gbr_white)
print('R2 score for Gradient Boosting Regressor with gs (white wine):', r2_gbr_white)



#visual model comparison table (red wine)
results_df_red = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'Gradient Boosting'],
    'MSE': [pos_mse_lr, mse_rfr, mse_gbr],
    'R2': [r2_lr, r2_rfr, r2_gbr]
})
print('Model performance comparison (red wine):')
print(results_df_red)

plt.figure(figsize=(10, 6)) #visual for R2 comparison
sns.barplot(x='Model', y='R2', data=results_df_red, palette='viridis')
plt.title('Model performance comparison - R2')
plt.ylabel('R2 score')
plt.ylim(0,1)
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6)) #visual for MSE comparison
sns.barplot(x='Model', y='MSE', data=results_df_red, palette='viridis')
plt.title('Model performance comparison - MSE')
plt.xlabel('Model type')
plt.ylabel('MSE')
plt.show()


#visual model comparison table (white wine)
results_df_white = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'Gradient Boosting'],
    'MSE': [pos_mse_lr_white, mse_rfr_white, mse_gbr_white],
    'R2': [r2_lr_white, r2_rfr_white, r2_gbr_white]
})
print('Model performance comparison (white wine):')
print(results_df_white)

plt.figure(figsize=(10, 6)) #visual for R2 comparison
sns.barplot(x='Model', y='R2', data=results_df_white, palette='viridis')
plt.title('Model performance comparison - R2 (white wine)')
plt.ylabel('R2 score')
plt.ylim(0,1)
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6)) #visual for MSE comparison
sns.barplot(x='Model', y='MSE', data=results_df_white, palette='viridis')
plt.title('Model performance comparison - MSE (white wine)')
plt.xlabel('Model type')
plt.ylabel('MSE')
plt.show()

# now i will try to create a graph for comparison between RED and WHITE types of wine

# creating columns for each type of wine:
results_df_white['Type of wine'] = 'White'
results_df_red['Type of wine'] = 'Red'

#concatenating both DFs
df_together = pd.concat([results_df_red, results_df_white])

#plotting for r2
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='R2', hue='Type of wine', data=df_together, palette='viridis')
plt.title('White and red comparison of R2 score')
plt.ylabel('R2')
plt.show()

#plotting for mse
plt.figure(figsize=(10, 8))
sns.barplot(x='Model', y='MSE', hue='Type of wine', data=df_together, palette='viridis')
plt.title('White and red comparison of MSE')
plt.ylabel('MSE')
plt.show()

results_df_red.to_csv("model_comparison_red.csv", index=False)
results_df_white.to_csv("model_comparison_white.csv", index=False)

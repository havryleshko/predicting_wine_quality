import pandas as pd
import matplotlib.pyplot as plt
import warnings
import seaborn as sns

#loading the data
df_red = pd.read_csv('/Users/ohavryleshko/Documents/GitHub/WineQuality/wine+quality/winequality-red.csv', sep=';')
df_white = pd.read_csv('/Users/ohavryleshko/Documents/GitHub/WineQuality/wine+quality/winequality-white.csv', sep=';')
with open("/Users/ohavryleshko/Documents/GitHub/WineQuality/wine+quality/winequality.names", 'r') as f:
  names_text = f.read()
print(names_text)

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

corr = df_red.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
plt.xlabel('Alcohol percentage')
plt.ylabel('Quality')
plt.show()


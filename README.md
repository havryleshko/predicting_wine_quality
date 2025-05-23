# predicting_wine_quality

# Wine Quality Prediction (trying to accomplish score as closest to 9 (realisically))

This project uses various regression models to predict the quality of red and white wine based on physicochemical tests. It applies data exploration, preprocessing, feature engineering, and multiple modeling techniques to determine the most effective model for predicting wine quality

## Dataset

The dataset is provided by the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wine+Quality). It includes two datasets:

- `winequality-red.csv`
- `winequality-white.csv`

(I have left the folder *wine+quality* in the repository (containing actuall .csv files and .names file)

Each dataset contains 11 physicochemical input variables (e.g., acidity, sugar, pH) and one output variable: wine quality (score between 0 and 10, but realistically I aimed for 3-9).

## Project Overview

### 1. **Exploratory Data Analysis (EDA) (performed of both datasets (white and red)**
- **Null checks**: Ensures there are no missing values.
- **Correlation analysis**: Identifies features most correlated with wine quality.
- **Skewness visualization**: Applies log transformations to reduce skew in select features.
- **Distribution plots, box plots, and heatmaps**: For visual insights.

**side note:** some blocks of code I need to reuse for white wine dataset I have wrapped into the function, whereas visulaisations I created separately. Also, comprehensive EDA worked as it have me much more visual information, so I could better see what am I working with.

### 2. **Feature Engineering**
- Log transformation for skewed features.
- Created interaction features between important variables.
- Removed highly correlated features (correlation > 0.9) to reduce multicollinearity.

**side note:** most of the feature engineering I have worked on here have have been new to me to learn, so I have experimented here a lot

### 3. **Modeling**
Separate models were trained for red and white wines using three algorithms:
- **Linear Regression**
- **Random Forest Regressor** (with `GridSearchCV`)
- **Gradient Boosting Regressor** (with `GridSearchCV`)

Each model uses a `Pipeline` for streamlined scaling and training. I used two evaluation metrics:
- Mean Squared Error (MSE)
- R² Score

### 4. **Model Evaluation**
Models were compared using cross-validation and visualized in a result summary table for both red and white wine datasets.


## Technologies Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

### What I have learned:
1. Evaluation stage of ML workflow takes the most time, and the whole process in build upon the idea of trying
2. Playing with models' parameters and hyperparameter tuning is easier when I visualise data during EDA stage
3. There are three main visuals to use during EDA stage: histogram, boxplot and heatmap correlation
4. In many real-world problems (like wine quality prediction), features influence the target in combination, not isolation





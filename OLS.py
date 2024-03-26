import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, Lasso
from sklearn.model_selection import cross_val_score, train_test_split
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import SelectFromModel
import statsmodels.stats.api as sms
from sklearn.feature_selection import SelectFromModel



#import data
train = pd.read_csv('D:/2023semester/spyder/project1/HWA/housing/train.csv')
test = pd.read_csv('D:/2023semester/spyder/project1/HWA/housing/test.csv')

#combine train and test
train = train.drop(['Id','SalePrice'], axis = 1)
test = test.drop('Id', axis = 1)
data = pd.concat([train, test],axis = 0)

#change NA to feature
cols_to_fix = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 
                'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 
                'PoolQC', 'Fence', 'MiscFeature']
data[ cols_to_fix ] = data[ cols_to_fix ]. fillna ('NotAvailable')

#missing values
numerical_features = data.select_dtypes(include=[np.number])
categorical_features = data.select_dtypes(exclude=[np.number])
#fill in categorical_features（mode)
categorical_features = categorical_features.fillna(categorical_features.mode().iloc[0])
categorical_features.isnull().sum().sort_values(ascending=False)
#fill in numerical_features(mean)
numerical_features.isnull().sum().sort_values(ascending=False)
numerical_features = numerical_features.fillna(numerical_features.mean())

#transformation(log）
skew = stats.skew(numerical_features)
skew_df = pd.DataFrame({'numerical_features': numerical_features.columns, 'Skewness': skew})
selected_features = skew_df[((skew_df['Skewness'] >= -0.5) & (skew_df['Skewness'] <= 0.5))]
selected_features['numerical_features']
for column in selected_features['numerical_features']:
    numerical_features[column] = np.log1p(numerical_features[column])#log


#encode
data = pd.concat([numerical_features, categorical_features], axis=1)
    #label encoding
label_encoding_cols = [ 'Alley','BsmtCond', 'BsmtExposure','BsmtFinType1', 'FireplaceQu','BsmtQual', 'MiscFeature',
                        'GarageFinish', 'GarageQual','GarageCond','PoolQC','Fence',
    'Street', 'LotShape', 'Utilities', 'ExterQual', 'ExterCond', 'BsmtFinType2', 
    'HeatingQC', 'Electrical', 'KitchenQual', 'Functional', 'GarageType', 
    'PavedDrive']
label_encoder = LabelEncoder()
for col in label_encoding_cols:
    data[col] = label_encoder.fit_transform(data[col])
    #one-hot encoding
one_hot_encoding_cols = ['MSSubClass','MasVnrType','CentralAir',
    'MSZoning', 'LandContour', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
    'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'Heating', 
    'Foundation', 'SaleType', 'SaleCondition']
data = pd.get_dummies(data, columns=one_hot_encoding_cols, prefix=one_hot_encoding_cols)


#split train and Test
train = data.iloc[:1460]
test = data.iloc[1460:]
train_original = pd.read_csv('D:/2023semester/spyder/project1/HWA/housing/train.csv')
train = train.assign(SalePrice=train_original['SalePrice'])#add back y
X = train.drop('SalePrice', axis=1)
y = train['SalePrice']


#lasso
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=200)
lasso_cv = LassoCV(cv=5, max_iter=10000)
lasso_cv.fit(X_train, y_train)
best_alpha = lasso_cv.alpha_
best_alpha#296256.2514678291
lasso = Lasso(alpha=best_alpha, max_iter=10000)
lasso.fit(X, y)
lasso_coefs = lasso.coef_
# cross-validation
cross_val_scores = cross_val_score(lasso, X, y, cv=5)
mean_cv_score = np.mean(cross_val_scores)
mean_cv_score#Mean Cross-Validation Score: 0.6761187560503625
# feature selection
sfm = SelectFromModel(lasso)
sfm.fit(X, y)
selected_features = X.columns[sfm.get_support()]
X = X[selected_features]
X.columns

#describe
desc_stats = train[['LotArea', 'YearBuilt', 'MasVnrArea', 'BsmtFinSF1', 'TotalBsmtSF',
        'GrLivArea', 'WoodDeckSF', 'MiscVal', 'SalePrice']].describe(). loc [[ 'mean', 'std', 'min','max']]
print (desc_stats)
desc_stats.to_excel("descriptive_statistics.xlsx")

#ols
y = np.log(y)
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())
result_summary = model.summary()
result_as_html = result_summary.tables[1].as_html()
result_df = pd.read_html(result_as_html, header=0, index_col=0)[0]
result_df.to_excel("ols_results.xlsx")


#plot of residual
residuals = model.resid
plt.figure(figsize=(8, 6))
plt.scatter(model.fittedvalues, residuals)
plt.grid(True)
plt.xlabel('Values')
plt.ylabel('Observation')
plt.show()

# VIF for Multicollinearity
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)

# Goldfeld-Quandt Test for Heteroscedasticity
heteroscedasticity_test = sms.het_goldfeldquandt(model.resid, X)
print(heteroscedasticity_test)

#predict
test = test[['LotArea', 'YearBuilt', 'MasVnrArea', 'BsmtFinSF1', 'TotalBsmtSF',
        'GrLivArea', 'WoodDeckSF', 'MiscVal']]
test = sm.add_constant(test)
predictions = model.predict(test)
predictions = np.exp(predictions)
predictions
submission = pd.DataFrame({'Id': range(1461, 1461 + len(predictions)), 'SalePrice': predictions})
submission.to_csv('D:/2023semester/spyder/project1/HWA/predict(ols).csv', index=False)

# Plot the influence of YearBuilt on Prediction of SalePrice
plt.figure(figsize=(8, 6))
plt.scatter(train['YearBuilt'], np.exp(model.fittedvalues), label='Predicted SalePrice', alpha=0.5)
plt.title('Influence of YearBuilt on Prediction of SalePrice')
plt.xlabel('YearBuilt')
plt.ylabel('SalePrice')
plt.grid(True)
plt.show()


#plot 






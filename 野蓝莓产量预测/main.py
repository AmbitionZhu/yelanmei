import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV, RepeatedKFold
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import statsmodels.api as sm
from xgboost import XGBRegressor
import shap
import joblib

df = pd.read_csv("WildBlueberryPollinationSimulationData.csv", index_col='Row#')
print(df.head())

# print the metadata of the dataset
df.info()

# data description
df.describe()

# create featureset and target variable from the dataset
features_df = df.drop('yield', axis=1)
tar = df['yield']

# plot the heatmap from the dataset
plt.figure(figsize=(15,15))
sns.heatmap(df.corr(), annot=True, vmin=-1, vmax=1)
plt.show()

# plot the boxplot using seaborn library of the target variable 'yield'
plt.figure(figsize=(5,5))
sns.boxplot(x='yield', data=df)
plt.show()

# matplotlib subplot for the categorical feature
nominal_df = df[['MaxOfUpperTRange','MinOfUpperTRange','AverageOfUpperTRange','MaxOfLowerTRange',
               'MinOfLowerTRange','AverageOfLowerTRange','RainingDays','AverageRainingDays']]

fig, ax = plt.subplots(2,4, figsize=(20,13))
for e, col in enumerate(nominal_df.columns):
    if e<=3:
        sns.boxplot(data=df, x=col, y='yield', ax=ax[0,e])
    else:
        sns.boxplot(data=df, x=col, y='yield', ax=ax[1,e-4])
plt.show()

# matplotlib subplot technique to plot distribution of bees in our dataset
plt.figure(figsize=(15,10))
plt.subplot(2,3,1)
plt.hist(df['bumbles'])
plt.title("Histogram of bumbles column")
plt.subplot(2,3,2)
plt.hist(df['andrena'])
plt.title("Histogram of andrena column")
plt.subplot(2,3,3)
plt.hist(df['osmia'])
plt.title("Histogram of osmia column")
plt.subplot(2,3,4)
plt.hist(df['clonesize'])
plt.title("Histogram of clonesize column")
plt.subplot(2,3,5)
plt.hist(df['honeybee'])
plt.title("Histogram of honeybee column")
plt.show()

# run the MI scores of the dataset
mi_score = mutual_info_regression(features_df, tar, n_neighbors=3,random_state=42)
mi_score_df = pd.DataFrame({'columns':features_df.columns, 'MI_score':mi_score})
mi_score_df.sort_values(by='MI_score', ascending=False)

# clustering using kmeans algorithm
X_clus = features_df[['honeybee','osmia','bumbles','andrena']]

scaler = StandardScaler()
scaler.fit(X_clus)
X_new_clus = scaler.transform(X_clus)

clustering = KMeans(n_clusters=3, random_state=42)
clustering.fit(X_new_clus)
n_cluster = clustering.labels_

# add new feature to feature_Df
features_df['n_cluster'] = n_cluster
df['n_cluster'] = n_cluster
features_df['n_cluster'].value_counts()

# let's plot most imporatant feature VS yield
plt.figure(figsize=(6,6))
sns.scatterplot(x='seeds', y='yield', hue='n_cluster', data=df)
plt.title("Clustering scatter plot")
plt.show()

features_set = ['AverageRainingDays','clonesize','AverageOfLowerTRange',
               'AverageOfUpperTRange','honeybee','osmia','bumbles','andrena','n_cluster']

# final dataframe
X = features_df[features_set]
y = tar.round(1)

# train and test dataset to build baseline model using GBT and RFs by scaling the dataset
mx_scaler = MinMaxScaler()
X_scaled = pd.DataFrame(mx_scaler.fit_transform(X))
X_scaled.columns = X.columns

# let's fit the data to the models
model_dict = {"abr": AdaBoostRegressor(),
              "gbr": GradientBoostingRegressor(),
              "rfr": RandomForestRegressor()
             }

for key, val in model_dict.items():
    print(f"cross validation for {key}")
    score = cross_val_score(val, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
    mean_score = -np.sum(score)/5
    sqrt_score = np.sqrt(mean_score)
    print(sqrt_score)

# split the train and test data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

bgt = GradientBoostingRegressor(random_state=42)
bgt.fit(X_train,y_train)
preds = bgt.predict(X_test)
score = bgt.score(X_train,y_train)
rmse_score = np.sqrt(mean_squared_error(y_test, preds))
r2_score = r2_score(y_test, preds)
print("RMSE score gradient boosting machine:", rmse_score)
print("R2 score for the model: ", r2_score)

kf = KFold(n_splits = 5, shuffle=True, random_state=0)

param_grid = {'n_estimators': [100,200,400,500,800],
             'learning_rate': [0.1,0.05,0.3,0.7],
             'min_samples_split': [2,4],
             'min_samples_leaf': [0.1,0.4],
             'max_depth': [3,4,7]
             }

estimator = GradientBoostingRegressor(random_state=42)

clf = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=kf,
                   scoring='neg_mean_squared_error', n_jobs=-1)
clf.fit(X_scaled,y)

best_estim = clf.best_estimator_
best_score = clf.best_score_
best_param = clf.best_params_
print("Best Estimator:", best_estim)
print("Best score:", np.sqrt(-best_score))

# building statsmodel regression model
model = sm.OLS(y, X_scaled)
results = model.fit()
print(results.summary())

shap_tree = shap.TreeExplainer(bgt)
shap_values = shap_tree.shap_values(X_test)
shap.summary_plot(shap_values, X_test)

shap.summary_plot(shap_values, X_test, plot_type='bar')


# repeated kfold to
cv = RepeatedKFold(n_splits= 5, n_repeats = 3, random_state = 1)
fs_info_v0 = SelectKBest(score_func = mutual_info_regression)

# define pipeline object
pipe_rf = Pipeline([
    ('sel', fs_info_v0),
    ('model', RandomForestRegressor(random_state=1))
])

pipe_xgb = Pipeline([
    ('sel', fs_info_v0),
    ('model', XGBRegressor(random_state=1))
])


# prepare the parameters grid
grid_params_rf = [{'sel__k': [i for i in range(X_train.shape[1]-6, X_train.shape[1]-4)],
                   'model__max_depth': [15, 18, 10],
                   'model__min_samples_split': [15, 18, 10],
                   'model__n_estimators': [100,200,400,500]
                  }]

grid_params_xgb = [{'sel__k': [i for i in range(X_train.shape[1]-6, X_train.shape[1]-4)],
                    'model__max_depth': [9,12],
                    'model__min_child_weight': [7,8],
                    'model__subsample': [i/10. for i in range(9,11)]
                   }]
# set up the gridsearchCV objects
RF = GridSearchCV(estimator=pipe_rf,
            param_grid=grid_params_rf,
            scoring='neg_mean_absolute_error',
            cv=cv,
            n_jobs= -1)

XGB = GridSearchCV(estimator=pipe_xgb,
            param_grid=grid_params_xgb,
            scoring='neg_mean_absolute_error',
            cv=cv,
            n_jobs= -1)

# list of regression models
grids = [RF,XGB]

# Creating a dict for our reference
grid_dict = {
        0: 'Random Forest',
        1: 'XGBoost'
    }


# Start form initial scaled model: X_train and X_test, y_train and y_test
def extract_best_model(grids: list, grid_dict: dict):
    print('Performing model optimizations...')
    least_mae = 270817
    best_regr = 0
    best_gs = ''
    for idx, gs in enumerate(grids):
        print('\nEstimator: %s' % grid_dict[idx])
        gs.fit(X_train, y_train)
        print('Best Config: %s' % gs.best_params_)
        # Best training data accuracy
        print('Best MAE: %.3f' % gs.best_score_)
        # Predict on test data with best params
        y_pred_v0 = gs.predict(X_test)
        # Test data accuracy of model with best params
        print('Test set mean absolute error for best params: %.3f ' % mean_absolute_error(y_test, y_pred_v0))
        print(
            'Test set root mean squared error for best params: %.3f ' % np.sqrt(mean_absolute_error(y_test, y_pred_v0)))

        # Track best (least test error) model
        if mean_absolute_error(y_test, y_pred_v0) < least_mae:
            least_mae = mean_absolute_error(y_test, y_pred_v0)
            best_gs = gs
            best_regr = idx
    print('\nClassifier with least test set MAE: %s' % grid_dict[best_regr])

    return (grid_dict[best_regr], best_gs, least_mae)
# run the pipeline and print the results
best_model_name_v0, best_model_v0, least_mae_v0 = extract_best_model(grids= grids, grid_dict = grid_dict)

print(f"Best Model: {best_model_name_v0}")
print(f"Error Rate: {least_mae_v0}")
print(best_model_v0)


X_train_n = X_train.drop('n_cluster', axis=1)
X_test_n = X_test.drop('n_cluster', axis=1)
# train flask deployment
xgb_model = XGBRegressor(max_depth=9, min_child_weight=7, subsample=1.0)
xgb_model.fit(X_train_n, y_train)
pr = xgb_model.predict(X_test_n)
err = mean_absolute_error(y_test, pr)
rmse_n = np.sqrt(mean_squared_error(y_test, pr))
print(err)
print(rmse_n)
var = X_test_n.columns
joblib.dump(xgb_model, 'wbb_xgb_model2.joblib')

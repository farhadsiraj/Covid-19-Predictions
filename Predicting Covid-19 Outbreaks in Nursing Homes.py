#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import packages
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from textwrap import wrap
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Random forest and boosting packages
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import SMOTE


# In[ ]:


# Set working directories
main_dir = "D:\\Users\\SirajF\\nhc"
data_dir = main_dir + "/data"
plot_dir = main_dir + "/plots"
os.chdir(data_dir)

# Check working directory.
print(os.getcwd())


# In[ ]:


# Set max column/row length
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)


# In[ ]:


# Load analytic file, option to import dates as datetime commented out if necessary
nhc_orig = pd.read_csv('nhc_covid_final 20200626.csv') #, parse_dates=[0]


# In[ ]:


nhc_orig.head(5)


# In[ ]:


# Keep period = 2-4
array = [2,3,4]
nhc = nhc_orig.loc[nhc_orig['period'].isin(array)].copy()


# In[ ]:


nhc_orig.cdc_outbreak.value_counts()


# In[ ]:


# Check data
nhc


# In[ ]:


nhc.cdc_outbreak.value_counts()


# In[ ]:


# Check shape of data
nhc.shape


# In[ ]:


# Create list of columns before we drop vars
nhc_raw = list(nhc)


# In[ ]:


# Create cleaned dataset for random forest
nhc_rf = nhc.copy()


# In[ ]:


# Convert all booleans to True/False
nhc_rf = nhc_rf.applymap(lambda x: 1 if x == 'Y' else x)
nhc_rf = nhc_rf.applymap(lambda x: 0 if x == 'N' else x)


# In[ ]:


# Convert all booleans to 1/0
nhc_rf = nhc_rf.applymap(lambda x: 1 if x == True else x)
nhc_rf = nhc_rf.applymap(lambda x: 0 if x == False else x)


# In[ ]:


# Convert target var to bool
nhc_rf['cdc_outbreak'] = np.where(nhc_rf['cdc_outbreak'] == 1 , True, False)


# In[ ]:


#print(nhc_rf['outbreak_2020-5-24'].dtypes)
print(nhc_rf['cdc_outbreak'].dtypes)


# In[ ]:


nhc_rf.cdc_outbreak.value_counts()


# In[ ]:


# Drop misc vars
nhc_rf = nhc_rf.drop(['provnum',
'weekending',
'numperiods',
'provname',
'facid',
'address',
'city',
'state',
'statename',
'zip',
'phone',
'county',
'FIPScty',
'county_name',
'cms_region',
'cbsa_code',
'nhc_urban',
'nhc_sffcand',
'nhc_comp5star',
'nhc_qm5star',
'nhc_staff5star',
'nhc_rnstf5',
'nhc_qm005',
'nhc_qm401',
'nhc_qm406',
'nhc_qm407',
'nhc_qm410',
'nhc_qm419',
'nhc_qm434',
'nhc_qm451',
'nhc_qm453',
'nhc_qm471',
'nhc_qm476',
'nhc_incident_cnt',
'pos_chow_cnt',
'nhc_is_ccrc',
'nhc_1st_cert_dt',
'nhc_chng_ownr_12m',
'nhc_srvy_gt2yrs',
'ahrf_ctyname',
'ahrf_hcd_pctmcradvtg2018',
'ahrf_ecn_pctpvrty2017',
'ahrf_ecn_unemprt2018',
'ahrf_oth_airqlty2018',
'reg_dum1',
'reg_dum2',
'reg_dum3',
'reg_dum4',
'div_dum1',
'div_dum2',
'div_dum3',
'div_dum4',
'div_dum5',
'div_dum6',
'div_dum7',
'div_dum8',
'div_dum9',
'ahrf_dmg_lo19_2017',
'ahrf_dmg_2034_2017',
'ahrf_dmg_3564_2017',
'cdc_submitteddata',
'cdc_passedqualcheck',
'cdc_c19admissions_res_week',
'cdc_c19admissions_res',
'cdc_c19confirmed_res_week',
'cdc_c19confirmed_res',
'cdc_c19confirmed_res_perc',
'cdc_c19suspected_res_week',
'cdc_c19suspected_res',
'cdc_c19deaths_res_week',
'cdc_c19deaths_res',
'cdc_c19deaths_res_perc',
'cdc_c19deaths_res_percconfirmed',
'cdc_c19confirmed_staff_week',
'cdc_c19confirmed_staff',
'cdc_c19suspected_staff_week',
'cdc_c19suspected_staff',
'cdc_c19deaths_staff_week',
'cdc_c19deaths_staff',
'cdc_alldeaths_res_week',
'cdc_alldeaths_res',
'cdc_beds',
'cdc_occupiedbeds',
'cdc_testing_res',
'cdc_shortage_nursingstaff',
'cdc_shortage_clinicalstaff',
'cdc_shortage_aides',
'cdc_shortage_otherstaff',
'cdc_weeksupply_n95mask',
'cdc_weeksupply_surgmask',
'cdc_weeksupply_eyeprotection',
'cdc_weeksupply_gowns',
'cdc_weeksupply_gloves',
'cdc_weeksupply_sanitizer',
'cdc_ventilatordepunit',
'cdc_numventilator',
'cdc_numventilator_c19use',
'cdc_weeksupply_ventilator',
'cdc_geolocationstate',
'cdc_geolocation'
], axis = 1)


# In[ ]:


#Replace NAN with 0
nhc_rf = nhc_rf.fillna(0)


# In[ ]:


nhc_rf.shape


# In[ ]:


nhc_rf.cdc_outbreak.value_counts()


# In[ ]:


# Get list of column names with total cases by week
totcase_week = list(nhc.columns[nhc.columns.str.startswith('usa_totcase_')])


# In[ ]:


totcase_week


# # Prep Test Data

# In[ ]:


# Keep period = 2 or 3
array1 = [2,3]
nhc_rf_test = nhc_rf.loc[nhc_rf['period'].isin(array1)]


# In[ ]:


nhc_rf_test.head(5)


# In[ ]:


# Set the target week we are trying to predict
targetweek = 'usa_totcase_wk0531'
targetweek_cntg = 'usa_totcase_wk0531_cntg'


targetweek2 = 'usa_totcase_wk0607'
targetweek_cntg2 = 'usa_totcase_wk0607_cntg'


# In[ ]:


# Create function to find vars for 3 to 7 weeks before target
def find_adjacents(value, items, x):
    i = items.index(value)
    return items[i-x]


# In[ ]:


# Store function output as var for each week
wago3_0531 = (find_adjacents(targetweek, totcase_week, 3))
wago4_0531 = (find_adjacents(targetweek, totcase_week, 4))
wago5_0531 = (find_adjacents(targetweek, totcase_week, 5))
wago6_0531 = (find_adjacents(targetweek, totcase_week, 6))
wago7_0531 = (find_adjacents(targetweek, totcase_week, 7))


# In[ ]:


wago3_cntg_0531 = (find_adjacents(targetweek_cntg, totcase_week, 3))
wago4_cntg_0531 = (find_adjacents(targetweek_cntg, totcase_week, 4))
wago5_cntg_0531 = (find_adjacents(targetweek_cntg, totcase_week, 5))
wago6_cntg_0531 = (find_adjacents(targetweek_cntg, totcase_week, 6))
wago7_cntg_0531 = (find_adjacents(targetweek_cntg, totcase_week, 7))


# In[ ]:


# Store function output as var for each week
wago3_0607 = (find_adjacents(targetweek2, totcase_week, 3))
wago4_0607 = (find_adjacents(targetweek2, totcase_week, 4))
wago5_0607 = (find_adjacents(targetweek2, totcase_week, 5))
wago6_0607 = (find_adjacents(targetweek2, totcase_week, 6))
wago7_0607 = (find_adjacents(targetweek2, totcase_week, 7))


# In[ ]:


wago3_cntg_0607 = (find_adjacents(targetweek_cntg2, totcase_week, 3))
wago4_cntg_0607 = (find_adjacents(targetweek_cntg2, totcase_week, 4))
wago5_cntg_0607 = (find_adjacents(targetweek_cntg2, totcase_week, 5))
wago6_cntg_0607 = (find_adjacents(targetweek_cntg2, totcase_week, 6))
wago7_cntg_0607 = (find_adjacents(targetweek_cntg2, totcase_week, 7))


# In[ ]:


# Print to check the weeks are correct
print("5/31/2020")
print(wago3_0531) 
print(wago4_0531) 
print(wago5_0531) 
print(wago6_0531) 
print(wago7_0531) 

print(wago3_cntg_0531)
print(wago4_cntg_0531) 
print(wago5_cntg_0531) 
print(wago6_cntg_0531) 
print(wago7_cntg_0531) 

print("6/7/2020")
print(wago3_0607) 
print(wago4_0607) 
print(wago5_0607) 
print(wago6_0607) 
print(wago7_0607) 

print(wago3_cntg_0607)
print(wago4_cntg_0607) 
print(wago5_cntg_0607) 
print(wago6_cntg_0607) 
print(wago7_cntg_0607) 


# In[ ]:


# Add the week vars to the data set
nhc_rf_test['case_3wago'] = np.where(nhc_rf_test['period']==2,  nhc_rf_test[wago3_0531], nhc_rf_test[wago3_0607])
nhc_rf_test['case_4wago'] = np.where(nhc_rf_test['period']==2,  nhc_rf_test[wago4_0531], nhc_rf_test[wago4_0607])
nhc_rf_test['case_5wago'] = np.where(nhc_rf_test['period']==2,  nhc_rf_test[wago5_0531], nhc_rf_test[wago5_0607])
nhc_rf_test['case_6wago'] = np.where(nhc_rf_test['period']==2,  nhc_rf_test[wago6_0531], nhc_rf_test[wago6_0607])
nhc_rf_test['case_7wago'] = np.where(nhc_rf_test['period']==2,  nhc_rf_test[wago7_0531], nhc_rf_test[wago7_0607])

nhc_rf_test['case_3wago_cntg'] = np.where(nhc_rf_test['period']==2,  nhc_rf_test[wago3_cntg_0531], nhc_rf_test[wago3_cntg_0607])
nhc_rf_test['case_4wago_cntg'] = np.where(nhc_rf_test['period']==2,  nhc_rf_test[wago4_cntg_0531], nhc_rf_test[wago4_cntg_0607])
nhc_rf_test['case_5wago_cntg'] = np.where(nhc_rf_test['period']==2,  nhc_rf_test[wago5_cntg_0531], nhc_rf_test[wago5_cntg_0607])
nhc_rf_test['case_6wago_cntg'] = np.where(nhc_rf_test['period']==2,  nhc_rf_test[wago6_cntg_0531], nhc_rf_test[wago6_cntg_0607])
nhc_rf_test['case_7wago_cntg'] = np.where(nhc_rf_test['period']==2,  nhc_rf_test[wago7_cntg_0531], nhc_rf_test[wago7_cntg_0607])


# In[ ]:


nhc_rf_test.head(5)


# In[ ]:


nhc_rf_test = nhc_rf_test.drop(['usa_totcase_wk0308',
 'usa_totcase_wk0315',
 'usa_totcase_wk0322',
 'usa_totcase_wk0329',
 'usa_totcase_wk0405',
 'usa_totcase_wk0412',
 'usa_totcase_wk0419',
 'usa_totcase_wk0426',
 'usa_totcase_wk0503',
 'usa_totcase_wk0510',
 'usa_totcase_wk0517',
 'usa_totcase_wk0524',
 'usa_totcase_wk0531',
 'usa_totcase_wk0607',
 'usa_totcase_wk0614',
 'usa_totcase_wk0621',
 'usa_totcase_wk0308_cntg',
 'usa_totcase_wk0315_cntg',
 'usa_totcase_wk0322_cntg',
 'usa_totcase_wk0329_cntg',
 'usa_totcase_wk0405_cntg',
 'usa_totcase_wk0412_cntg',
 'usa_totcase_wk0419_cntg',
 'usa_totcase_wk0426_cntg',
 'usa_totcase_wk0503_cntg',
 'usa_totcase_wk0510_cntg',
 'usa_totcase_wk0517_cntg',
 'usa_totcase_wk0524_cntg',
 'usa_totcase_wk0531_cntg',
 'usa_totcase_wk0607_cntg',
 'usa_totcase_wk0614_cntg',
 'usa_totcase_wk0621_cntg',
 'period'], axis = 1)


# In[ ]:


nhc_rf_test.head(5)


# In[ ]:


# # Create list of columns after cleans for comparison
nhc_clean = list(nhc_rf_test)


# In[ ]:


dropped_vars = list(set(nhc_raw) - set(nhc_clean))


# In[ ]:


nhc_rf_test.cdc_outbreak.value_counts()


# In[ ]:


# Pickle the cleaned dataset
pickle.dump(nhc_rf_test, open("nhc_covid19_rf_test.csv", "wb"))


# # Random Forest

# In[ ]:


# Select the predictors and target.
X = nhc_rf_test.drop(['cdc_outbreak'], axis = 1)

y = np.array(nhc_rf_test['cdc_outbreak'])


# In[ ]:


# Set the seed to 1.
seed = 1

# Split into the training and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=seed)

# Balancing data with SMOTE
sm = SMOTE(random_state=seed)
X_train, y_train = sm.fit_sample(X_train,y_train)


# In[ ]:


vars_rf = list(X)


# In[ ]:


X.head(5)


# In[ ]:


# Set model parameters
forest = RandomForestClassifier(criterion = 'gini',
                                n_estimators = 500,
                                random_state = 1)


# In[ ]:


# Fit the saved model to your training data.
forest.fit(X_train, y_train)


# In[ ]:


# Predict on test data.
y_predict_forest = forest.predict(X_test)

# Look at the first few predictions.
print(y_predict_forest[0:50,])


# In[ ]:


# Take a look at test data confusion matrix.
conf_matrix_forest = metrics.confusion_matrix(y_test, y_predict_forest)
print(conf_matrix_forest)
accuracy_forest = metrics.accuracy_score(y_test, y_predict_forest)
print("Accuracy for random forest on test data: ", accuracy_forest)


# In[ ]:


plt.clf()
plt.imshow(conf_matrix_forest, interpolation='nearest', cmap=plt.cm.tab20)
classNames = ['Negative','Positive']
plt.title('Nursing Home Compare Outbreak Confusion Matrix - Test Data')
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j-0.25,i, str(s[i][j])+" = "+str(conf_matrix_forest[i][j]))
plt.show()


# In[ ]:


# Compute accuracy using training data.
acc_train_forest = forest.score(X_train, y_train)

print ("Train Accuracy:", acc_train_forest)


# In[ ]:


# Create a dictionary with accuracy values for model 
model_final_dict = {'metrics': ["accuracy"],
               'values':[round(accuracy_forest,4)],
                'model':['random_forest_05312020']}
model_final = pd.DataFrame(data = model_final_dict)
print(model_final)


# In[ ]:


nhc_rf_features = nhc_rf_test.drop('cdc_outbreak', axis = 1)
features = nhc_rf_features.columns
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
top_indices = indices[0:10][::-1]

plt.figure(1)
plt.title('Feature Importance')
plt.barh(range(len(top_indices)), importances[top_indices], color = 'b', align = 'center')
labels = features[top_indices]
labels = [ '\n'.join(wrap(l,30)) for l in labels ]
plt.yticks(range(len(top_indices)), labels)
plt.xlabel('Relative Importance')


# In[ ]:


# Predict on test.
forest_y_predict = forest.predict(X_test)
print(forest_y_predict[:5])
#Predict on test, but instead of labels
# we will get probabilities for class 0 and 1.
forest_y_predict_prob = forest.predict_proba(X_test)
print(forest_y_predict_prob[5:])


# In[ ]:


def get_performance_scores(y_test, y_predict, y_predict_prob, eps=1e-15, beta=0.5):

    from sklearn import metrics

    # Scores keys.
    metric_keys = ["accuracy", "precision", "recall", "f1", "fbeta", "log_loss", "AUC"]

    # Score values.
    metric_values = [None]*len(metric_keys)

    metric_values[0] = metrics.accuracy_score(y_test, y_predict)
    metric_values[1] = metrics.precision_score(y_test, y_predict)
    metric_values[2] = metrics.recall_score(y_test, y_predict)
    metric_values[3] = metrics.f1_score(y_test, y_predict)
    metric_values[4] = metrics.fbeta_score(y_test, y_predict, beta=beta)
    metric_values[5] = metrics.log_loss(y_test, y_predict_prob[:, 1], eps=eps)
    metric_values[6] = metrics.roc_auc_score(y_test, y_predict_prob[:, 1])

    perf_metrics = dict(zip(metric_keys, metric_values))

    return(perf_metrics)


# In[ ]:


forest_scores = get_performance_scores(y_test, forest_y_predict, forest_y_predict_prob)

ensemble_methods_metrics = {"RF": forest_scores}
print(ensemble_methods_metrics)


# In[ ]:


rf_roc = metrics.plot_roc_curve(forest, X_test, y_test, name = "RF")
plt.show()


# In[ ]:


pickle.dump(forest, open("nhc_covid19_rfmodel.sav","wb" ))


# # Optimized Random Forest

# In[ ]:


forest.get_params()


# In[ ]:


# Number of trees in random forest.
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 20)]

# Number of features to consider at every split.
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree.
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)

# Minimum number of samples required to split a node.
min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node.
min_samples_leaf = [1, 2, 4]

# Set Minimal Cost-Complexity Pruning parameter (has to be >= 0.0).
ccp_alpha = [0.0, 0.001, 0.01, 0.1, 0.2, 0.3]

# Create the random grid
# (a python dictionary in a form `'parameter_name': parameter_values`)
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'ccp_alpha': ccp_alpha}

print(random_grid)


# In[ ]:


rf_random = RandomizedSearchCV(estimator = forest, #<- model object
                               param_distributions = random_grid, #<- param grid
                               n_iter = 100,#<- number of param. settings sampled
                               cv = 3,      #<- 3-fold CV
                               verbose = 0, #<- silence lengthy output to console
                               random_state = 1, #<- set random state
                               n_jobs = -1)      #<- use all available processors

# Fit the random search model.
rf_random.fit(X_train, y_train) #<- fit like any other scikit-learn model
# Take a look at optimal combination of parameters.
print(rf_random.best_params_)


# In[ ]:


pickle.dump(rf_random, open("rf_random_0627.sav","wb"))


# In[ ]:


rf_random = pickle.load(open("rf_random_0627.sav","rb"))


# In[ ]:


# Pass best parameters obtained through randomized search to RF classifier.
optimized_forest = RandomForestClassifier(**rf_random.best_params_)

# Train the optimized RF model.
optimized_forest.fit(X_train, y_train)


# In[ ]:


# Get predicted labels for test data.
optimized_forest_y_predict = optimized_forest.predict(X_test)

# Get predicted probabilities.
optimized_forest_y_predict_proba = optimized_forest.predict_proba(X_test)
# Compute performance scores.
optimized_forest_scores = get_performance_scores(y_test,
optimized_forest_y_predict,
optimized_forest_y_predict_proba)


# In[ ]:


# Take a look at test data confusion matrix.
conf_matrix_forest_opt = metrics.confusion_matrix(y_test, optimized_forest_y_predict)
print(conf_matrix_forest_opt)
accuracy_forest_opt = metrics.accuracy_score(y_test, optimized_forest_y_predict)
print("Accuracy for random forest on test data: ", accuracy_forest_opt)


# In[ ]:


plt.clf()
plt.imshow(conf_matrix_forest_opt, interpolation='nearest', cmap=plt.cm.tab20)
classNames = ['Negative','Positive']
plt.title('Nursing Home Compare Outbreak Confusion Matrix - Test Data')
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j-0.25,i, str(s[i][j])+" = "+str(conf_matrix_forest_opt[i][j]))
plt.show()


# In[ ]:


nhc_rf_features_opt = nhc_rf_test.drop('cdc_outbreak', axis = 1)
features = nhc_rf_features.columns
importances = optimized_forest.feature_importances_
indices = np.argsort(importances)[::-1]
top_indices = indices[0:15][::-1]

plt.figure(1)
plt.title('Feature Importance')
plt.barh(range(len(top_indices)), importances[top_indices], color = 'b', align = 'center')
labels = features[top_indices]
labels = [ '\n'.join(wrap(l,30)) for l in labels ]
plt.yticks(range(len(top_indices)), labels)
plt.xlabel('Relative Importance')


# In[ ]:


for feat, importance in zip(nhc_rf_features_opt.columns, optimized_forest.feature_importances_):
    print ('feature: {f}, importance: {i}'.format(f=feat, i=importance))


# In[ ]:



opt_rf_roc = metrics.plot_roc_curve(optimized_forest,
                                    X_test,
                                    y_test,
                                    name = "Optimized RF",
                                    )

plt.show()


# In[ ]:


optimized_forest_scores = get_performance_scores(y_test, optimized_forest_y_predict, optimized_forest_y_predict_proba)


# In[ ]:


ensemble_methods_metrics.update({"Optimized RF": optimized_forest_scores})
print(ensemble_methods_metrics)


# In[ ]:


# Convert metrics dictionary to dataframe

# Convert all metrics for each model to a dataframe.
ensemble_methods_metrics_df = pd.DataFrame(ensemble_methods_metrics)
ensemble_methods_metrics_df["metric"] = ensemble_methods_metrics_df.index
ensemble_methods_metrics_df = ensemble_methods_metrics_df.reset_index(drop = True)
print(ensemble_methods_metrics_df)


# In[ ]:


pickle.dump(optimized_forest, open("nhc_covid19_rfmodel_optimized.sav","wb" ))


# ## Gradient Boosting
# 

# In[ ]:


# Save the parameters we will be using for our gradient boosting classifier.
gbm = GradientBoostingClassifier(n_estimators = 200, 
                                learning_rate = 1,
                                max_depth = 2, 
                                random_state = 1)


# In[ ]:


# Fit the saved model to the training data.
gbm.fit(X_train, y_train)


# In[ ]:


# Predict on test data.
predicted_values_gbm = gbm.predict(X_test)
print(predicted_values_gbm)


# In[ ]:


# Take a look at test data confusion matrix.
conf_matrix_boosting = metrics.confusion_matrix(y_test, predicted_values_gbm)
print(conf_matrix_boosting)
# Compute test model accuracy score.
accuracy_gbm = metrics.accuracy_score(y_test, predicted_values_gbm)
print('Accuracy of gbm on test data: ', accuracy_gbm)


# In[ ]:


plt.clf()
plt.imshow(conf_matrix_boosting, interpolation='nearest', cmap=plt.cm.tab20)
classNames = ['Negative','Positive']
plt.title('Nursing Home Compare Outbreak GBM Confusion Matrix - Test Data')
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j-0.25,i, str(s[i][j])+" = "+str(conf_matrix_boosting[i][j]))
plt.show()


# In[ ]:


# Compute accuracy using training data.
train_accuracy_gbm = gbm.score(X_train, y_train)

print ("Train Accuracy:", train_accuracy_gbm)


# In[ ]:


# Create a dictionary with accuracy values for model 
model_final_dict = {'metrics': ["accuracy"],
               'values':[round(accuracy_gbm,4)],
                'model':['train_gbm_20200628']}
model_final = pd.DataFrame(data = model_final_dict)
print(model_final)


# In[ ]:


nhc_gbm_features = nhc_rf_test.drop('cdc_outbreak', axis = 1)
features = nhc_gbm_features.columns
importances = gbm.feature_importances_
indices = np.argsort(importances)[::-1]
top_indices = indices[0:10][::-1]

plt.figure(1)
plt.title('Feature Importance')
plt.barh(range(len(top_indices)), importances[top_indices], color = 'b', align = 'center')
labels = features[top_indices]
labels = [ '\n'.join(wrap(l,30)) for l in labels ]
plt.yticks(range(len(top_indices)), labels)
plt.xlabel('Relative Importance')


# In[ ]:


pickle.dump(gbm, open("nhc_covid19_gbm_20200628.sav","wb" ))


# In[ ]:


# Predict on test.
gbm_y_predict = gbm.predict(X_test)
print(gbm_y_predict[:5])
#Predict on test, but instead of labels
# we will get probabilities for class 0 and 1.
gbm_y_predict_prob = gbm.predict_proba(X_test)
print(gbm_y_predict_prob[5:])


# In[ ]:


def get_performance_scores(y_test, y_predict, y_predict_prob, eps=1e-15, beta=0.5):

    from sklearn import metrics

    # Scores keys.
    metric_keys = ["accuracy", "precision", "recall", "f1", "fbeta", "log_loss", "AUC"]

    # Score values.
    metric_values = [None]*len(metric_keys)

    metric_values[0] = metrics.accuracy_score(y_test, y_predict)
    metric_values[1] = metrics.precision_score(y_test, y_predict)
    metric_values[2] = metrics.recall_score(y_test, y_predict)
    metric_values[3] = metrics.f1_score(y_test, y_predict)
    metric_values[4] = metrics.fbeta_score(y_test, y_predict, beta=beta)
    metric_values[5] = metrics.log_loss(y_test, y_predict_prob[:, 1], eps=eps)
    metric_values[6] = metrics.roc_auc_score(y_test, y_predict_prob[:, 1])

    perf_metrics = dict(zip(metric_keys, metric_values))

    return(perf_metrics)


# In[ ]:


gbm_scores = get_performance_scores(y_test, gbm_y_predict, gbm_y_predict_prob)
ensemble_methods_metrics.update({"GBM": gbm_scores})
print(ensemble_methods_metrics)


# In[ ]:


gbm_roc = metrics.plot_roc_curve(gbm, X_test, y_test, name = "GBM")
plt.show()


# In[ ]:


pickle.dump(gbm, open("nhc_covid19_gbmmodel.sav","wb" ))


# ## Optimized Gradient Boosting

# In[ ]:


gbm.get_params()


# In[ ]:


# Randomized CV for GBM optimization: parameters 

gbm = GradientBoostingClassifier()

# Number of trees in random forest.
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 20)]

# Number of features to consider at every split.
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree.
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)

# Minimum number of samples required to split a node.
min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node.
min_samples_leaf = [1, 2, 4]

# Define learning rate parameters.
learning_rate = [0.001, 0.01, 0.1, 0.2, 0.3]

# Create the random grid.
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'learning_rate': learning_rate}


# In[ ]:


#Randomized CV for GBM optimization: fit 

# Initialize the randomized search model
gbm_random = RandomizedSearchCV(estimator = gbm,
                               param_distributions = random_grid,
                               n_iter = 100,
                               cv = 3,
                               verbose = 0,
                               random_state = 1,
                               n_jobs = -1)

Fit the random search model.
gbm_random.fit(X_train, y_train)


# In[ ]:


# Load pre-saved randomized search CV model.
gbm_random = pickle.load(open("gbm_random.sav","rb"))
gbm_random.best_params_


# In[ ]:


gbm_random = pickle.load(open("nhc_covid19_gbm_test.csv","rb"))


# In[ ]:


#Implement optimized GBM model

# Pass parameters from randomized search to GBM classifier.
optimized_gbm = GradientBoostingClassifier(**gbm_random.best_params_)

# Fit model to train data.
optimized_gbm.fit(X_train, y_train)


# In[ ]:


# Predict and evaluate optimized GBM model

# Get class predictions.
optimized_gbm_y_predict = gbm_random.predict(X_test)

# Get prediction probabilities.
optimized_gbm_y_predict_proba = optimized_gbm.predict_proba(X_test)

# Compute performance metrics.
optimized_gbm_scores = get_performance_scores(y_test,
optimized_gbm_y_predict,
optimized_gbm_y_predict_proba)


# In[ ]:


optimized_gbm_scores = get_performance_scores(y_test, optimized_gbm_y_predict, optimized_gbm_y_predict_proba)


# In[ ]:


# Take a look at test data confusion matrix.
conf_matrix_boosting_opt = metrics.confusion_matrix(y_test, optimized_gbm_y_predict)
print(conf_matrix_boosting_opt)
# Compute test model accuracy score.
accuracy_gbm_opt = metrics.accuracy_score(y_test, optimized_gbm_y_predict)
print('Accuracy of gbm on test data: ', accuracy_gbm_opt)


# In[ ]:


plt.clf()
plt.imshow(conf_matrix_boosting_opt, interpolation='nearest', cmap=plt.cm.tab20)
classNames = ['Negative','Positive']
plt.title('Nursing Home Compare Outbreak GBM Confusion Matrix - Test Data')
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j-0.25,i, str(s[i][j])+" = "+str(conf_matrix_boosting_opt[i][j]))
plt.show()


# In[ ]:


# Compute accuracy using training data.
train_accuracy_gbm_opt = optimized_gbm.score(X_train, y_train)

print ("Train Accuracy:", train_accuracy_gbm_opt)


# In[ ]:


# Create a dictionary with accuracy values for model 
model_final_dict = {'metrics': ["accuracy"],
               'values':[round(accuracy_gbm,4)],
                'model':['train_gbm_20200628']}
model_final = pd.DataFrame(data = model_final_dict)
print(model_final)


# In[ ]:


nhc_gbm_features_opt = nhc_rf_test.drop('cdc_outbreak', axis = 1)
features = nhc_gbm_features_opt.columns
importances = optimized_gbm.feature_importances_
indices = np.argsort(importances)[::-1]
top_indices = indices[0:10][::-1]

plt.figure(1)
plt.title('Feature Importance')
plt.barh(range(len(top_indices)), importances[top_indices], color = 'b', align = 'center')
labels = features[top_indices]
labels = [ '\n'.join(wrap(l,30)) for l in labels ]
plt.yticks(range(len(top_indices)), labels)
plt.xlabel('Relative Importance')


# In[ ]:


ensemble_methods_metrics.update({"Optimized GBM": optimized_gbm_scores})
print(ensemble_methods_metrics)


# In[ ]:


# ROC curve
lw = 2
ax = plt.gca()
opt_gbm_roc = metrics.plot_roc_curve(optimized_gbm,
                                     X_test,
                                     y_test,
                                     ax = ax,
                                     name = "Optimized GBM")

rf_roc.plot(ax = ax, name = "RF")
opt_rf_roc.plot(ax = ax, name = "Optimized RF")
gbm_roc.plot(ax = ax, name = "GBM")
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')

plt.show()


# In[ ]:


lw = 2
ax = plt.gca()
opt_gbm_roc = metrics.plot_roc_curve(optimized_gbm,
                                     X_test,
                                     y_test,
                                     ax = ax,
                                     name = "Optimized GBM")


opt_rf_roc.plot(ax = ax, name = "Optimized RF")

plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')

plt.show()


# In[ ]:


# model compare


# In[ ]:


# Convert all metrics for each model to a dataframe.
ensemble_methods_metrics_df = pd.DataFrame(ensemble_methods_metrics)
ensemble_methods_metrics_df["metric"] = ensemble_methods_metrics_df.index
ensemble_methods_metrics_df = ensemble_methods_metrics_df.reset_index(drop = True)
print(ensemble_methods_metrics_df)


# # Predict 6/14

# In[ ]:


# Keep period = 2 or 3
array2 = [4]
nhc_pred0614 = nhc_rf.loc[nhc_rf['period'].isin(array2)]


# In[ ]:


nhc_pred0614.head(5)


# In[ ]:


totcase_week


# In[ ]:


# Set the target week we are trying to predict
targetweek = 'usa_totcase_wk0614'
targetweek_cntg = 'usa_totcase_wk0614_cntg'


# In[ ]:


# Create function to find vars for 3 to 7 weeks before target
def find_adjacents(value, items, x):
    i = items.index(value)
    return items[i-x]


# In[ ]:


# Store function output as var for each week
wago3_0614 = (find_adjacents(targetweek, totcase_week, 3))
wago4_0614 = (find_adjacents(targetweek, totcase_week, 4))
wago5_0614 = (find_adjacents(targetweek, totcase_week, 5))
wago6_0614 = (find_adjacents(targetweek, totcase_week, 6))
wago7_0614 = (find_adjacents(targetweek, totcase_week, 7))


# In[ ]:


wago3_cntg_0614 = (find_adjacents(targetweek_cntg, totcase_week, 3))
wago4_cntg_0614 = (find_adjacents(targetweek_cntg, totcase_week, 4))
wago5_cntg_0614 = (find_adjacents(targetweek_cntg, totcase_week, 5))
wago6_cntg_0614 = (find_adjacents(targetweek_cntg, totcase_week, 6))
wago7_cntg_0614 = (find_adjacents(targetweek_cntg, totcase_week, 7))


# In[ ]:


# Print to check the weeks are correct
print("6/14/2020")
print(wago3_0614) 
print(wago4_0614) 
print(wago5_0614) 
print(wago6_0614) 
print(wago7_0614) 

print(wago3_cntg_0614)
print(wago4_cntg_0614) 
print(wago5_cntg_0614) 
print(wago6_cntg_0614) 
print(wago7_cntg_0614) 


# In[ ]:


# Add the week vars to the data set
nhc_pred0614['case_3wago'] =   nhc_pred0614[wago3_0614] 
nhc_pred0614['case_4wago'] =   nhc_pred0614[wago4_0614]
nhc_pred0614['case_5wago'] =   nhc_pred0614[wago5_0614] 
nhc_pred0614['case_6wago'] =   nhc_pred0614[wago6_0614] 
nhc_pred0614['case_7wago'] =   nhc_pred0614[wago7_0614]

nhc_pred0614['case_3wago_cntg'] =   nhc_pred0614[wago3_cntg_0614] 
nhc_pred0614['case_4wago_cntg'] =   nhc_pred0614[wago4_cntg_0614] 
nhc_pred0614['case_5wago_cntg'] =   nhc_pred0614[wago5_cntg_0614] 
nhc_pred0614['case_6wago_cntg'] =   nhc_pred0614[wago6_cntg_0614]
nhc_pred0614['case_7wago_cntg'] =   nhc_pred0614[wago7_cntg_0614]


# In[ ]:


nhc_pred0614.head(5)


# In[ ]:


nhc_pred0614 = nhc_pred0614.drop(['usa_totcase_wk0308',
 'usa_totcase_wk0315',
 'usa_totcase_wk0322',
 'usa_totcase_wk0329',
 'usa_totcase_wk0405',
 'usa_totcase_wk0412',
 'usa_totcase_wk0419',
 'usa_totcase_wk0426',
 'usa_totcase_wk0503',
 'usa_totcase_wk0510',
 'usa_totcase_wk0517',
 'usa_totcase_wk0524',
 'usa_totcase_wk0531',
 'usa_totcase_wk0607',
 'usa_totcase_wk0614',
 'usa_totcase_wk0621',
 'usa_totcase_wk0308_cntg',
 'usa_totcase_wk0315_cntg',
 'usa_totcase_wk0322_cntg',
 'usa_totcase_wk0329_cntg',
 'usa_totcase_wk0405_cntg',
 'usa_totcase_wk0412_cntg',
 'usa_totcase_wk0419_cntg',
 'usa_totcase_wk0426_cntg',
 'usa_totcase_wk0503_cntg',
 'usa_totcase_wk0510_cntg',
 'usa_totcase_wk0517_cntg',
 'usa_totcase_wk0524_cntg',
 'usa_totcase_wk0531_cntg',
 'usa_totcase_wk0607_cntg',
 'usa_totcase_wk0614_cntg',
 'usa_totcase_wk0621_cntg',
 'period',
 'cdc_outbreak'], axis = 1)


# In[ ]:


# Predict an outbreak with best model here
nhc_pred0614_pred = optimized_forest.predict(nhc_pred0614)
print(nhc_pred0614_pred[0:5,])


# In[ ]:


optimized_forest.classes_


# In[ ]:


# Predict an outbreak with best model here
nhc_pred0614_prob = optimized_forest.predict_proba(nhc_pred0614)[:, 1]


# In[ ]:


nhc_pred0614_prob


# In[ ]:


# Append predictions to dataset
nhc_pred0614['pred_outbreak_0614'] = nhc_pred0614_pred


# In[ ]:


nhc_pred0614['pred_outbreak_prob_0614'] = nhc_pred0614_prob


# In[ ]:


nhc_pred0614


# # Predict 7/12

# In[ ]:


# Keep period = 4
array2 = [4]
nhc_pred0712 = nhc_rf.loc[nhc_rf['period'].isin(array2)]


# In[ ]:


nhc_pred0712.head(5)


# In[ ]:


totcase_week


# In[ ]:


# Set the target week we are trying to predict
targetweek = 'usa_totcase_wk0621'
targetweek_cntg = 'usa_totcase_wk0621_cntg'


# In[ ]:


# Store function output as var for each week
wago3_0712 = (find_adjacents(targetweek, totcase_week, 0))
wago4_0712 = (find_adjacents(targetweek, totcase_week, 1))
wago5_0712 = (find_adjacents(targetweek, totcase_week, 2))
wago6_0712 = (find_adjacents(targetweek, totcase_week, 3))
wago7_0712 = (find_adjacents(targetweek, totcase_week, 4))


# In[ ]:


wago3_cntg_0712 = (find_adjacents(targetweek_cntg, totcase_week, 0))
wago4_cntg_0712 = (find_adjacents(targetweek_cntg, totcase_week, 1))
wago5_cntg_0712 = (find_adjacents(targetweek_cntg, totcase_week, 2))
wago6_cntg_0712 = (find_adjacents(targetweek_cntg, totcase_week, 3))
wago7_cntg_0712 = (find_adjacents(targetweek_cntg, totcase_week, 4))


# In[ ]:


# Print to check the weeks are correct
print("7/12/2020")
print(wago3_0712) 
print(wago4_0712) 
print(wago5_0712) 
print(wago6_0712) 
print(wago7_0712) 

print(wago3_cntg_0712)
print(wago4_cntg_0712) 
print(wago5_cntg_0712) 
print(wago6_cntg_0712) 
print(wago7_cntg_0712) 


# In[ ]:


# Add the week vars to the data set
nhc_pred0712['case_3wago'] =   nhc_pred0712[wago3_0712] 
nhc_pred0712['case_4wago'] =   nhc_pred0712[wago4_0712]
nhc_pred0712['case_5wago'] =   nhc_pred0712[wago5_0712] 
nhc_pred0712['case_6wago'] =   nhc_pred0712[wago6_0712] 
nhc_pred0712['case_7wago'] =   nhc_pred0712[wago7_0712]

nhc_pred0712['case_3wago_cntg'] =   nhc_pred0712[wago3_cntg_0712] 
nhc_pred0712['case_4wago_cntg'] =   nhc_pred0712[wago4_cntg_0712] 
nhc_pred0712['case_5wago_cntg'] =   nhc_pred0712[wago5_cntg_0712] 
nhc_pred0712['case_6wago_cntg'] =   nhc_pred0712[wago6_cntg_0712]
nhc_pred0712['case_7wago_cntg'] =   nhc_pred0712[wago7_cntg_0712]


# In[ ]:


nhc_pred0712.head(5)


# In[ ]:


nhc_pred0712 = nhc_pred0712.drop(['usa_totcase_wk0308',
 'usa_totcase_wk0315',
 'usa_totcase_wk0322',
 'usa_totcase_wk0329',
 'usa_totcase_wk0405',
 'usa_totcase_wk0412',
 'usa_totcase_wk0419',
 'usa_totcase_wk0426',
 'usa_totcase_wk0503',
 'usa_totcase_wk0510',
 'usa_totcase_wk0517',
 'usa_totcase_wk0524',
 'usa_totcase_wk0531',
 'usa_totcase_wk0607',
 'usa_totcase_wk0614',
 'usa_totcase_wk0621',
 'usa_totcase_wk0308_cntg',
 'usa_totcase_wk0315_cntg',
 'usa_totcase_wk0322_cntg',
 'usa_totcase_wk0329_cntg',
 'usa_totcase_wk0405_cntg',
 'usa_totcase_wk0412_cntg',
 'usa_totcase_wk0419_cntg',
 'usa_totcase_wk0426_cntg',
 'usa_totcase_wk0503_cntg',
 'usa_totcase_wk0510_cntg',
 'usa_totcase_wk0517_cntg',
 'usa_totcase_wk0524_cntg',
 'usa_totcase_wk0531_cntg',
 'usa_totcase_wk0607_cntg',
 'usa_totcase_wk0614_cntg',
 'usa_totcase_wk0621_cntg',                                
 'period',
 'cdc_outbreak'], axis = 1)


# In[ ]:


# Predict an outbreak with best model here
nhc_pred0712_pred = optimized_forest.predict(nhc_pred0712)
print(nhc_pred0712_pred[0:5,])


# In[ ]:


optimized_forest.classes_


# In[ ]:


# Predict an outbreak with best model here
nhc_pred0712_prob = optimized_forest.predict_proba(nhc_pred0712)[:, 1]


# In[ ]:


nhc_pred0712_prob


# In[ ]:


# Append predictions to dataset
nhc_pred0712['pred_outbreak_0712'] = nhc_pred0712_pred


# In[ ]:


nhc_pred0712['pred_outbreak_prob_0712'] = nhc_pred0712_prob


# In[ ]:


nhc_pred0712


# In[ ]:


# Keep period = 2-4
array = [4]
nhc_final = nhc_orig.loc[nhc_orig['period'].isin(array)]


# In[ ]:


# Append predictions to dataset
nhc_final['pred_outbreak_0614'] = nhc_pred0614_pred


# In[ ]:


nhc_final['pred_outbreak_prob_0614'] = nhc_pred0614_prob


# In[ ]:


# Append predictions to dataset
nhc_final['pred_outbreak_0712'] = nhc_pred0712_pred


# In[ ]:


nhc_final['pred_outbreak_prob_0712'] = nhc_pred0712_prob


# In[ ]:


nhc_final.pred_outbreak_0614.value_counts()


# In[ ]:


nhc_final.pred_outbreak_0712.value_counts()


# In[ ]:


# Convert target var to bool
nhc_final['cdc_outbreak'] = np.where(nhc_final['cdc_outbreak'] == 1 , True, False)


# In[ ]:


nhc_final['correct_pred'] = np.where(nhc_final['cdc_outbreak']==nhc_final['pred_outbreak_0614'], "Yes", "No")


# In[ ]:


nhc_final.correct_pred.value_counts()


# In[ ]:


cols_at_end = ['cdc_outbreak', 'pred_outbreak_0614', 'pred_outbreak_prob_0614', 'correct_pred', 'pred_outbreak_0712', 'pred_outbreak_prob_0712']
nhc_final = nhc_final[[c for c in nhc_final if c not in cols_at_end]+ [c for c in cols_at_end if c in nhc_final]]


# In[ ]:


nhc_final.to_csv("nhc_covid19_rf_final.csv")


# In[ ]:


nhc_final


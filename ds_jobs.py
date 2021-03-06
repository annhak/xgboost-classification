"""
Predict whether a data scientist would be eager to change jobs.
Author: Anna Hakala
"""

import pandas as pd
from sklearn import model_selection
from sklearn.feature_selection import SelectFromModel
import xgboost

import tools

# Dataset from Kaggle
data = pd.read_csv("./aug_train.csv")

# Clean data by transforming experience years to numeric
data.experience[data['experience'] == '>20'] = 21
data.experience[data['experience'] == '<1'] = 0
data.experience = pd.to_numeric(data.experience)

# Transform also company size to numeric
data['company_size'] = pd.to_numeric(data['company_size'].map(
    {'<10': 1, '10/49': 10, '50-99': 50, '100-500': 100, '500-999': 500,
     '1000-4999': 1000, '5000-9999': 5000, '10000+': 10000}))

# Encode labels for categorical columns. Pandas get_dummies() works as one-hot
# encoder but works well with pandas dfs. Leave non-interesting columns out
info = pd.get_dummies(data.drop(['enrollee_id', 'target', 'city'], axis=1))

# The new column names can include symbols the model cannot handle. Replace them
info.columns = info.columns.str.replace('>', 'over')
info.columns = info.columns.str.replace('<', 'under')
# What to predict: Is the person willing to change jobs (0 / 1)
target = data['target']

# Split data into train and test sets
info_train, info_test, target_train, target_test = model_selection.\
    train_test_split(info, target, test_size=0.3, random_state=1)

# Create a classifier and test the dataset with cross validation
classifier_xgb = xgboost.XGBClassifier(
    max_depth=6, random_state=1, use_label_encoder=False)
accuracies = model_selection.cross_val_score(classifier_xgb, info, target)
print(f'Model accuracies with 5-fold CV: {accuracies}')

# K-fold cross-validation gives ~same results -> dataset OK -> fit and predict
classifier_xgb.fit(info_train, target_train)

predicted_test = classifier_xgb.predict(info_test)
tools.print_results(target_test, predicted_test)

# Plot one of the resulting trees to examine what happens inside
xgboost.plot_tree(classifier_xgb, rankdir='LR')
# Plot feature importance to see what are the most relevant features
xgboost.plot_importance(classifier_xgb, importance_type='weight')
xgboost.plot_importance(classifier_xgb, importance_type='cover')
xgboost.plot_importance(classifier_xgb, importance_type='gain')

# Select only best features for the final model and closer examination
selection = SelectFromModel(classifier_xgb, threshold=0.03, prefit=True)
feature_names = info_train.columns[selection.get_support()]
print(f'{len(feature_names)} most important features:  {str(feature_names)}')

# Get only the selected features for further training and testing
train_features = pd.DataFrame(selection.transform(info_train))
train_features.columns = feature_names
test_features = pd.DataFrame(selection.transform(info_test))
test_features.columns = feature_names

# Create a new classifier as the old one was for all features
predictor = xgboost.XGBClassifier(
    max_depth=3, random_state=1, use_label_encoder=False)
predictor.fit(train_features, target_train)
predicted = predictor.predict(test_features)
tools.print_results(target_test, predicted)

# Look at a treeplot to see paths of one tree
xgboost.plot_tree(predictor, rankdir='LR')

"""
import matplotlib.pyplot as plt
import numpy as np
for i in feature_names:
    if info[i].max() != 1:
        continue
    dat = info[i]
    false_and_wants = round((dat[target == 1] == 0).sum() / len(dat.index), 2)
    true_and_wants = round(dat[target == 1].sum() / len(dat.index), 2)
    false_and_wants_not = round(
        (dat[target == 0] == 0).sum() / len(dat.index), 2)
    true_and_wants_not = round(dat[target == 0].sum() / len(dat.index), 2)
    has_true = round(info[i].sum() / len(dat.index), 2)

    print(str(i) + str(true_and_wants / (true_and_wants + true_and_wants_not)))
    print(str(i) + ' ' + str([[false_and_wants, true_and_wants], 
        [false_and_wants_not, true_and_wants_not]]))

    print(i + ' ' + str(true_and_wants) + ' ' + str(true_and_wants_not) + ' ' + 
    str(true_and_wants / has_true))

column = 'city_development_index'
results = []
for i in sorted(data[column].astype('category').unique()):
    try:
        d = data[data[column] == i]
        perc = round(sum(d['target'] == 1) / len(d.index), 3)
        print(str(i) + ': ' + str(perc))
        results.append(perc)
    except Exception as ex:
        print(ex)
        continue

plt.figure()
plt.plot(results)

nonnan = data[np.isfinite(data['experience'])]
plt.figure()
plt.hist(nonnan['experience'], bins=22)

"""
"""
Predict whether a data scientist would be eager to change jobs.
Author: Anna Hakala
"""

import pandas as pd
from sklearn import model_selection
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder
import xgboost
import matplotlib.pyplot as plt

# Note: you need to install graphviz on your machine if you want to plot trees
import tools

# Dataset from Kaggle
data = pd.read_csv("./xgboost-classification/data/aug_train.csv")

# Prepare the dataset #####

# Clean data by transforming experience years to numeric
# This is done by manually investigating the data, as the dataset is not clean
data.experience[data['experience'] == '>20'] = 21
data.experience[data['experience'] == '<1'] = 0
data.experience = pd.to_numeric(data.experience)

# Transform also company size to numeric
data['company_size'] = pd.to_numeric(data['company_size'].map(
    {'<10': 1, '10/49': 10, '50-99': 50, '100-500': 100, '500-999': 500,
     '1000-4999': 1000, '5000-9999': 5000, '10000+': 10000}))

# Last new job to numeric
data['last_new_job'] = pd.to_numeric(data['last_new_job'].map(
    {'1': 1, '2': 2, '3': 3, '4': 4, '>4': 5, 'never': 6}
))

data['enrolled_university'] = pd.to_numeric(data['enrolled_university'].map({
    'no_enrollment': 0, 'Part time course': 1,'Full time course': 2}))

data['education_level'] = pd.to_numeric(data['education_level'].map({
    'Primary School': 0, 'High School': 1, 'Graduate': 2, 'Masters': 3, 'Phd': 4}))

# Encode columns with a lot of options. Each category (e.g. STEM) gets a number
data[['major_discipline', 'company_type']] = data[
    ['major_discipline', 'company_type']].apply(LabelEncoder().fit_transform)

# Encode labels for categorical columns. Pandas get_dummies() works as one-hot
# encoder but works well with pandas dfs. Leave non-interesting columns out
features = pd.get_dummies(data.drop(['enrollee_id', 'target', 'city'], axis=1))

# Show an example about label encoding and what happens to missing
tools.encoding_example(data, features)

# The new column names can include symbols the model cannot handle. Replace them
features.columns = features.columns.str.replace('>', 'over')
features.columns = features.columns.str.replace('<', 'under')
# What to predict: Is the person willing to change jobs (0 / 1)
target = data['target']

# Move to model training #####

# Split data into train and test sets
features_train, features_test, target_train, target_test = model_selection.\
    train_test_split(features, target, test_size=0.3, random_state=1)

# Stop here to show what an example tree could contain and look like
# tools.tree_example(features_train, target_train)

# Create a classifier and test the dataset with cross validation
classifier_xgb = xgboost.XGBClassifier(
    max_depth=6, random_state=1, use_label_encoder=False, eval_metric='logloss')
accuracies = model_selection.cross_val_score(
    classifier_xgb, features, target, cv=100, verbose=True, n_jobs=-1)
# print(f'Model accuracies with 5-fold CV: {accuracies}')
pd.DataFrame(accuracies).hist()

# K-fold cross-validation gives ~same results -> dataset OK -> fit and predict
classifier_xgb.fit(features_train, target_train)

predicted_test = classifier_xgb.predict(features_test)
tools.print_results(target_test, predicted_test)

# Plot one of the resulting trees to examine what happens inside
# xgboost.plot_tree(classifier_xgb, rankdir='LR')

# Plot feature importance to see what are the most relevant features
plt.figure()
pd.Series(
    abs(classifier_xgb.feature_importances_),
    index=features_train.columns).nlargest(35).plot(kind='barh')

# Select only best features for the final model and closer examination
selection = SelectFromModel(classifier_xgb, threshold=0.05, prefit=True)
feature_names = features_train.columns[selection.get_support()]
print(f'{len(feature_names)} most important features:  {str(feature_names)}')

# Get only the selected features for further training and testing
train_features = pd.DataFrame(selection.transform(features_train))
train_features.columns = feature_names
test_features = pd.DataFrame(selection.transform(features_test))
test_features.columns = feature_names

# Create a new classifier as the old one was for all features
predictor = xgboost.XGBClassifier(
    max_depth=3, random_state=1, use_label_encoder=False)
predictor.fit(train_features, target_train)
predicted = predictor.predict(test_features)
tools.print_results(target_test, predicted)

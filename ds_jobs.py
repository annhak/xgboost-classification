"""
Predict whether a data scientist would be eager to change jobs.
Author: Anna Hakala
"""

import pandas as pd
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectFromModel
import xgboost

import tools


def load_and_label_data(path):
    # Dataset from Kaggle
    data = pd.read_csv(path)

    # Clean data by transforming experience years to numeric
    data.experience[data['experience'] == '>20'] = 21
    data.experience[data['experience'] == '<1'] = 0
    data.experience = pd.to_numeric(data.experience)

    # Transform also company size to numeric
    data['company_size'] = pd.to_numeric(data['company_size'].map(
        {'<10': 1, '10/49': 10, '50-99': 50, '100-500': 100, '500-999': 500,
         '1000-4999': 1000, '5000-9999': 5000, '10000+': 10000}))

    # Encode categorical columns. Pandas get_dummies() works as one-hot
    # encoder but works well with pandas dfs. Leave non-interesting columns out
    info = pd.get_dummies(data.drop(['enrollee_id', 'target', 'city'], axis=1))

    # New column names can include symbols the model cannot handle. Replace them
    info.columns = info.columns.str.replace('>', 'over')
    info.columns = info.columns.str.replace('<', 'under')
    # What to predict: Is the person willing to change jobs (0 / 1)
    target = data['target']
    return info, target


def validate_and_split(info, target):
    # Create a classifier and test the dataset with cross validation
    classifier_xgb = xgboost.XGBClassifier(
        max_depth=6, random_state=1, use_label_encoder=False)
    accuracies = model_selection.cross_val_score(classifier_xgb, info, target)
    print(f'Model accuracies with 5-fold CV: {accuracies}')

    # Split data into train and test sets
    info_train, info_test, target_train, target_test = model_selection.\
        train_test_split(info, target, test_size=0.3, random_state=1)
    return info_train, info_test, target_train, target_test


def feature_selection(info_train, info_test, target_train, target_test):
    # K-fold cv gives ~same results -> dataset OK -> fit and predict
    classifier_xgb = xgboost.XGBClassifier(
        max_depth=6, random_state=1, use_label_encoder=False)
    classifier_xgb.fit(info_train, target_train)

    predicted_test = classifier_xgb.predict(info_test)
    [tn, fp], [fn, tp] = confusion_matrix(target_test, predicted_test)
    tools.print_results(target=target_test, predicted=predicted_test)

    # Plot one of the resulting trees to examine what happens inside
    xgboost.plot_tree(classifier_xgb, rankdir='LR')
    # Plot feature importance to see what are the most relevant features
    xgboost.plot_importance(classifier_xgb, importance_type='weight')
    xgboost.plot_importance(classifier_xgb, importance_type='cover')
    xgboost.plot_importance(classifier_xgb, importance_type='gain')
#    plt.figure()
#    pd.Series(abs(classifier.coef_[0]), index=info_train.columns).plot(
#        kind='barh')

    # Select only best features for the final model and closer examination
    selection = SelectFromModel(classifier_xgb, threshold=0.03, prefit=True)
    feature_names = info_train.columns[selection.get_support()]
    print(f'{len(feature_names)} most important columns:  {str(feature_names)}')
    return selection


def pick_features(selection, data):
    # Get only the selected features for further training and testing
    feature_names = data.columns[selection.get_support()]
    features = pd.DataFrame(selection.transform(data))
    features.columns = feature_names
    return features


def create_labeler(train_features, test_features, target_train, target_test):
    # Create a new classifier as the old one was for all features
    predictor = xgboost.XGBClassifier(
        max_depth=3, random_state=1, use_label_encoder=False)
    predictor.fit(train_features, target_train)
    predicted = predictor.predict(test_features)

    tools.print_results(target=target_test, predicted=predicted)

    # Look at a tree plot to see paths of one tree
    xgboost.plot_tree(predictor, rankdir='LR')
    return predictor


def train():
    info, target = load_and_label_data("./aug_train.csv")
    info_train, info_test, target_train, target_test = validate_and_split(
        info, target)
    selection = feature_selection(
        info_train, info_test, target_train, target_test)
    train_features = pick_features(selection, info_train)
    test_features = pick_features(selection, info_test)
    predictor = create_labeler(
        train_features, test_features, target_train, target_test)
    return selection, predictor


def predict(selection, predictor):
    info, target = load_and_label_data('./aug_train.csv')
    predicted = predictor.predict(pick_features(selection, info))
    tools.print_results(target, predicted)


if __name__ == '__main__':
    feature_selection, final_predictor = train()
    predict(feature_selection, final_predictor)

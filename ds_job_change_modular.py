"""
Predict whether a data scientist would be eager to change jobs.
Author: Anna Hakala
"""

import pandas as pd
from sklearn import model_selection
from sklearn.feature_selection import SelectFromModel
import xgboost

import tools


def load_and_label_data(path):
    # Dataset from Kaggle
    dataset = pd.read_csv(path)

    # Clean dataset by transforming experience years to numeric
    dataset.experience[dataset['experience'] == '>20'] = 21
    dataset.experience[dataset['experience'] == '<1'] = 0
    dataset.experience = pd.to_numeric(dataset.experience)

    # Transform also company size to numeric
    dataset['company_size'] = pd.to_numeric(dataset['company_size'].map(
        {'<10': 1, '10/49': 10, '50-99': 50, '100-500': 100, '500-999': 500,
         '1000-4999': 1000, '5000-9999': 5000, '10000+': 10000}))

    if not pd.api.types.is_numeric_dtype(dataset.last_new_job):
        dataset['last_new_job'] = pd.to_numeric(dataset['last_new_job'].map(
            {'1': 1, '2': 2, '3': 3, '4': 4, '>4': 5, 'never': 6}
        ))

    # Encode categorical columns. Pandas get_dummies() works as one-hot
    # encoder but works well with pandas dfs. Leave non-interesting columns out
    info = pd.get_dummies(dataset.drop(
        ['enrollee_id', 'target', 'city'], axis=1))

    # New column names can include symbols the model cannot handle. Replace them
    info.columns = info.columns.str.replace('>', 'over')
    info.columns = info.columns.str.replace('<', 'under')
    col_order = sorted(info.columns.tolist())
    info = info[col_order]
    # What to predict: Is the person willing to change jobs (0 / 1)
    target = dataset['target']
    all_feature_names = info.columns
    return info, target, all_feature_names


def validate_and_split(info, target):
    # Create a classifier and cross validate the dataset
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
    tools.print_results(target=target_test, predicted=predicted_test)

#    plt.figure()
#    pd.Series(abs(classifier.coef_[0]), index=info_train.columns).plot(
#        kind='barh')

    # Select only best features for the final model and closer examination
    selection = SelectFromModel(classifier_xgb, threshold=0.03, prefit=True)
    feature_names = info_train.columns[selection.get_support()]
    print(f'{len(feature_names)} most important columns:  {str(feature_names)}')
    return selection, feature_names


def pick_features(selection, df):
    # Get only the selected features for further training and testing
    feature_names = df.columns[selection.get_support()]
    features = pd.DataFrame(selection.transform(df))
    features.columns = feature_names
    return features


def create_labeler(train_features, test_features, target_train, target_test):
    # Create a new classifier as the old one was for all features
    predictor = xgboost.XGBClassifier(
        max_depth=3, random_state=1, use_label_encoder=False)
    predictor.fit(train_features, target_train)
    predicted_results = predictor.predict(test_features)

    tools.print_results(target=target_test, predicted=predicted_results)

    # Look at a tree plot to see paths of one tree
    xgboost.plot_tree(predictor, rankdir='LR')
    return predictor


def train():
    info, target, all_feature_names = load_and_label_data(
        "./data/aug_train.csv")
    info_train, info_test, target_train, target_test = validate_and_split(
        info, target)
    selection, feature_names = feature_selection(
        info_train, info_test, target_train, target_test)
    train_features = pick_features(selection, info_train)
    test_features = pick_features(selection, info_test)
    predictor = create_labeler(
        train_features, test_features, target_train, target_test)
    return selection, predictor, target, all_feature_names


def predict(selection, predictor):
    info, target, _ = load_and_label_data('./data/aug_train.csv')
    predicted_results = predictor.predict(pick_features(selection, info))
    tools.print_results(target, predicted_results)


def select_columns(df, names):
    selected = [column for column in df.columns if column in names]
    df = df[selected]
    missing = [name for name in names if name not in df.columns]
    df[missing] = 0
    cols = sorted(df.columns.tolist())
    df = df[cols]
    return df


if __name__ == '__main__':
    selector, final_predictor, all_targets, features_from_training = train()

    anna, _, _ = load_and_label_data('./data/me_now.txt')
    data = select_columns(anna, features_from_training)

    predicted = final_predictor.predict(pick_features(selector, data))
    print(predicted)

#    predict(selector, final_predictor)

    all_ds, targets, feature_names_2 = load_and_label_data(
        './data/aug_train.csv')
    all_ds = select_columns(all_ds, feature_names_2)

    predicted = final_predictor.predict(
        pick_features(selector, all_ds))

    tools.print_results(targets, predicted)
    print(predicted)
    all_ds['score'] = predicted == targets

    wrong = all_ds[all_ds['score'] == 0]

    len(all_ds[all_ds['gender_Male'] == 1]) / len(all_ds)
    len(wrong[wrong['gender_Male'] == 1]) / len(wrong)

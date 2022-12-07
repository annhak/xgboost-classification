from sklearn.metrics import confusion_matrix
import xgboost
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'


def print_results(target, predicted):
    [_, fp], [_, tp] = confusion_matrix(target, predicted)

    data_scientists = len(target.index)
    wants_to_change = target.sum()
    # does_not_want = data_scientists - wants_to_change
    percentage = round(wants_to_change / data_scientists * 100, 1)

    print(f"""
    Without ML model:
    {data_scientists} calls to data scientists, out of which 
    {percentage} % would be beneficial, resulting in 
    {wants_to_change} data scientists changing jobs.""")

    to_call = tp + fp
    beneficial = round(tp / to_call * 100, 1)
    found_ds = tp
    found_percentage = round(found_ds / wants_to_change * 100, 1)
    called_percentage = round(to_call / data_scientists * 100, 1)

    print(f"""
    With ML model:
    {to_call} calls to data scientists, out of which
    {beneficial} % would be beneficial, resulting in
    {found_ds} data scientists changing jobs.
    
    Call {called_percentage} % of data scientists 
    -> find {found_percentage} % of the ones willing to change""")


def tree_example(info_train, target_train):
    # Create and fit a classifier to plot an example of a tree
    classifier_xgb = xgboost.XGBClassifier(
        max_depth=3, random_state=1, use_label_encoder=False)
    classifier_xgb.fit(info_train, target_train)

    # Plot one of the resulting trees to examine what happens inside
    xgboost.plot_tree(classifier_xgb, rankdir='LR')


def encoding_example(original, labelled):
    print('Unique values for gender: ' + str(pd.unique(original['gender'])))
    original = original['gender'].loc[43:47].reset_index(drop=True).to_frame()
    print('Example of the original column:')
    print(original)
    input('Press any key to continue')
    gender_columns = [col for col in labelled.columns if col.startswith('gender')]
    print('New gender columns after labelling:' + str(gender_columns))
    input('Press any key to continue')
    transformed = labelled[gender_columns].loc[43:47].reset_index(drop=True)
    print('Original and label encoded version:')
    print(original.join(transformed))

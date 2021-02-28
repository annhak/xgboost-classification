import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
import xgboost
import numpy as np

data = pd.read_csv("./aug_train.csv")

# fig, (ax1, ax2) = plt.subplots(2, 1)
# ax1.hist(data['target'])

data.experience[data['experience'] == '>20'] = 21
data.experience[data['experience'] == '<1'] = 0
data.experience = pd.to_numeric(data.experience)


target = data['target']
info = pd.get_dummies(data.drop(['enrollee_id', 'target', 'city'], axis=1))
info.columns = info.columns.str.replace('>', 'over')
info.columns = info.columns.str.replace('<', 'under')
info_train, info_test, target_train, target_test = model_selection.train_test_split(info, target, test_size=0.3, random_state=1)

xgb = xgboost.XGBClassifier(max_depth=3)
xgb.fit(info_train, target_train)
xgb.score(info_test, target_test)
xgboost.plot_tree(xgb, rankdir='LR')

model_selection.cross_val_score(xgb, info, target)

plt.figure()
pd.Series(abs(xgb.feature_importances_), index=info.columns).nlargest(40).plot(kind='barh')

column = 'company_size'
results = []
for i in sorted(data[column].astype('category').unique()):
    print(i)
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


import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm, preprocessing, model_selection, metrics

salaries = pd.read_csv("./data/employee_data.csv")

plt.figure()
plt.hist(salaries['salary'])
salaries['salary_category'] = \
    [int(salary/1000) for salary in salaries['salary']]
salaries.salary_category[salaries['salary_category'] > 4] = 4
plt.figure()
plt.hist(salaries['salary_category'])

encoder = preprocessing.OneHotEncoder(handle_unknown='ignore')
encoder.fit(salaries[['groups']])
encoded = encoder.transform(salaries[['groups']])

salaries = salaries.join(pd.DataFrame(encoded.toarray()))

info = salaries[['age', 'healthy_eating', 'active_lifestyle', 0, 1, 2, 3]]
target = salaries['salary_category']

info_train, info_test, target_train, target_test = \
    model_selection.train_test_split(
        info, target, test_size=0.3, random_state=1)

classifier = svm.SVC(kernel='linear')
classifier.fit(info_train, target_train)

predicted = classifier.predict(info_test)
print(metrics.accuracy_score(target_test, predicted))
predicted_train = classifier.predict(info_train)
print(metrics.accuracy_score(target_train, predicted_train))

plt.figure()
pd.Series(abs(classifier.coef_[0]), index=info_train.columns).plot(kind='barh')

plt.figure()
plt.plot(salaries['age'], salaries['salary_category'], 'p', color='navy')

plt.figure()
plt.plot(salaries['healthy_eating'], salaries['salary_category'],
         'p', color='navy')

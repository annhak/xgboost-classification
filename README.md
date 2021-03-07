# xgboost-classification

This is a hobby project for creating classifiers for two problems
* Would a data scientist be eager to change jobs based on knowledge about their current situation (binary classifier)
* Salary prediction based on few variables (multiclass classifier for salary group prediction)

### DS job change prediction
There are two versions of the code: ds_job_change.py and ds_job_change_modular.py. The first one is intended for a walk-through of what has been done and why, as the second one can be used to predict unseen rows from a file.
This project also utilizes tools.py to report results and demonstrate examples to be used in explaining the project. 

### Salary prediction
This was the first small project. The dataset was too small for meaningful feature selection, and as blood type was a better predictor for salary than age, it seemed that the dataset was fabricated -> not interesting.

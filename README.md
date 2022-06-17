This readme file briefly explains the function of each file.

The `/data` folder includes raw data, processed data, and intermediate predictions (i.e. trip distance predictions used as input to mode-choice prediction models).

`utils.py` contains common code that are reused throughtout several files.

### Data preprocessing
- `data_preprocessing.ipynb`

### Feature selection
- `feature_selection_mutual_info.ipynb`

### Compare different models for predicting trip distance
- `linear_regression.ipynb`
- `decision_tree.ipynb`
- `xgboost_distance.ipynb`

### Compare different models for predicting transport mode-choice
These two files compare XGBoost, Random forest, Naive bayes, Neural network models for predicting the mode-choice:
- `mode_choice.ipynb`:  Uses ground truth trip distance as input.
- `mode_choice_categorical.ipynb`: Uses categorized ground trip distance as input.

### XGboost variants for predicting trip distance
These files uses xgboost to predict trip distance:
- `xgboost_distance.ipynb`: predicts trip distance as continuous variable (without merged features).
- `xgboost_distance_category.ipynb`: predicts trip distance as 3 categories, outputs the softmax probilities (with merged features). 
- `xgboost_distance_log.ipynb`: predicts log (base 2) of the trip distance (with merged features).
- `xgboost_distance_merged_feature.ipynb`: predicts the trip distance as continuous variables (with merged features).
- `xgboost_distance_weighted.ipynb`: predicts the trip distance as continuous variables (with merged features). Add weights to the training samples. 

### XGBoost variants for predicting mode-choice
These files uses xgboost to predict mode-choice:
- `xgboost_mode.ipynb`: predicts mode choice with trip distance predicted from `xgboost_distance.ipynb` as input.
- `xgboost_mode_categorized_distance.ipynb`: predicts mode choice with trip distance softmax probabilities predicted from `xgboost_distance_category` as input.
- `xgboost_mode_groundtruth_merged.ipynb`: predicts mode choice with ground truth trip distance as input.
- `xgboost_mode_log.ipynb`:predicts mode choice with trip distance predicted from `xgboost_distance_log.ipynb` as input.
- `xgboost_mode_without_distance.ipynb`: predicts mode choice without predicted nor ground truth trip distance as input.

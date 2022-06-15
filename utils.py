from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, plot_confusion_matrix, log_loss
import matplotlib.pyplot as plt
import matplotlib

def mse_mae_mape(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    return mse, mae, mape




def acc_pre_rec(y_true, y_pred, verbose=False):
    ''' Returns accuracy, precision, and recall together. 
    If verbose is set to True, it prints the scores for 
    each mode.'''
    acc = accuracy_score(y_true, y_pred)
    prec, rec, fsc, sup = precision_recall_fscore_support(
        y_true, y_pred)
    
    if verbose:
        print(f'Accuracy: \n    {acc*100:.3f}%')
        scrs = {'Precision': prec, 'Recall': rec}
        for k, v in scrs.items():
            str_ = '%;\n    '.join(
                f'{TRANSPORT_MODES[i]} - {100*s:.3f}'
                for i, s in enumerate(v)
            )
            print(f"{k}: \n    {str_}%")
            
    return acc, prec, rec



def k_fold_cross_validation_regression(X, y, group, model, fold=5, weight=[]):

    k_fold = GroupKFold(n_splits=5)

    train_metrics  = [] #[(mse, mae, mape) for each fold]
    val_metrics = []  
    

    for train_idx, validate_idx in k_fold.split(X, y, groups=group):
        X_train, X_val = X[train_idx], X[validate_idx]
        y_train, y_val = y[train_idx], y[validate_idx]
        
        if len(weight) > 0:
            train_weight = weight[train_idx]
        else:
            train_weight = None
        
        
        model.fit(X_train, y_train, sample_weight=train_weight)
        
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        train_metrics.append(mse_mae_mape(y_train, y_train_pred))
        val_metrics.append(mse_mae_mape(y_val, y_val_pred))
    
    return train_metrics, val_metrics



# +
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def k_fold_cross_validation_classification(X, y, group, model, fold=5,
                                           confusion_matrix_save_path=None,
                                           display_labels=None):

    k_fold = GroupKFold(n_splits=5)

    train_metrics  = [] #[(accuracy, cross entropy) for each fold]
    val_metrics = []  
    

    for i, (train_idx, validate_idx) in enumerate(k_fold.split(X, y, groups=group)):
        X_train, X_val = X[train_idx], X[validate_idx]
        y_train, y_val = y[train_idx], y[validate_idx]
        
        model.fit(X_train, y_train)
        
        y_train_pred = model.predict(X_train)
        y_train_prob = model.predict_proba(X_train)
        
        y_val_pred = model.predict(X_val)
        y_val_prob = model.predict_proba(X_val)
        
        # training metrics
        acc, prec, rec = acc_pre_rec(y_train, y_train_pred)
        cross_entropy = log_loss(y_train, y_train_prob)
        train_metrics.append((acc, cross_entropy))
        
        # validation metrics
        acc, prec, rec = acc_pre_rec(y_val, y_val_pred)
        cross_entropy = log_loss(y_val, y_val_prob)
        val_metrics.append((acc, cross_entropy))
        
        if i == 0 and confusion_matrix_save_path != None:
            # save confusion metric

            ConfusionMatrixDisplay.from_predictions(y_val, y_val_pred, cmap=plt.cm.Blues,
                                                    display_labels=display_labels,
                                                    xticks_rotation=270
                                                    
            )
            plt.subplots_adjust(bottom=0.3)
            plt.savefig(confusion_matrix_save_path)

            
    
    return train_metrics, val_metrics
# -



from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ClassifierWrapper import ClassifierWrapper


def get_classifier(method, current_args,class_weight_dict, random_state=None, min_cases_for_training=30, hardcoded_prediction=0.5, binary=True):

    if method == "rf":
        tmp = current_args['max_features']
        final_tmp = float(tmp)
        return ClassifierWrapper(RandomForestClassifier(n_estimators=current_args['n_estimators'],
                                                       max_features=final_tmp,
                                                       random_state=random_state,
                                                     class_weight=class_weight_dict
                                                        ),
                                method=method,
                                # class_weight=class_weight_dict,

                                min_cases_for_training=min_cases_for_training,
                                hardcoded_prediction=hardcoded_prediction,
                                binary=binary)

    elif method == "xgboost":
        return ClassifierWrapper(xgb.XGBClassifier(objective='binary:logistic',
                                                  # scale_pos_weight=0.1, # traffic_fines_1(0.007,0.1)
                                                  # scale_pos_weight=0.32,  # BPIC2012(0.32,0.1)

                                                  n_estimators=current_args['n_estimators'],
                                                  learning_rate= current_args['learning_rate'],
                                                  subsample=current_args['subsample'],
                                                  max_depth=int(current_args['max_depth']),
                                                  colsample_bytree=current_args['colsample_bytree'],
                                                  min_child_weight=int(current_args['min_child_weight'])),
                                method=method,
                                #class_weight=class_weight_dict,
                                #scale_pos_weight
                                min_cases_for_training=min_cases_for_training,
                                hardcoded_prediction=hardcoded_prediction,
                                binary=binary)

    elif method == "logit":
        return ClassifierWrapper(LogisticRegression(C=2**current_args['C'],
                                                   random_state=random_state),
                                method=method,
                                min_cases_for_training=min_cases_for_training,
                                hardcoded_prediction=hardcoded_prediction,
                                binary=binary)

    elif method == "svm":
        return ClassifierWrapper(SVC(C=2**current_args['C'],
                                    gamma=2**current_args['gamma'],
                                    random_state=random_state), 
                                method=method,
                                min_cases_for_training=min_cases_for_training,
                                hardcoded_prediction=hardcoded_prediction,
                                binary=binary)
    
    
    elif method == "gbm":
        return ClassifierWrapper(
            cls=GradientBoostingClassifier(n_estimators=current_args['n_estimators'],
                                           max_features=current_args['max_features'],
                                           learning_rate=current_args['learning_rate'],
                                           random_state=random_state),
                                method=method,
                                min_cases_for_training=min_cases_for_training,
                                hardcoded_prediction=hardcoded_prediction,
                                binary=binary)
    
    elif method == "dt":
        return ClassifierWrapper(
                                cls=DecisionTreeClassifier(random_state=random_state),
                                method=method,
                                min_cases_for_training=min_cases_for_training,
                                hardcoded_prediction=hardcoded_prediction,
                                binary=binary)
    
    else:
        print("Invalid classifier type")
        return None
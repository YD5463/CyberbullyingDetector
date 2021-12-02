import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def xgboost_pipeline(x_train, y_train, x_test, y_test):
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25,
                                                      random_state=1)  # 0.25 x 0.8 = 0.2
    param = {"learning_rate": 0.05, "max_depth": 8, "min_child_weight": 1, "gamma": 0, "subsample": 0.7,
             "objective": 'binary:logistic', "scale_pos_weight": 1, "seed": 93, "eval_metric": "logloss"}
    # "colsample_bytree":0.8# evals=evals, early_stopping_rounds=10
    d_train = xgb.DMatrix(x_train.values, label=y_train.values)
    d_eval = xgb.DMatrix(x_val.values, label=y_val.values)
    evallist = [(d_eval, 'eval'), (d_train, 'train')]
    num_round = 1000
    bst = xgb.train(param, d_train, num_round, evallist)
    d_test = xgb.DMatrix(x_test.values)
    xgboost_pred = bst.predict(d_test)
    xgboost_pred[xgboost_pred > 0.5] = 1
    xgboost_pred[xgboost_pred <= 0.5] = 0
    print(classification_report(y_test, xgboost_pred))
    return bst


def lstm():
    pass


def cnn_with_word2vec():
    pass

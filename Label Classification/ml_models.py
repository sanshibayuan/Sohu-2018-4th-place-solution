import numpy as np, pandas as pd
from keras.utils import to_categorical, normalize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer  , TfidfVectorizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
import pickle
from sklearn.externals import joblib

def model_lgb(train_features,test_features,train_labels,size):
    lgbm_preds = np.zeros((size, 2))
    for i in range(0,2):
        print ('fit{}'.format(i))
        train_target = train_labels[:,i]
        model = LogisticRegression(solver='sag')
        sfm = SelectFromModel(model, threshold=0.2)
        print(train_features.shape)
        train_sparse_matrix = sfm.fit_transform(train_features, train_target)
        print(train_sparse_matrix.shape)
        train_sparse_matrix, valid_sparse_matrix, y_train, y_valid = train_test_split(train_sparse_matrix, train_target, test_size=0.05, random_state=144)
        test_sparse_matrix = sfm.transform(test_features)
        train_sparse_matrix, valid_sparse_matrix, y_train, y_valid = train_test_split(train_features, train_target, test_size=0.05, random_state=144)
        test_sparse_matrix = test_features
        d_train = lgb.Dataset(train_sparse_matrix, label=y_train)
        d_valid = lgb.Dataset(valid_sparse_matrix, label=y_valid)
        watchlist = [d_train, d_valid]
        params = {'learning_rate': 0.05,
                  'application': 'binary',
                  'num_leaves': 12,
#                  'verbosity': -1,
                  'metric': 'auc',
                  'data_random_seed': 4,
                  'bagging_fraction': 0.6,
                  'feature_fraction': 0.6,
                  'nthread': 4,
                  'lambda_l1': 1,
                  'lambda_l2': 1}
        rounds_lookup = [80,
                    # 180,
                     80
                     ]
        model = lgb.train(params,
                          train_set=d_train,
                          num_boost_round=rounds_lookup[i],
                          valid_sets=watchlist,
                          verbose_eval=20)
        print('save model')
        with open ('lgbm'+str(i)+'.pk','wb') as fin:
            pickle.dump(model,fin,protocol = 2)
            fin.close()  
        test_y = model.predict(test_sparse_matrix)
        lgbm_preds[:,i] = test_y
    return lgbm_preds


def model_nbsvm(train_features,test_features,train_labels,size):
    def pr(y_i, y):
        p = train_features[y==y_i].sum(0)
        return (p+1) / ((y==y_i).sum()+1)
    scores = []
    preds = np.zeros((size, 2))
    for i in range(0,2):
        print ('fit{}'.format(i))
        y = train_labels[:,i]
        r = np.log(pr(1,y) / pr(0,y))
        m = LogisticRegression(solver = 'sag')
        x_nb = train_features.multiply(r)
        m.fit(x_nb, y)
        joblib.dump(m, 'nbsvm'+ str(i) + '.model',protocol =2 )
        with open ('r'+str(i)+'.pk','wb') as fin:
            pickle.dump(r,fin,protocol = 2)
            fin.close()              
        test_y = m.predict_proba(test_features.multiply(r))
        preds[:,i] = test_y[:,1]
        cv_score = np.mean(cross_val_score(
            m, x_nb, y, cv=3, scoring='roc_auc'))
        scores.append(cv_score)
        print ('cv_score for label {} is {}'.format(i,cv_score))
    print('Total CV score is {}'.format(np.mean(scores)))
    return preds
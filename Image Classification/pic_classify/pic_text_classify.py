
# coding: utf-8

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
import random
from sentence_split import sent_after_split_word
from sentence_split import cut_stopwords
from sentence_split import load_stopwords
from sentence_split import cut_punctuation
import sys
import os
import joblib


# LGBM
def model_lgb(train_features,test_features,train_labels, model_path):

    lgbm_preds = np.zeros((test_features.shape[0], 2))
    train_target = train_labels[:,1]
    model = LogisticRegression(solver='sag')
    sfm = SelectFromModel(model, threshold=0.2)
    print(train_features.shape)
    train_sparse_matrix = sfm.fit_transform(train_features, train_target)

    sfm_path = os.path.join(os.path.split(model_path)[0], "lgb.sfm")
    joblib.dump(sfm, sfm_path)
    print(train_sparse_matrix.shape)
    train_sparse_matrix, valid_sparse_matrix, y_train, y_valid = train_test_split(train_sparse_matrix, train_target, test_size=0.05, random_state=144)

    test_sparse_matrix = sfm.transform(test_features)
    d_train = lgb.Dataset(train_sparse_matrix, label=y_train)
    d_valid = lgb.Dataset(valid_sparse_matrix, label=y_valid)
    watchlist = [d_train, d_valid]
    params = {'learning_rate': 0.2,
              'application': 'binary',
              'num_leaves': 31,
              'verbosity': -1,
              'metric': 'auc',
              'data_random_seed': 2,
              'bagging_fraction': 0.8,
              'feature_fraction': 0.6,
              'nthread': 4,
              'lambda_l1': 1,
              'lambda_l2': 1}

    model = lgb.train(params,
                          train_set=d_train,
                          num_boost_round=360,
                          valid_sets=watchlist,
                          verbose_eval=30)
    joblib.dump(model, model_path)

    test_y = model.predict(test_sparse_matrix)
    lgbm_preds[:,0] = 1-test_y
    lgbm_preds[:,1] = test_y
    return lgbm_preds

def lgb_predict(model_path, test_features):
    lgbm_preds = np.zeros((test_features.shape[0], 2))
    sfm_path = os.path.join(os.path.split(model_path)[0], "lgb.sfm")
    sfm = joblib.load(sfm_path)
    test_sparse_matrix = sfm.transform(test_features)

    m = joblib.load(model_path)
    test_y = m.predict(test_sparse_matrix)
    lgbm_preds[:,0] = 1-test_y
    lgbm_preds[:,1] = test_y
    return lgbm_preds

# NBSVM
def model_nbsvm(train_features,test_features,train_labels, model_path):
    def pr(y_i, y):
        p = train_features[y==y_i].sum(0)
        return (p+1) / ((y==y_i).sum()+1)
    scores = []
    y = train_labels[:,1]
    r = np.log(pr(1,y) / pr(0,y))
    
    r_path = os.path.join(os.path.split(model_path)[0], "nbsvm.r")
    joblib.dump(r, r_path)

    m = LogisticRegression(C=3,dual=True)
    x_nb = train_features.multiply(r)
    m.fit(x_nb, y)
    joblib.dump(m, model_path)

    test_y = m.predict_proba(test_features.multiply(r))
    preds = test_y
    cv_score = np.mean(cross_val_score(
        m, x_nb, y, cv=3, scoring='roc_auc'))
    scores.append(cv_score)
    print ('cv_score: {}'.format(cv_score))
    return preds

def nbsvm_predict(model_path, test_features):
    
    r_path = os.path.join(os.path.split(model_path)[0], "nbsvm.r")
    r = joblib.load(r_path)
    m = joblib.load(model_path)
    
    test_y = m.predict_proba(test_features.multiply(r))
    return test_y
    

# LR
def model_lr(train_features,test_features,train_labels, model_path): 
    scores = []
    y = train_labels[:,1]
    m = LogisticRegression(C =10, solver='sag') 
    cv_score = np.mean(cross_val_score(m, train_features, y, cv=3, scoring='roc_auc'))
    print ('cv_score: {}'.format(cv_score))
    scores.append(cv_score)
    m.fit(train_features, y)
    joblib.dump(m, model_path)

    test_y = m.predict_proba(test_features)
    lrpreds = test_y
    return lrpreds  

def lr_predict(model_path, test_features):
    m = joblib.load(model_path)
    test_y = m.predict_proba(test_features)
    return test_y 
    
def trans_result(preds):
    result = np.dot(preds, np.array([[-1],[1]]))
    result = (result > 0).astype(int)
    result = result.reshape(result.shape[0]).tolist()
    print (result.count(0),result.count(1),result.count(2))
    return result

def train(train_file, test_file, out_file, model_dump_dir):
    out_dir = os.path.split(out_file)[0]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    sent_label = pd.read_csv(train_file, encoding='UTF8')
    sent_label_1 = sent_label[sent_label['type']==1]
    sent_label_0 = sent_label[sent_label['type']==0]

    n=3
    index_1 = random.sample(list(range(len(sent_label_1))), len(sent_label_1))
    index_0 = random.sample(list(range(len(sent_label_0))), len(index_1)*n)

    train = pd.DataFrame(columns=['id','sentence','type'])

    j=0
    k=0
    for i in range(len(index_1)):
        if j < len(index_1):
            idx_1 = index_1[j]
            j = j+1
            #train.loc[len(train)] = {'id':sent_label.iloc[idx_1]['id'], 'sentence':sent_label.loc[idx_1]['sentence'], 'type':sent_label.loc[idx_1]['type']}
            train.loc[len(train)] = sent_label_1.iloc[idx_1]
        for m in range(n):
            if k < len(index_0):
                idx_0 = index_0[k]
                k = k+1
                train.loc[len(train)] = sent_label_0.iloc[idx_0]

    train['sentence'] = train['sentence'].apply(lambda text: text.replace(' ', ''))
    train['sentence'] = train['sentence'].apply(lambda text: sent_after_split_word(text, True, os.path.abspath('stopwords/stopwords.txt')))


    test = pd.read_csv(test_file, encoding='UTF8')
    test['text']=test['text'].apply(lambda text: '' if isinstance(text, float) else text)
    test['text']=test['text'].apply(lambda text: text.replace(' ', ''))
    test['text'] = test['text'].apply(lambda text: sent_after_split_word(text, True, os.path.abspath('stopwords/stopwords.txt')))


    if sys.version_info[0]==3:
        train.to_csv(os.path.join(out_dir, 'short_text_classify_train.csv'))
        test.to_csv(os.path.join(out_dir, 'short_text_classify_test.csv'))

    train_y = to_categorical(train['type'])

    vec = TfidfVectorizer(
                max_df=0.95, min_df=2,
                sublinear_tf=True,
                strip_accents='unicode',
                analyzer='char',
                ngram_range=(2,4),
                max_features=30000)

    all_text = pd.concat([train['sentence'],test['text']])
    vec.fit(all_text)

    joblib.dump(vec, os.path.join(model_dump_dir,'TfidfVector'))
    
    trn_term_doc = vec.transform(train['sentence'])
    test_term_doc = vec.transform(test['text'])

    #print(vec.get_feature_names())
    print(type(trn_term_doc))
    print(trn_term_doc.shape)

    lgbm_prds = model_lgb(trn_term_doc,test_term_doc,train_y, os.path.join(model_dump_dir, "lgb.model"))
    
    svm_preds = model_nbsvm(trn_term_doc,test_term_doc,train_y, os.path.join(model_dump_dir, "nbsvm.model"))

    lr_preds = model_lr(trn_term_doc,test_term_doc,train_y, os.path.join(model_dump_dir, "lr.model"))

    lgbm_p = trans_result(lgbm_prds)
    nbsvm_p = trans_result(svm_preds)
    lr_p = trans_result(lr_preds)

    merge = (lgbm_prds + svm_preds + lr_preds)/3
    merge_result = trans_result(merge)

    test['type']=merge_result
    test.to_csv(out_file, encoding='UTF8', index=None)

def predict(test_file, out_file, model_dir):
    out_dir = os.path.split(out_file)[0]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    test = pd.read_csv(test_file, encoding='UTF8')
    test['text']=test['text'].apply(lambda text: '' if isinstance(text, float) else text)
    test['text']=test['text'].apply(lambda text: text.replace(' ', ''))
    test['text'] = test['text'].apply(lambda text: sent_after_split_word(text, True, '../stopwords/stopwords.txt'))

    vec = joblib.load(os.path.join(model_dir, "TfidfVector"))

    test_term_doc = vec.transform(test['text'])

    #lgbm_prds = model_lgb(trn_term_doc,test_term_doc,train_y)
    lgbm_prds = lgb_predict(os.path.join(model_dir, "lgb.model"), test_term_doc)
    
    #svm_preds = model_nbsvm(trn_term_doc,test_term_doc,train_y)
    svm_preds = nbsvm_predict(os.path.join(model_dir, "nbsvm.model"), test_term_doc)

    #lr_preds = model_lr(trn_term_doc,test_term_doc,train_y)
    lr_preds = lr_predict(os.path.join(model_dir, "lr.model"), test_term_doc)

    lgbm_p = trans_result(lgbm_prds)
    nbsvm_p = trans_result(svm_preds)
    lr_p = trans_result(lr_preds)

    merge = (lgbm_prds + svm_preds + lr_preds)/3
    merge_result = trans_result(merge)

    print(merge_result.count(0), merge_result.count(1))

    test['type']=merge_result
    test.to_csv(out_file, encoding='UTF8', index=None)

def preprocess_test(test):

    test['text']=test['text'].apply(lambda text: '' if isinstance(text, float) else text)
    test['text']=test['text'].apply(lambda text: text.replace(' ', ''))
    test_s = test.copy()
    test_s['text'] = test_s['text'].apply(lambda text: sent_after_split_word(text, True, '/home/sohu3/ayfkm/pic_classify/stopwords/stopwords.txt'))
    return test, test_s



def predict_proba(test, model_dir):
    vec = joblib.load(os.path.join(model_dir, "TfidfVector"))

    test_term_doc = vec.transform(test['text'])

    lgbm_prds = lgb_predict(os.path.join(model_dir, "lgb.model"), test_term_doc)

    svm_preds = nbsvm_predict(os.path.join(model_dir, "nbsvm.model"), test_term_doc)

    lr_preds = lr_predict(os.path.join(model_dir, "lr.model"), test_term_doc)

    merge = (lgbm_prds + svm_preds + lr_preds)/3.0
    return merge

def predict_merge(test_file, modeldir_type, out_file):

    test = pd.read_csv(test_file, encoding='UTF8')

    test_char, test_word = preprocess_test(test)

    merge_all = np.zeros((len(test),2))
    for model_dir, char_word in modeldir_type:
        test_data = test_char if char_word=='char' else test_word

        merge_all = merge_all + predict_proba(test_data, model_dir)

    merge_all = merge_all/len(modeldir_type)

    test['type'] = trans_result(merge_all)
    test.to_csv(out_file, encoding='UTF8', index=None)


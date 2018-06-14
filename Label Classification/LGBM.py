#coding:utf8
import numpy as np
import pandas as pd 
import lightgbm as lgb
from lightgbm.sklearn import LGBMClassifier
from gensim.models import KeyedVectors
from gensim.models import word2vec

from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.externals import joblib
import gc

from sklearn.metrics import confusion_matrix


def eval_file(label, pre):
    try:
        num = 0
        den1 = 0
        den2 = 0
        for i in range(len(label)):

            if label[i] in (1, 2):
                den1 += 1
            if pre[i] in (1, 2):
                den2 += 1
            if pre[i] == label[i]:
                if label[i] == 2:
                    num += 1
                elif label[i] == 1:
                    num += 1
        if num == 0:
            recall = precision = f = 0.0
        else:
            recall = float(num) / den1
            precision = float(num) / den2
            f = 2 * recall * precision / (recall + precision)
    except Exception, e:
        return 1, str(e)

    return f, recall, precision

def customedscore(preds, dtrain):   #preds是结果（概率值），dtrain是个带label的DMatrix
    label = dtrain.get_label()
    # pred = [int(i>=0.5) for i in preds]
    f,r,a = eval_file(label,preds)
    return 'FSCORE',float(f)

#source file
train_content = []
train_add_feature = [] 
train_file = open('News_info_train_seg_2.txt','r')
train_file_list = train_file.readlines()
for index, texts in enumerate(train_file_list):
    train_content.append(texts.split('\t')[-1].split(' '))
    train_add_feature.append([int(x) for x in texts.split('\t')[1:5]])

label = []
label_file = open('train/News_pic_label_train.txt','r')
label_file_list = label_file.readlines()
for index, texts in enumerate(label_file_list): 
    h = int(texts.split('\t')[1])
    if h == 1:
        label.append(0)
    elif h == 2:
        label.append(1)
    else:
        label.append(0)


def choosedata(x, ff):
    if x == 50:
        word2vec_dict50=word2vec.Word2Vec.load('word2vec50/word2vec_test_model_50')
        article_vector50 = []
        for index, texts in enumerate(ff):
            temp=np.array([0 for _ in range(50)], dtype= np.float64)
            for i in texts:
                if i.decode('utf8') in word2vec_dict50:
                    temp += word2vec_dict50[i.decode('utf8')]
            article_vector50.append(temp)
        return article_vector50
    if x == 100:      
        word2vec_dict100=word2vec.Word2Vec.load('word2vec100/word2vec_test_model_100')
        article_vector100 = []
        for index, texts in enumerate(ff):
            temp=np.array([0 for _ in range(100)], dtype= np.float64)
            for i in texts:
                if i.decode('utf8') in word2vec_dict100:
                    temp += word2vec_dict100[i.decode('utf8')]
            article_vector100.append(temp)
        return article_vector100
    if x == 150:
        word2vec_dict150=word2vec.Word2Vec.load('word2vec150/word2vec_test_model_150')
        article_vector150 = []
        for index, texts in enumerate(ff):
            temp=np.array([0 for _ in range(150)], dtype= np.float64)
            for i in texts:
                if i.decode('utf8') in word2vec_dict150:
                    temp += word2vec_dict150[i.decode('utf8')]
            article_vector150.append(temp)
        return article_vector150
    if x == 200:
        word2vec_dict200=word2vec.Word2Vec.load('word2vec200/word2vec_test_model')
        article_vector200 = []
        for index, texts in enumerate(ff):
            temp=np.array([0 for _ in range(200)], dtype= np.float64)
            for i in texts:
                if i.decode('utf8') in word2vec_dict200:
                    temp += word2vec_dict200[i.decode('utf8')]
            article_vector200.append(temp)
        return article_vector200
    if x == 300:
        word2vec_dict300 = KeyedVectors.load_word2vec_format('word2vec300/word2vec_test_model_300_final.bin', binary=True)
        article_vector300 = []
        for index, texts in enumerate(ff):
            temp=np.array([0 for _ in range(300)], dtype= np.float64)
            for i in texts:
                if i.decode('utf8') in word2vec_dict300:
                    temp += word2vec_dict300[i.decode('utf8')]
            article_vector300.append(temp)
        return article_vector300
    if x == 256:
        word2vec_dict256=word2vec.Word2Vec.load('word2vec256_2/word2vec_test_model_256_2')
        article_vector256 = []
        for index, texts in enumerate(ff):
            temp=np.array([0 for _ in range(256)], dtype= np.float64)
            for i in texts:
                if i.decode('utf8') in word2vec_dict256:
                    temp += word2vec_dict256[i.decode('utf8')]
            article_vector256.append(temp)
        return article_vector256
    if x == 250:
        word2vec_dict200=word2vec.Word2Vec.load('word2vec200/word2vec_test_model')
        article_vector200 = []
        for index, texts in enumerate(ff):
            temp=np.array([0 for _ in range(200)], dtype= np.float64)
            for i in texts:
                if i.decode('utf8') in word2vec_dict200:
                    temp += word2vec_dict200[i.decode('utf8')]
            article_vector200.append(temp)
        word2vec_dict50=word2vec.Word2Vec.load('word2vec50/word2vec_test_model_50')
        article_vector50 = []
        for index, texts in enumerate(ff):
            temp=np.array([0 for _ in range(50)], dtype= np.float64)
            for i in texts:
                if i.decode('utf8') in word2vec_dict50:
                    temp += word2vec_dict50[i.decode('utf8')]
            article_vector50.append(temp)
        return np.hstack((article_vector200,article_vector50)) 

###分类器###
def modelfit(alg, dtrain,label,useTrainCV=True, cv_folds=5, early_stopping_rounds=10):
    if useTrainCV:
        lgb_param = alg.get_params()
        lgtrain =lgb.Dataset(dtrain, label=label)
        cvresult = lgb.cv(lgb_param, lgtrain, num_boost_round=1000, nfold=cv_folds,
           early_stopping_rounds=early_stopping_rounds,verbose_eval=1,shuffle=True)

        alg.set_params(n_estimators=cvresult['binary_logloss-mean'].index(np.min(cvresult['binary_logloss-mean']))
)
        print 'the best n_estimators is :',cvresult['binary_logloss-mean'].index(np.min(cvresult['binary_logloss-mean']))

    
    alg.fit(dtrain, label, eval_metric='merror')
    # del dtrain; gc.collect()
    joblib.dump(alg, 'train_lmgb_300_add_l1_l2_1to0_binary.model')
    # lr = joblib.load('lr.model')       
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain)
    # dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    #Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(label, dtrain_predictions)
    # print metrics.confusion_matrix(label, dtrain_predictions)
    f,R,P  = eval_file(label, dtrain_predictions)
    print 'recall : ',R
    print 'precision : ',P
    print 'f-meature : ',f


lgb1 = LGBMClassifier(
n_estimators=1000,
boosting_type= 'gbdt',
# objective='multiclass',
# num_class=3,
objective = 'binary',
learning_rate= 0.05,
max_bin = 500,
num_leaves= 31,
n_jobs=4,
is_unbalance='true',
subsample = 0.8,
colsample_bytree=0.8,
lambda_l1 = 1,
lambda_l2 = 1
)
print 'Read data done!'
X1 = np.array(np.hstack((choosedata(300,train_content),train_add_feature)))
print np.shape(X1)
y = label
print len(y)

modelfit(lgb1, X1, y)

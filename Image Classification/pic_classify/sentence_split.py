#encoding:utf-8
import re
import pandas as pd
import numpy as np
import os
import jieba
import sys
from csv2txt import news_train_txt2csv
from csv2txt import news_pic_lable_train_txt2csv 
from multiprocessing import Pool
import time

#punctuation = "[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+"
#punctuation = "[\s+\.\!\/_,$%^*(+\"\']+|[s——！，。？、~@#￥%……&*（）]+"
#re.sub(ur"[%s]+" %punctuation, "", text) 
def load_stopwords(filename):
    stopwords_list = []
    words_list = []
    with open(filename, 'r') as f:
        words_list = f.readlines()

    for words in words_list:
        if sys.version_info[0]==2:
            stopwords_list.append(words.replace('\n', '').decode('utf-8')) # for python2
        elif sys.version_info[0]==3:
            stopwords_list.append(words.replace('\n', ''))  # for python3
    
    return stopwords_list

def len_sent(text, punctuation=u'.*?[！|,|，|。|...|？|?|!|；|~|～|。|：：]+'):
    l=[]
    #for s in re.findall(punctuation, text+u'：：'):
    #for s in re.findall(u'.*?[！|。|...|？|?|!|；|~|～|。|：：]+',text+u'：：'):
    for s in re.findall(punctuation, text):
        l.append(len(s))
    
    return l


def split_sent(text, punctuation=u'.*?[！|,|，|。|...|？|?|!|；|~|～|。|：：]+'):
    start=time.time()
    sents=[]
    #for s in re.findall(punctuation,text+u'：：'):
    #for s in re.findall(u'.*?[！|。|...|？|?|!|；|~|～|。|：：]+',text+u'：：'):
    for s in re.findall(punctuation,text):
        sents.append(s)
    
    if not sents:
        sents.append(text)
    end=time.time()
    print("split_sent use time:%d"%(end-start))
    return sents

def split_word(sent):
    words=[]
    words = jieba.cut(sent, cut_all=False)
    return words

def sent_after_split_word(sent, sub_punctuation=False, stopwords_file=None):
    """给句子加空格——即分词后的句子"""
    if sub_punctuation:
        #sent=re.sub(r"[\s+\.\!\/_,%^*(+\"\']+|[s——！，。？、~@#%……&*（）]+", "", sent)  # Not sub '￥' and '$' 
        sent=re.sub(r"[\s+\.\!\/_,%^*(+\"\“\”\']+|[s——！,，。?？:、~@#%……&*（）]+", "", sent)  # Not sub '￥' and '$' 
    
    words = split_word(sent)

    if stopwords_file:
        words_cut_stopwords = []
        stopwords = load_stopwords(stopwords_file)
        for w in words:
            if w not in stopwords:
                words_cut_stopwords.append(w)
    
        words_sent = ' '.join(word for word in words_cut_stopwords)
    else:
        words_sent = ' '.join(word for word in words)
    return words_sent

def split_text(text, sub_punctuation=False, stopwords_file=None):
    """Split text into words"""
    all_words = ''
    sents = split_sent(text)
    for sent in sents:
        s = sent_after_split_word(sent, sub_punctuation=sub_punctuation, stopwords_file=stopwords_file)
        all_words = all_words + s + ' '
    
    return all_words


def split_text_get_last_sent(text, sub_punctuation=False, stopwords_file=None):
    """Split text into words"""
    all_words = ''
    sents = split_sent(text)[-1]
    for sent in sents:
        s = sent_after_split_word(sent, sub_punctuation=sub_punctuation, stopwords_file=stopwords_file)
        all_words = all_words + s + ' '
    
    return all_words

def cut_stopwords(words, stopwords):
    ret_words=''
    for w in words:
        if w not in stopwords:
            ret_words=ret_words+w

    return ret_words
   
def cut_punctuation(words):
    return re.sub(r"[\s+\.\!\/_,%^*(+\"\“\”\']+|[s——！,，。?？:、~@#%……&*（）]+", "", words) 


def get_sent_label(trainX='News_info_train_filted.csv', label='News_pic_label_train.csv', output='sent_label.csv'):
    """以句子为单位，获取 （句子， 是否营销）
        句子来自训练集合中所有标签为0的文本（非营销）和训练集标注文件中标注出来的文本（营销）
    """
    train = pd.read_csv(trainX)
    label = pd.read_csv(label)
    if len(train) != len(label):
        print('Length of text and label are not equal!')
        return

    train_sent = pd.DataFrame(columns=['id', 'sentence', 'type'])   #id is not unique
    for i in range(len(label)):
        start = time.time()
        if train.iloc[i]['id'] != label.iloc[i]['id']:
            print('Id of text and label are not equal!')
            return

        kind = label.iloc[i]['type']
        if  kind == 0:
            if sys.version_info[0]==3:
                sents = split_sent(train.iloc[i]['text'], punctuation=u'.*?[！|。|...|？|?|!|；|~|～|。|：：]+')
            else:       #python2
                sents = split_sent(train.iloc[i]['text'].decode('utf-8'), punctuation=u'.*?[！|。|...|？|?|!|；|~|～|。|：：]+')
                
            for s in sents:
                train_sent.loc[len(train_sent)] = {'id':train.iloc[i]['id'], 'sentence':s, 'type':0}
        elif kind == 1:
            if isinstance(label.iloc[i]['text'], float):
                continue
            segs = label.iloc[i]['text'].split('\t')
            for seg in segs:
                if sys.version_info[0]==3:
                    sents = split_sent(seg, punctuation=u'.*?[！|。|...|？|?|!|；|~|～|。|：：]+')
                else:   #python2
                    sents = split_sent(seg.decode('utf-8'), punctuation=u'.*?[！|。|...|？|?|!|；|~|～|。|：：]+')
                    
                for s in sents:
                    train_sent.loc[len(train_sent)] = {'id':train.iloc[i]['id'], 'sentence':s, 'type':1}
        end=time.time()
        print("get_sent_lable loop:%d use time:%d"%(i, end-start))
    if output:
        train_sent.to_csv(output, encoding='UTF8')
    return train_sent


def get_sent_label2(train, label, output='sent_label.csv'):
    """以句子为单位，获取 （句子， 是否营销）
        句子来自训练集合中所有标签为0的文本（非营销）和训练集标注文件中标注出来的文本（营销）
    """
    if len(train) != len(label):
        print('Length of text and label are not equal!')
        return

    train_sent = pd.DataFrame(columns=['id', 'sentence', 'type'])   #id is not unique
    for i in range(len(label)):
        start = time.time()
        if train.iloc[i]['id'] != label.iloc[i]['id']:
            print('Id of text and label are not equal!')
            return

        kind = label.iloc[i]['type']
        if  kind == 0:
            if sys.version_info[0]==3:
                sents = split_sent(train.iloc[i]['text'], punctuation=u'.*?[！|。|...|？|?|!|；|~|～|。|：：]+')
            else:       #python2
                sents = split_sent(train.iloc[i]['text'].decode('utf-8'), punctuation=u'.*?[！|。|...|？|?|!|；|~|～|。|：：]+')
                
            for s in sents:
                train_sent.loc[len(train_sent)] = {'id':train.iloc[i]['id'], 'sentence':s, 'type':0}
        elif kind == 1:
            if isinstance(label.iloc[i]['text'], float):
                continue
            segs = label.iloc[i]['text'].split('\t')
            for seg in segs:
                if sys.version_info[0]==3:
                    sents = split_sent(seg, punctuation=u'.*?[！|。|...|？|?|!|；|~|～|。|：：]+')
                else:   #python2
                    sents = split_sent(seg.decode('utf-8'), punctuation=u'.*?[！|。|...|？|?|!|；|~|～|。|：：]+')
                    
                for s in sents:
                    train_sent.loc[len(train_sent)] = {'id':train.iloc[i]['id'], 'sentence':s, 'type':1}
        end=time.time()
        print("get_sent_lable loop:%d use time:%d"%(i, end-start))

    train_sent.to_csv(output, encoding='UTF8', index=None)

def quick_get_sent_label(train_file, label_file, out_file, nproc=3):
    train = pd.read_csv(train_file)
    label = pd.read_csv(label_file)
    
    size = len(train)
    if len(label) != size:
        print('Length of text and label are not equal!')
        return 
    out_dir, f  = os.path.split(out_file)
    pre, ext = f.split('.')
    
    batch_size = size//nproc
    p = Pool(processes=nproc+1)
    for batch in range(0, size, batch_size):
        f_ = os.path.join(out_dir, "%s_%d.%s"%(pre, batch//batch_size, ext))
        p.apply_async(get_sent_label2, args=(train.iloc[batch:batch+batch_size], 
            label.iloc[batch:batch+batch_size], f_))
    p.close()
    p.join()

    train_sent = pd.DataFrame(columns=['id', 'sentence', 'type'])   #id is not unique
    for batch in range(0, size, batch_size):
        f_ = os.path.join(out_dir, "%s_%d.%s"%(pre, batch//batch_size, ext))
        one = pd.read_csv(f_)
        train_sent = pd.concat([train_sent, one])
        
    train_sent.to_csv(out_file, encoding='UTF8', index=None)
        

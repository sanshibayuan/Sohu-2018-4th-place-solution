#news id   news type  id of pictures with sale   news text with sale
#encoding='utf-8'
import sys
import pandas as pd

def result_txt2csv(txt_file='submission.txt', csv_file=None):
    data = pd.DataFrame(columns=['id','type','pic','text'])
    item = pd.DataFrame(columns=['id','type','pic','text'])
    if sys.version_info[0]==3:
        with open(txt_file, "r",encoding='utf-8') as txt:
            for line in txt:
                ls = line.split('\t', 3)
                item.loc[0] = [s for s in ls]
                data = pd.concat([data,item])
    else:
        with open(txt_file, "r") as txt:
            for line in txt:
                ls = line.split('\t', 3)
                item.loc[0] = [s for s in ls]
                data = pd.concat([data,item])
    
    if csv_file:
        data.to_csv(csv_file,index=True)
    return data

def reuslt_csv2txt(csv_file='submission.csv', txt_file=None):
    data = pd.read_csv(csv_file)
    with open(txt_file, "w") as txt:
        for row in data.itertuples(index=True, name='Pandas'):
            str = row['id']+ '\t'+ row['type'] + '\t' + row['pic'] + '\t' + row['text']
            txt.writelines(str)


def news_train_txt2csv(txt_file='News_info_train.txt', csv_file=None):
    data = pd.DataFrame(columns=['id', 'text', 'pic'])
    item = pd.DataFrame(columns=['id', 'text', 'pic'])
    if sys.version_info[0]==3:
        with open(txt_file, "r", encoding='utf-8') as txt:
            for line in txt:
                ls = line.split('\t', 2)
                item.loc[0] = [s.rstrip('\n') for s in ls]
                data = pd.concat([data, item])
    else:
        with open(txt_file, "r") as txt:
            for line in txt:
                ls = line.split('\t', 2)
                item.loc[0] = [s.rstrip('\n') for s in ls]
                data = pd.concat([data, item])
    
    if csv_file:
        data.to_csv(csv_file, index=True)
    return data

def news_train_csv2txt(csv_file='News_info_train.csv', txt_file=None):
    data = pd.read_csv(csv_file)
    with open(txt_file, "w") as txt:
        for row in data.itertuples(index=True, name='Pandas'):
            str = row['id']+ '\t'+ row['text'] + '\t' + row['pic']
            txt.writelines(str)

def news_pic_lable_train_txt2csv(txt_file='News_pic_label_train.txt', csv_file=None):
    data = pd.DataFrame(columns=['id','type','pic','text'])
    item = pd.DataFrame(columns=['id','type','pic','text'])
    if sys.version_info[0]==3:
        with open(txt_file, "r",encoding='utf-8') as txt:
            for line in txt:
                ls = line.split('\t', 3)
                item.loc[0] = [s.rstrip('\n') for s in ls]
                data = pd.concat([data,item])
    else:
        with open(txt_file, "r") as txt:
            for line in txt:
                ls = line.split('\t', 3)
                item.loc[0] = [s.rstrip('\n') for s in ls]
                data = pd.concat([data,item])

    if csv_file:
        data.to_csv(csv_file,index=True)
    return data

def news_pic_lable_train_csv2txt(csv_file='News_pic_label_train.csv', txt_file=None):
    data = pd.read_csv(csv_file)
    with open(txt_file, "w") as txt:
        for row in data.itertuples(index=True, name='Pandas'):
            str = row['id']+ '\t'+ row['type'] + '\t' + row['pic'] + '\t' + row['text']
            txt.writelines(str)

def news_unlabel_txt2csv(txt_file='News_info_unlabel.txt', csv_file=None):
    return news_train_txt2csv(txt_file, csv_file)

def new_unlabel_csv2txt(csv_file='News_info_unlabel.txt', txt_file=None):
    return news_train_csv2txt(csv_file,txt_file)

def news_validate_txt2csv(txt_file='News_info_validate.txt', csv_file=None):
    return news_train_txt2csv(txt_file, csv_file)

def news_validate_csv2txt(csv_file='News_pic_validate.csv', txt_file=None):
    return news_train_csv2txt(csv_file, txt_file)


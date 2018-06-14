import joblib
import pandas as pd
import sys
import os
from pic_text_classify import predict
from pic_text_classify import train
from pic_text_classify import predict_merge
from img2text import quick_ocr_pics
from img2text import quick_get_pics_area
from img2text import get_pics_area
from img2text import get_news1_id
from img2text import get_pics_from_id
from img2text import get_pic_result, handle_little_area,combine
import shutil


if __name__ == '__main__':
    #input:
    news_file = sys.argv[1]
    img_dir = sys.argv[2]
    text_result_file = sys.argv[3]

    #output:
    tmpfold = sys.argv[4]
    predict_file = sys.argv[5]
    pic_text_file = os.path.join(tmpfold, "pic_text.csv")
    pic_text_result = os.path.join(tmpfold, "pic_text_result.csv")
    pic_result = os.path.join(tmpfold, "pic_result.csv")
    pic_result_ = os.path.join(tmpfold, "pic_result_.csv")
    
    pics_file = os.path.join(tmpfold, "pics_file")
    area_file = os.path.join(tmpfold, "pic_area.csv")

    #args:
    modeldir_type = [('/data1/sohu3/pic_model/model1', 'char'),
                     ('/data1/sohu3/pic_model/model2', 'word'),
                     ('/data1/sohu3/pic_model/model4', 'word')
                     ]

    model_path = '/data1/sohu3/pic_model'
    min_area = 0.065

    #-------------ocr------------------------------------
    print("ocr:")
    news_id = get_news1_id(text_result_file)
    pics = get_pics_from_id(news_id, news_file) 
    pics = list(set(pics))  # clean duplicated pictures

    pics_path = []
    for pic in pics:
        if pic.find('GIF') !=-1 or pic.find('gif') !=-1:
            continue
        else:
            pics_path.append(os.path.join(os.path.abspath(img_dir), pic))

    if len(pics_path)>0:
        quick_ocr_pics(pics_path, '', pic_text_file )
    else:
        print("log: length of  pics_path: 0")
        shutil.copy(text_result_file, predict_file)
	sys.exit(0)
    #----------------------------------------------------
  
    #------------pic text classify-----------------------
    print("predict:")
    #train(sent_label_file, pic_text_file, pic_text_result, model_path)
    #predict(pic_text_file, pic_text_result, model_path)
    predict_merge(pic_text_file, modeldir_type, pic_text_result)

    print("get_pic_result:")
    get_pic_result(news_file, text_result_file, pic_text_result, pic_result) 
    #----------------------------------------------------

    #-----------cut little text area pictures------------
    print("get_pics_area:")
    pr = pd.read_csv(pic_result)
    pics_ = list(pr[pr['type']==1]['pic'])
    pics = []
    for pic in pics_:
        pics.append(os.path.join(os.path.abspath(img_dir), pic))

    print("log: type 1 pics: %d"%(len(pics)))
    quick_get_pics_area(pics, area_file, nproc=5)

    print("handle_little_area:")
    handle_little_area(min_area, pic_result, area_file, pic_result_)
    #---------------------------------------------------

    #-----------------get predict file------------------
    print('combine:')
    combine(text_result_file, pic_result_, predict_file)
    #---------------------------------------------------


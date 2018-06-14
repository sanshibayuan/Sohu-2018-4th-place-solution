import cv2
import pandas as pd
import pytesseract
import os
import glob
from multiprocessing import Pool
import joblib

pytesseract.pytesseract.tesseract_cmd='/usr/local/bin/tesseract'
tessdata_dir_config='/data1/sohu3/pic_model/tessdata'

#pytesseract.pytesseract.tesseract_cmd='/usr/bin/tesseract'
#tessdata_dir_config='/usr/share/tesseract-ocr/tessdata/'


def get_rects(path):
    rects=[]
    with open(path, 'r') as f:
        for line in f:
            line = line.rstrip('\r\n')
            if len(line) == 0:
                continue
            rect = line.split(',')
            rect = [int(v) for v in rect]
            rects.append(rect)
    return rects

def ocr(pic, rects_txt):
    return tesseract_ocr(pic, 'opencv', True, '--tessdata-dir "/data1/sohu3/pic_model/tessdata"')

def tesseract_ocr(pic, method='opencv', gray=True, configs='--tessdata-dir "/data1/sohu3/pic_model/tessdata"'):
    if method=='opencv':
        img = cv2.imread(pic)
        if gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif method == 'Image':
        img = Image.open(pic)
        if gray:
            img = img.convert("L")
    text = pytesseract.image_to_string(img, lang='chi_sim', config=configs)
    texts = text.split('\n')
    #text = []
    text = ''
    for t in texts:
        t = t.replace(' ', '')
        if t != '':
            #text.append(t)
            text = text + t
    return text

def ocr_img(img_dir, rects_dir, dst):

    pics = glob.glob(os.path.join(img_dir, '*.png')) \
               + glob.glob(os.path.join(img_dir, '*.PNG')) \
               + glob.glob(os.path.join(img_dir, '*.jpg')) \
               + glob.glob(os.path.join(img_dir, '*.JPG')) \
               + glob.glob(os.path.join(img_dir, '*.jpeg')) \
               + glob.glob(os.path.join(img_dir, '*.JPEG')) \
               + glob.glob(os.path.join(img_dir, '*.bmp')) \
               + glob.glob(os.path.join(img_dir, '*.BMP'))
    pic_text = pd.DataFrame(columns=['pic','text'])
    for pic in pics:
        name = os.path.split(pic)[1]
        rect_txt = 'res_{}.txt'.format(name.split('.')[0])
        try:
            t = ocr(pic, os.path.join(rects_dir, rect_txt) )
            pic_text.loc[len(pic_text)] = {'pic':name, 'text':t}
        except Exception as e:
            #print('{} exception:{}'.format(pic, e))
            pass

    if dst is not None:
        pic_text.to_csv(dst, encodign='UTF8') 
    return pic_text

def ocr2(pics, rects_dir, dst):
    pic_text = pd.DataFrame(columns=['pic','text'])
    for pic in pics:
        name = os.path.split(pic)[1]
        rect_txt = 'res_{}.txt'.format(name.split('.')[0])
        try:
            t = ocr(pic, os.path.join(rects_dir, rect_txt) )
            pic_text.loc[len(pic_text)] = {'pic':name, 'text':t}
        except Exception as e:
            #print('{} exception:{}'.format(pic, e))
            pass

    if dst is not None:
        pic_text.to_csv(dst, encoding='UTF8') 
    return pic_text
     
def quick_ocr_img(img_dir, rects_dir, dst, nproc=5):
    pics = glob.glob(os.path.join(img_dir, '*.png')) \
               + glob.glob(os.path.join(img_dir, '*.PNG')) \
               + glob.glob(os.path.join(img_dir, '*.jpg')) \
               + glob.glob(os.path.join(img_dir, '*.JPG')) \
               + glob.glob(os.path.join(img_dir, '*.jpeg')) \
               + glob.glob(os.path.join(img_dir, '*.JPEG')) \
               + glob.glob(os.path.join(img_dir, '*.bmp')) \
               + glob.glob(os.path.join(img_dir, '*.BMP'))
    size = len(pics)//nproc
    pic_text = pd.DataFrame(columns=['pic','text'])
    p = Pool(processes=nproc+1)
    dst_file = os.path.split(dst)[1].split('.')[0]
    dst_dir = os.path.split(dst)[0]

    for batch in range(0, len(pics), size):
        batch_dst = os.path.join(dst_dir, "%s_%d.csv"%(dst_file, batch/size))
        p.apply_async(ocr2, args=(pics[batch:batch+size], rects_dir, batch_dst))

    p.close()
    p.join()

    for batch in range(0, len(pics), size):
        batch_one = pd.read_csv(os.path.join(dst_dir, "%s_%d.csv"%(dst_file, batch/size)))
        pic_text = pd.concat([pic_text, batch_one]) 
    
    pic_text.to_csv(dst, encoding='UTF8') 
    return pic_text

def quick_ocr_pics(pics, rects_dir, dst, nproc=5):
    if len(pics)<=nproc:
        size = len(pics)
    else:
        size = len(pics)//nproc
    pic_text = pd.DataFrame(columns=['pic','text'])
    p = Pool(processes=nproc+1)
    dst_file = os.path.split(dst)[1].split('.')[0]
    dst_dir = os.path.split(dst)[0]

    for batch in range(0, len(pics), size):
        batch_dst = os.path.join(dst_dir, "%s_%d.csv"%(dst_file, batch/size))
        p.apply_async(ocr2, args=(pics[batch:batch+size], rects_dir, batch_dst))

    p.close()
    p.join()

    for batch in range(0, len(pics), size):
        batch_one = pd.read_csv(os.path.join(dst_dir, "%s_%d.csv"%(dst_file, batch/size)))
        pic_text = pd.concat([pic_text, batch_one]) 
    
    pic_text.to_csv(dst, encoding='UTF8') 
    return pic_text

def get_news1_id(predict_file):
    pred = pd.read_csv(predict_file, header=None, sep='\t')
    news1_ids = []
    for i in range(len(pred)):
        if pred[1][i]==1:
            news1_ids.append(pred[0][i])
    return news1_ids

def get_pics_from_id(news_id, news_file):
    news = pd.read_csv(news_file, header=None, sep='\t')
    news_dict = dict(zip(news[0],news[2]))
    pics = []
    for k in news_id:
        v = news_dict.get(k,'NULL')
        if isinstance(v, float) or v=='NULL':
            continue
        else:
            pics = pics + v.split(';')
    return pics

def news_train_txt2csv(txt_file, csv_file=None):
    data = pd.DataFrame(columns=['id', 'text', 'pic'])
    item = pd.DataFrame(columns=['id', 'text', 'pic'])
    with open(txt_file, "r") as txt:
        for line in txt:
            ls = line.split('\t', 2)
            item.loc[0] = [s.rstrip('\n') for s in ls]
            data = pd.concat([data, item])
    if csv_file:
        data.to_csv(csv_file, index=True, encoding='UTF8')
    return data

def get_pic_result(news_file, text_result_file, pic_type,  output_file=None):
    news = news_train_txt2csv(news_file)
    name_type = pd.read_csv(pic_type)
    text_result = pd.read_csv(text_result_file, header=None, sep='\t')
    ntd = dict(zip(list(name_type['pic']), list(name_type['type']))) 
    id_pics = dict(zip(list(news['id']), list(news['pic'])))
    news1_id = list(text_result[text_result[1]==1][0])
    
    pic_result = pd.DataFrame(columns=['id', 'pic', 'type'])
    for id1 in news1_id:
        pics = id_pics.get(id1, 'NULL')
        pics = pics.replace('\r', '') 
        pics = pics.replace('\n', '')
        if pics != 'NULL':
            for pic in pics.split(';'):
                t = ntd.get(pic, 0) 
                pic_result.loc[len(pic_result)] = {'id':id1, 'pic':pic, 'type':t}
    
    if output_file is not None:
        pic_result.to_csv(output_file, encoding='UTF8', index=None)
    return pic_result

def get_pics_area(pics_file, area_file):
    
    cwd = os.getcwd()
    os.chdir('/home/sohu3/ayfkm/text-detection-ctpn/')
    os.system('python ctpn/detect.py detect_area %s %s'%(os.path.join(cwd, pics_file), os.path.join(cwd,area_file)))
    os.chdir(cwd)

def quick_get_pics_area(pics, area_file, nproc=1):
    af_dir, af = os.path.split(area_file)
    apre, aext = af.split('.')
   
    if len(pics)<=nproc:
        size = len(pics)
    else: 
        size = len(pics)//nproc
    p = Pool(processes=nproc+1)
    for batch in range(0, len(pics), size):
        pf_ = os.path.join(af_dir, "%s_%d.%s"%("quick_get_pics_area", batch//size, "list"))
        joblib.dump(pics[batch:batch+size], pf_ )
        af_ = os.path.join(af_dir, "%s_%d.%s"%(apre, batch//size, aext))
        p.apply_async(get_pics_area, args=(pf_, af_))
    p.close()
    p.join()

    pic_area = pd.DataFrame(columns=['pic', 'area'])
    for batch in range(0, len(pics), size):
        af_ = os.path.join(af_dir, "%s_%d.%s"%(apre, batch//size, aext))
        one = pd.read_csv(af_)
        pic_area = pd.concat([pic_area, one])
    pic_area.to_csv(area_file, encoding='UTF8', index=None)
     

def handle_little_area(min_area, pic_result_file, pic_area_file, new_result_file):
    result = pd.read_csv(pic_result_file)
    area = pd.read_csv(pic_area_file)
    area_dict = dict(zip(list(area['pic']),list(area['area'])))
    ids = []
    pics = []
    types = []
    cnt=0
    for i in range(len(result)):
        news_id = result.iloc[i]['id']
        pic = result.iloc[i]['pic']
        area = area_dict.get(pic, 0)
        ids.append(news_id)
        pics.append(pic)
        if  area < min_area:
            if area != 0:
                cnt = cnt + 1
            types.append(0)
        else:
            types.append(1)
    print('log: 1->0 nums:%d'%(cnt))
    new_result = pd.DataFrame.from_dict({'id':ids, 'pic':pics, 'type':types}) 
    new_result.to_csv(new_result_file)

def combine(text_result_file, pic_result_file, out_file):
    text_result = pd.read_csv(text_result_file, header=None, sep='\t')
    if list(text_result.columns) != ['id', 'type', 'pic', 'text']:
        tmp = pd.DataFrame(columns=['id', 'type', 'pic', 'text'])
        for i in range(len(text_result)):
            pic = text_result.iloc[i][2]
            pic = pic if pd.notnull(pic) else 'NULL'
            text = text_result.iloc[i][3]
            text = text if pd.notnull(text) else 'NULL'
            tmp.loc[len(tmp)] = {'id':text_result.iloc[i][0], 'type':text_result.iloc[i][1], 'pic':pic, 'text':text}
        text_result = tmp

    pic_result = pd.read_csv(pic_result_file)
    result = pd.DataFrame(columns=['id','type','pic','text'])

    for i in range(len(text_result)):
        kind = text_result.iloc[i]['type']
        id = text_result.iloc[i]['id']
        if kind == 1:
            pic_item = pic_result[(pic_result['id']==id)&(pic_result['type']==1)]
            pics = pic_item['pic']
            #if (len(pic_item) > 0) & (pd.notnull(pics)) :
            if (len(pic_item) > 0) :
                pics_str = ';'.join(pic for pic in pics)
                result.loc[len(result)] = {'id':id, 'type':kind, 'pic':pics_str, 'text':text_result.iloc[i]['text']}
            else:
                result.loc[len(result)] = {'id':id, 'type':kind, 'pic':'NULL', 'text':text_result.iloc[i]['text']}
        else:
            result.loc[len(result)] = {'id':id, 'type':kind, 'pic':text_result.iloc[i]['pic'], 'text':text_result.iloc[i]['text']}
   
    outdir, outfile = os.path.split(out_file)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    result.to_csv(os.path.join(outdir, outfile.split('.')[0]+'.csv'), encoding='UTF8')
    result.to_csv(os.path.join(outdir, outfile),header = None,index = None ,encoding = 'UTF8',na_rep = 'NULL',sep = '\t')



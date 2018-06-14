from __future__ import print_function
import tensorflow as tf
import numpy as np
import os, sys, cv2
import glob
import shutil
sys.path.append(os.getcwd())
from lib.networks.factory import get_network
from lib.fast_rcnn.config import cfg,cfg_from_file
from lib.fast_rcnn.test import test_ctpn
from lib.utils.timer import Timer
from lib.text_connector.detectors import TextDetector
from lib.text_connector.text_connect_cfg import Config as TextLineCfg
import pandas as pd
import joblib



def resize_im(im, scale, max_scale=None):
    f=float(scale)/min(im.shape[0], im.shape[1])
    if max_scale!=None and f*max(im.shape[0], im.shape[1])>max_scale:
        f=float(max_scale)/max(im.shape[0], im.shape[1])
    return cv2.resize(im, None,None, fx=f, fy=f,interpolation=cv2.INTER_LINEAR), f


def draw_boxes(img,image_name,boxes,scale, dst, draw_img=True, show_area=False, area_min=-0.1, area_max=1.1):
    #base_name = image_name.split('/')[-1]
    base_name = os.path.split(image_name)[1]
    with open(os.path.join(dst , 'res_{}.txt'.format(base_name.split('.')[0])), 'w') as f:
    #with open('data/results/' + os.path.split(base_name)[1], 'w') as f:
        area=0
        if show_area:
            draw_img = True
        for box in boxes:
            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                continue
            if draw_img:
                if box[8] >= 0.9:
                    color = (0, 255, 0)
                elif box[8] >= 0.8:
                    color = (255, 0, 0)
    
                cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                cv2.line(img, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), color, 2)
                cv2.line(img, (int(box[6]), int(box[7])), (int(box[2]), int(box[3])), color, 2)
                cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color, 2)
            
            min_x = min(int(box[0]/scale),int(box[2]/scale),int(box[4]/scale),int(box[6]/scale))
            min_y = min(int(box[1]/scale),int(box[3]/scale),int(box[5]/scale),int(box[7]/scale))
            max_x = max(int(box[0]/scale),int(box[2]/scale),int(box[4]/scale),int(box[6]/scale))
            max_y = max(int(box[1]/scale),int(box[3]/scale),int(box[5]/scale),int(box[7]/scale))

            area = area + (max_x-min_x)*(max_y-min_y)
                
            line = ','.join([str(min_x),str(min_y),str(max_x),str(max_y)])+'\r\n'
            f.write(line)
    fx=1.0/scale
    fy=1.0/scale
    area_relative = float(area)/float(img.shape[0]*img.shape[1]*fx*fy)
    if show_area:
        cv2.putText(img, "text area: %d"%(area), (0, img.shape[0]//2-25), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
        cv2.putText(img, "relative area:%.3f"%(area_relative), (0, img.shape[0]//2+25), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
    if draw_img and area_relative > area_min and area_relative < area_max:
        img=cv2.resize(img, None, None, fx=1.0/scale, fy=1.0/scale, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(dst, base_name), img)
    return area_relative
    
def compute_area(img,image_name,boxes,scale, dst, draw_img=True, show_area=False, area_min=-0.1, area_max=1.1):
    #base_name = image_name.split('/')[-1]
    base_name = os.path.split(image_name)[1]
    area=0
    if show_area:
        draw_img = True
    for box in boxes:
        if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
            continue
        if draw_img:
            if box[8] >= 0.9:
                color = (0, 255, 0)
            elif box[8] >= 0.8:
                color = (255, 0, 0)
    
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), color, 2)
            cv2.line(img, (int(box[6]), int(box[7])), (int(box[2]), int(box[3])), color, 2)
            cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color, 2)
        
        min_x = min(int(box[0]/scale),int(box[2]/scale),int(box[4]/scale),int(box[6]/scale))
        min_y = min(int(box[1]/scale),int(box[3]/scale),int(box[5]/scale),int(box[7]/scale))
        max_x = max(int(box[0]/scale),int(box[2]/scale),int(box[4]/scale),int(box[6]/scale))
        max_y = max(int(box[1]/scale),int(box[3]/scale),int(box[5]/scale),int(box[7]/scale))

        area = area + (max_x-min_x)*(max_y-min_y)
            
    fx=1.0/scale
    fy=1.0/scale
    area_relative = float(area)/float(img.shape[0]*img.shape[1]*fx*fy)
    if show_area:
        cv2.putText(img, "text area: %d"%(area), (0, img.shape[0]//2-25), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
        cv2.putText(img, "relative area:%.3f"%(area_relative), (0, img.shape[0]//2+25), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
    if draw_img and area_relative > area_min and area_relative < area_max:
        img=cv2.resize(img, None, None, fx=1.0/scale, fy=1.0/scale, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(dst, base_name), img)
    return area_relative

def ctpn(sess, net, image_name, dst, draw_img=False, show_area=False, area_min=-0.1, area_max=1.1):
    timer = Timer()
    timer.tic()

    img = cv2.imread(image_name)
    img, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
    scores, boxes = test_ctpn(sess, net, img)

    textdetector = TextDetector()
    boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
    ret = draw_boxes(img, image_name, boxes, scale, dst, draw_img=draw_img, show_area=show_area, area_min=area_min, area_max=area_max)
    timer.toc()
    print(('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0]))

    return ret

def ctpn_area(sess, net, image_name, dst, draw_img=False, show_area=False, area_min=-0.1, area_max=1.1):
    #timer = Timer()
    #timer.tic()

    img = cv2.imread(image_name)
    if img is None:
        return 0.0
    img, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
    scores, boxes = test_ctpn(sess, net, img)

    textdetector = TextDetector()
    boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
    ret = compute_area(img, image_name, boxes, scale, dst, draw_img=draw_img, show_area=show_area, area_min=area_min, area_max=area_max)
    #timer.toc()
    #print(('Detection took {:.3f}s for '
    #       '{:d} object proposals').format(timer.total_time, boxes.shape[0]))

    return ret

def detect(src, dst, draw_img=False, show_area=False, area_min=-0.0, area_max=1.1):
    #if os.path.exists("data/results/"):
    #    shutil.rmtree("data/results/")
    if not os.path.exists(dst):
        os.makedirs(dst)
    cfg_from_file('ctpn/text.yml')

    # init session
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    # load network
    net = get_network("VGGnet_test")
    # load model
    print(('Loading network {:s}... '.format("VGGnet_test")), end=' ')
    saver = tf.train.Saver()

    try:
        ckpt = tf.train.get_checkpoint_state(cfg.TEST.checkpoints_path)
        print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('done')
    except:
        raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)

    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in range(2):
        _, _ = test_ctpn(sess, net, im)

    '''
    im_names = glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.png')) + \
               glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.jpg')) + \
               glob.glob(os.path.join(cfg.DATA_DIR, 'demo', '*.jpeg'))
    '''
    im_names = glob.glob(os.path.join(src, '*.png')) \
               + glob.glob(os.path.join(src, '*.PNG')) \
               + glob.glob(os.path.join(src, '*.jpg')) \
               + glob.glob(os.path.join(src, '*.JPG')) \
               + glob.glob(os.path.join(src, '*.jpeg')) \
               + glob.glob(os.path.join(src, '*.JPEG')) \
               + glob.glob(os.path.join(src, '*.bmp')) \
               + glob.glob(os.path.join(src, '*.BMP'))
    print("images:{}".format(len(im_names))) 
    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(('Demo for {:s}'.format(im_name)))
        ctpn(sess, net, im_name, dst, draw_img=draw_img, show_area=show_area, area_min=area_min, area_max=area_max)


def detect_area(pics_file, dst_file, draw_img=False, show_area=False, area_min=-0.0, area_max=1.1):
    cfg_from_file('ctpn/text.yml')

    # init session
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    # load network
    net = get_network("VGGnet_test")
    # load model
    print(('Loading network {:s}... '.format("VGGnet_test")), end=' ')
    saver = tf.train.Saver()

    try:
        ckpt = tf.train.get_checkpoint_state(cfg.TEST.checkpoints_path)
        print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('done')
    except:
        raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)

    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in range(2):
        _, _ = test_ctpn(sess, net, im)

    dst_dir = os.path.split(dst_file)[0]
    pic_area = pd.DataFrame(columns=['pic', 'area']) 
    pics = joblib.load(pics_file)
    print("images:{}".format(len(pics))) 
    for im_name in pics:
        area = ctpn_area(sess, net, im_name, dst_dir)
        pic = os.path.split(im_name)[1]
        pic_area.loc[len(pic_area)] = {'pic':pic, 'area':area}
    pic_area.to_csv(dst_file, index=None, encoding='UTF8')
    

def detect_one(src, dst, draw_img=False, show_area=False, area_min=-0.0, area_max=1.1):
    #if os.path.exists("data/results/"):
    #    shutil.rmtree("data/results/")
    if not os.path.exists(dst):
        os.makedirs(dst)
    cfg_from_file('ctpn/text.yml')

    # init session
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    # load network
    net = get_network("VGGnet_test")
    # load model
    print(('Loading network {:s}... '.format("VGGnet_test")), end=' ')
    saver = tf.train.Saver()

    try:
        ckpt = tf.train.get_checkpoint_state(cfg.TEST.checkpoints_path)
        print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('done')
    except:
        raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)

    im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    for i in range(2):
        _, _ = test_ctpn(sess, net, im)

    ret = ctpn_area(sess, net, src, dst)
    return ret


if __name__ == '__main__':
    if sys.argv[1] == 'detect_one':
        area = detect_one(sys.argv[2], sys.argv[3])
        print("%s relative area:%.3f"%(sys.argv[2], area))
    elif sys.argv[1] == 'detect_area':
        pics_file = sys.argv[2]
        dst_file = sys.argv[3]
        detect_area(pics_file, dst_file)
    else:
        draw_img = False
        show_area = False
        little_dir = None
        area_min=-0.1
        area_max=1.1
        if sys.argv[3]=="draw_img":
            draw_img=True
        if sys.argv[4]=="show_area":
            show_area=True
        if len(sys.argv) >= 6:
            area_min = float(sys.argv[5])
        if len(sys.argv) >= 7:
            area_max = float(sys.argv[6])
        detect(sys.argv[1], sys.argv[2], draw_img, show_area, area_min, area_max)

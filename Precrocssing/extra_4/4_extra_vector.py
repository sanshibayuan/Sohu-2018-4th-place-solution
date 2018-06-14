#coding:utf8
import re
import sys
import getopt

phone = re.compile(r"\(?\d{3,4}[) -]?\d{7,8}")
email = re.compile(r'[^\._][\w\._-]+@(?:[A-Za-z0-9]+\.)+[A-Za-z]+$')
net1 = re.compile(r'[a-zA-z]+://[^\s]*')
net2 = re.compile(r'www.[^\s]{0,15}')
wx = open('/home/sohu3/ayfkm/extra_4/wx.txt','r').readlines()
wx = [i.strip() for i in wx]


# f_ = open('../train/News_info_train_filter.txt','r')
# fout_ = open('4_hand_unlabel_features.txt', 'w')

def extra_4(f_,fout_):
    for index,i in enumerate(f_):
        #文本获取
        t = i.split('\t')[1].strip()

        #找电话号码之类的
        m_ = re.findall(phone, t)
        e_ = re.findall(email, t)
        # q_ = re.findall(qq, t)
        n1_ = re.findall(net1, t)
        n2_ = re.findall(net2, t)
        w_=0
        for j in wx:
            if j in t:
                w_=1
                continue
        #############
        fout_.writelines(i.split('\t')[0].strip())
        fout_.writelines('\t')
        if len(m_)!=0:
            fout_.writelines('1')
        else:
            fout_.writelines('0')
        fout_.writelines('\t')
        if len(e_)!=0:
            fout_.writelines('1')
        else:
            fout_.writelines('0')
        fout_.writelines('\t')
        if len(n1_)!=0 or len(n2_)!=0:
            fout_.writelines('1')
        else:
            fout_.writelines('0')
        fout_.writelines('\t')
        if w_!=0:
            fout_.writelines('1')
        else:
            fout_.writelines('0')
        fout_.writelines('\n')
def usage():

    print '''
    word seg tools

    ./file_seg.py  --source <source_file> --target <target_file> 

    source|-s: input file  
    target|-t: output file 
    Here are one examples .
    seg:
        ./file_seg.py  -s train/News_info_train_filter.txt -t News_info_train_final.txt
    '''

    return
if __name__ == '__main__':
    source_file = ''
    target_file = ''   
    try:
        if len(sys.argv) <= 1:
            usage()
            sys.exit(0)
        options, args = getopt.getopt(sys.argv[1:], "hm:s:t:f:", 
                ["help","source=", "target="])
        for key, value in options:
            if key in ('-h', '--help'):
                usage()
                sys.exit(0)
            if key in ('-s', '--source'):
                source_file = value
            if key in ('-t', '--target'):
                target_file = value
        f = open(source_file,'r')
        f_ = open(target_file,'w')
        extra_4(f,f_)
    except Exception, e:
        print e
        sys.exit(1)


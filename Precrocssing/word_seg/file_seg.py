#coding:utf8
import sys
import jieba
import getopt
# f = open('../train/News_info_train_filter.txt','r')
# f_ = open('News_info_train_final.txt','w')


def make_seg(f,f_):
    jieba.load_userdict("/home/sohu3/ayfkm/word_seg/userdict.txt")
    stopwords = open('/home/sohu3/ayfkm/word_seg/stopwords', 'r').readlines()
    stopwords = [i.strip() for i in stopwords]

    for index, texts in enumerate(f): 
        text = texts.split('\t')[1]
        # print(text)
        words = jieba.cut(text)
        # print words[0]

        f_.writelines(texts.split('\t')[0])
        f_.writelines('\t')

        for w in words:
            if not w.encode('utf8').strip() or w.encode('utf8') in stopwords:
                continue
            else:
                f_.writelines(w.encode('utf8'))
                f_.writelines(' ')
        f_.writelines('\n')

def usage():

    print '''
    word seg tools

    ./file_seg.py  --source <source_file> --target <target_file> 

    source|-s: input file for word seg 
    target|-t: output file for word seg 
    Here are one examples for word seg.
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
        make_seg(f,f_)
    except Exception, e:
        print e
        sys.exit(1)
    # f = open('../train/News_info_train_filter.txt','r')
    # f_ = open('News_78.txt','w')
    # make_seg(f,f_)





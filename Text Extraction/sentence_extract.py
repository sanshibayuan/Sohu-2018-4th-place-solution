#coding:utf8
import sys
import re
import HTMLParser
import jieba
import getopt
import sys


html_parser = HTMLParser.HTMLParser()
space_pat = re.compile(r'\\t|\\n', re.S)
p_pat = re.compile(r'(<p(>| ))|<br>|<br/>', re.S)
sc_tag_pat = re.compile(r'<[^>]+>', re.S)
multi_space_pat = re.compile(r' +', re.S)
count = 0


def str_q2b(s):

    res = ""
    for u in s:

        c = ord(u)
        if c == 12288:
            c = 32
        elif 65281 <= c <= 65374:
            c -= 65248

        res += unichr(c)

    return res

def html_filter(fin, fout_):
    global count
    f_ = open(fin,'r')
    fout = open(fout_, 'w')
    for index,raw_line in enumerate(f_):

        line = raw_line.rstrip('\r\n')
        items = line.split('\t')
        assert len(items) == 3
        s1 = space_pat.sub(' ', items[1]).replace(r'\r', '')
        s2 = p_pat.sub(lambda x: ' ' + x.group(0), s1)
        s3 = sc_tag_pat.sub('<s>', s2).strip()
        s4 = html_parser.unescape(s3.decode('utf8')).encode('utf8')
        s5 = str_q2b(s4.decode('utf8')).encode('utf8').replace('\xc2\xa0', ' ')
        s6 = s5.split('<s>')

        y = ''
        for i in s6:
            count += 1
            if i.strip():
                fout.writelines(items[0])
                fout.writelines('\t')
                fout.writelines(i)

                fout.writelines('\n')

        # return content_txt

def html_filter_file(fin, fout_):

    fout = open(fout_, 'w')
    for raw_line in open(fin):

        line = raw_line.rstrip('\r\n')
        items = line.split('\t')
        assert len(items) == 3



original_file = sys.argv[1] 
sen_file = sys.argv[2] 
html_filter(original_file, sen_file)

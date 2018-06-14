#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import re
import HTMLParser
import getopt


html_parser = HTMLParser.HTMLParser()
space_pat = re.compile(r'\\t|\\n', re.S)
p_pat = re.compile(r'(<p(>| ))|<br>|<br/>', re.S)
sc_tag_pat = re.compile(r'<[^>]+>', re.S)
multi_space_pat = re.compile(r' +', re.S)


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

def html_filter(content):

    s1 = space_pat.sub(' ', content).replace(r'\r', '')
    s2 = p_pat.sub(lambda x: ' ' + x.group(0), s1)
    s3 = sc_tag_pat.sub('', s2).strip()
    s4 = html_parser.unescape(s3.decode('utf8')).encode('utf8')
    s5 = str_q2b(s4.decode('utf8')).encode('utf8').replace('\xc2\xa0', ' ')
    content_txt = multi_space_pat.sub(' ', s5).strip()
    return content_txt

def html_filter_file(fin, fout_):

    fout = open(fout_, 'w')
    for raw_line in open(fin):

        line = raw_line.rstrip('\r\n')
        items = line.split('\t')
        assert len(items) == 3
        items[1] = html_filter(items[1])
        fout.write('%s\n' % '\t'.join(items))

    fout.close()

def jaccard(a1, a2):

    s1 = set(a1)
    s2 = set(a2)
    and_set = s1 & s2
    or_set = s1 | s2
    return 1.0 \
            if len(or_set) == 0 \
            else float(len(and_set)) / len(or_set)

def eval_file(ffilt, ftgt, fsrc):

    try:
        filt_dict = {}
        for raw_line in open(ffilt):

            line = raw_line.rstrip('\r\n')
            items = line.split('\t')
            docid = items[0]
            content = items[1]
            pics = items[2]
            pics = [] if pics == 'NULL' else pics.split(';')
            filt_dict[docid] = (content, pics)

        tgt_dict = {}
        for raw_line in open(ftgt):

            line = raw_line.rstrip('\r\n')
            items = line.split('\t')
            docid = items[0]
            label = items[1]
            pics = items[2]
            segs = items[3:]
            label = int(label)
            pics = [] if pics == 'NULL' else pics.split(';')
            if len(segs) == 1 and segs[0] == 'NULL':
                segs = [] 
            tgt_dict[docid] = (label, pics, segs)

        # check
        src_dict = {}
        for raw_line in open(fsrc):

            line = raw_line.rstrip('\r\n')
            items = line.split('\t')
            docid = items[0]
            label = items[1]
            pics = items[2]
            segs = items[3:]
            label = int(label)
            if label not in (0, 1, 2):
                return 1, 'label out of bound'
            pics = [] if pics == 'NULL' else pics.split(';')
            if len(segs) == 1 and segs[0] == 'NULL':
                segs = [] 
            if docid not in tgt_dict:
                return 1, 'docid not in target'
            _content, _pics = filt_dict[docid]
            for pic in pics:

                if pic not in _pics:
                    return 1, 'not all picture in filtered_file'

            for seg in segs:

                if seg not in _content:
                    return 1, 'not all segment in filtered_file'

            src_dict[docid] = 1

        if len(src_dict) != len(tgt_dict):
            return 1, 'not all target docid in source'

        num = 0
        den1 = 0
        den2 = 0
        for raw_line in open(fsrc):

            line = raw_line.rstrip('\r\n')
            items = line.split('\t')
            docid = items[0]
            label = items[1]
            pics = items[2]
            segs = items[3:]
            label = int(label)
            pics = [] if pics == 'NULL' else pics.split(';')
            if len(segs) == 1 and segs[0] == 'NULL':
                segs = [] 
            _label, _pics, _segs = tgt_dict[docid]
            if _label in (1, 2):
                den1 += 1
            if label in (1, 2):
                den2 += 1
            if label == _label:
                if _label == 2:
                    num += 1
                elif _label == 1:
                    num += (2 * jaccard(_pics, pics) + jaccard(_segs, segs)) / 3

        if num == 0:
            recall = precision = f = 0.0
        else:
            recall = float(num) / den1
            precision = float(num) / den2
            f = 2 * recall * precision / (recall + precision)

    except Exception, e:
        return 1, str(e)

    return 0, (f, recall, precision)

def usage():

    print '''
    match tools, for document character filtering or result evaluation.

    ./tools.py --method <method> --source <source_file> --target <target_file> --filtered <filtered_file>
    or for short,
    ./tools.py -m <method> -s <source_file> -t <target_file> -f <filtered_file>

    method|-m: filter or eval, filter for document character filtering, and eval for result evaluation
    source|-s: input file for method filter, or your label file for method eval
    target|-t: output file for method filter, or target label file for method eval
    filtered|-f: filtered file for method eval, file must be filtered first, please remember it


    Here are two examples for method filter and eval.

    filter:
        ./tools.py -m filter -s News_info_train_example100.txt -t News_info_train_example100_filter.txt
    eval:
        ./tools.py -m eval -s Your_label_example100.txt -t News_pic_label_train_example100.txt -f News_info_train_example100_filter.txt
    '''
    return


if __name__ == '__main__':

    method = 0
    source_file = ''
    target_file = ''
    filtered_file = ''
    try:
        if len(sys.argv) <= 1:
            usage()
            sys.exit(0)
        options, args = getopt.getopt(sys.argv[1:], "hm:s:t:f:", 
                ["help", "method=", "source=", "target=", "filtered="])
        for key, value in options:

            if key in ('-h', '--help'):
                usage()
                sys.exit(0)
            if key in ('-m', '--method'):
                method = value
                assert method in ('filter', 'eval')
            if key in ('-s', '--source'):
                source_file = value
            if key in ('-t', '--target'):
                target_file = value
            if key in ('-f', '--filtered'):
                filtered_file = value
        if method == 'filter':
            assert source_file != '' and \
                    target_file != ''
        else:
            assert source_file != '' and \
                    target_file != '' and \
                    filtered_file != ''
    except Exception, e:
        print e
        sys.exit(1)

    if method == 'filter':
        html_filter_file(source_file, target_file)
    else:
        ret, res = eval_file(filtered_file, target_file, source_file)
        if ret != 0:
            print(res)
            sys.exit(ret)
        else:
            print 'f_measure:\t%.6f\nrecall:\t\t%.6f\nprecision:\t%.6f' % \
                    (res[0], res[1], res[2])

    sys.exit(0)

#coding:utf8

import sys
filter_file = sys.argv[1] 
sen_file = sys.argv[2] 
pre_file = sys.argv[3] 
result_file = sys.argv[4] 
#打开最终预测文件
pre = open(pre_file,'r').readlines()
#打开分句文件
sen = open(sen_file, 'r')
#打开filter文件
content = open(filter_file, 'r')
keywords = ['来源:',
'关注微信公众号',
'微信公众号',
'公众号:',
'微信号',
'官方微信',
'http://www.',
'关注我们',
'阅读原文',
'电话:',
'微信:',
'转载',
'新闻网',
'信息来源',
'联系微信',
'编辑',
'编 辑',
'版权归',
'原创',
'搜狐号:',
'微信搜索',
'后台回复',
'联系删除',
'联系我们删除']

con_dict = {}
for i in content:
    con_dict[i.split('\t')[0]] = i.split('\t')[1]

sen_dict = {}
for i in sen:
    if i.split('\t')[0] not in sen_dict.keys():
        sen_dict[i.split('\t')[0]] = []
    else:
        sen_dict[i.split('\t')[0]].append(i.strip().split('\t')[1].strip())



result = open(result_file, 'w')

for i in pre:
    result.writelines(i.split('\t')[0])
    result.writelines('\t')
    result.writelines(i.split('\t')[1])
    result.writelines('\t')
    result.writelines(i.split('\t')[2])
    result.writelines('\t')
    if i.split('\t')[1]=='1':
        if i.split('\t')[2]=='NULL':
            flag=1
            try:
                for s in sen_dict[i.split('\t')[0]]:
                    for k in keywords:
                        if k in s:
                            if s in con_dict[i.split('\t')[0]]:
                                result.writelines(s)
                                flag=0
                        if flag == 0:
                            break
                    if flag==0:
                        break
            except:
                flag = 0
                result.writelines('NULL')
            if flag:
                result.writelines('NULL')
        else:
            result.writelines('NULL')
    else:
        result.writelines('NULL')
    result.writelines('\n')

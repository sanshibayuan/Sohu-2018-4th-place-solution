from ml_models import *
import os

train = pd.read_csv('News_info_train_filter.csv',sep = '\t',header = None)
train_ = pd.read_csv('News_info_train_filter_seg.csv',sep = '\t',header = None)
test = pd.read_table('News_info_validate_filter.txt',sep =  '\t' , header = None)
test_ = pd.read_table('News_info_validate_filter_seg.txt',sep =  '\t' , header = None)
train_y = to_categorical(train[2])

def tfidf_word(train_text,test_text):
	if os.path.exists('word_vec.pk'):
		print  ('load pre_trained char vectorizer...')
		with open('word_vec.pk', 'rb') as fin:
		    vec = pickle.load(fin)
		fin.close()		
	else:
		vec = TfidfVectorizer(
		            max_df=0.95, min_df=2,
		            sublinear_tf=True,
		            strip_accents='unicode',
		            analyzer='word',
		            ngram_range=(1, 3),
		            max_features=30000)
		all_text = pd.concat([train_text,test_text])
		print  ('fit...')
		vec.fit(all_text)
		with open ('word_vec.pk','wb') as fin:
		    pickle.dump(vec,fin,protocol = 2 )
		fin.close()		
	#查看字典
	sorted(vec.vocabulary_.items(),key = lambda x:x[1],reverse = True)

	print  ('transforming...')
	trn_term_doc = vec.transform(train_text)
	test_term_doc = vec.transform(test_text)
	return trn_term_doc,test_term_doc


def tfidf_word(train_text,test_text):#未分词的文本
	if os.path.exists('char_vec.pk'):
		print  ('load pre_trained char vectorizer...')
		with open('char_vec.pk', 'rb') as fin:
		    vec = pickle.load(fin)
		fin.close()		
	else:
		vec = TfidfVectorizer(
		            max_df=0.95, min_df=2,
		            sublinear_tf=True,
		            strip_accents='unicode',
		            analyzer='word',
		            ngram_range=(1, 3),
		            max_features=30000)
		all_text = pd.concat([train_text,test_text])
		print  ('fit...')
		vec.fit(all_text)
		with open ('char_vec.pk','wb') as fin:
		    pickle.dump(vec,fin,protocol = 2 )
		fin.close()		
	#查看字典
	sorted(vec.vocabulary_.items(),key = lambda x:x[1],reverse = True)
	print  ('transforming...')
	trn_term_doc = vec.transform(train_text)
	test_term_doc = vec.transform(test_text)
	return trn_term_doc,test_term_doc

trn_term_doc,test_term_doc = tfidf_word(train_[1],test_[1])
trn_term_doc2,test_term_doc2 = tfidf_char(train[1],test[1])

lgbm_prds = model_lgb(trn_term_doc,test_term_doc,train_y,len(test))
svm_preds = model_nbsvm(trn_term_doc,test_term_doc,train_y,len(test))

lgbm_prds2 = model_lgb(trn_term_doc2,test_term_doc2,train_y,len(test))
svm_preds2 = model_nbsvm(trn_term_doc2,test_term_doc2,train_y,len(test))


prds = []
preds = np.zeros((len(test),train_y[0].shape))
for i in range(0,,len(test)):
    preds[i] = ((lgbm_prds[i]+svm_preds[i])/2)
    x = np.argmax((preds[i]))
#    if x == 1:
#        x = 0
    prds.append(x)
print (prds.count(0),prds.count(1),prds.count(2))
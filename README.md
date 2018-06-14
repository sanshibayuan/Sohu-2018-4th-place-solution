# Sohu-2018
2018 搜狐内容识别算法大赛

# Overview

Preprocessing
--
- Html filter
- Segmentation
- Extra-features
- Data Augementation


Task1：Label Classification 
--
EDA
- Word_tfidf
- Char_tfidf
- Word2vec

Models
- NBSVM	
- LGBM
- TextCNN	
- RCNN	
- Bi-LSTM
- Bi-GRU

Ensemble
- Word2vec dimentions
- Embedding layer
- 01-2 0-1 classification

Task2：Text Extraction
--
- Keywords
- Extract text

Task3：Image Classification
--
- Text Recognition
- Text Classification
- Area Filtering (CTPN)



Futher Thoughts
--
- Use pre-trained Chinese word-vectors https://github.com/Embedding/Chinese-Word-Vectors
- Use word2vec vector in embedding layer
- Any better way to classify images like these?
- Implement seq2seq in text_extract
- How to use these unlabel data better? Autoencoder/Dual Learning/Semi-supervised Learning


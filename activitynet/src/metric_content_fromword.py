import numpy as np
import cPickle as pkl
import sys
import nltk.tokenize as tk
import nltk.stem.porter as pt
import nltk.stem.lancaster as lc
import nltk.stem.snowball as sb
import nltk.stem as ns
import re
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
import os
from sklearn.metrics.pairwise import cosine_similarity


lemmatizer = ns.WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
glove_word_fts = np.load('../activitynet_data/glove.840B.300d_dict.npy')
glove_word_fts = glove_word_fts.tolist()

def encoder_sentence_with_words(sentence_list):
	all_sentence_fts_list = []
	for doc in sentence_list:
		doc = doc.lower()
		doc = re.sub(r'[.!,;?]', ' ', doc)
		word_tokens = word_tokenize(doc)
		filtered_words = [w for w in word_tokens if not w in stop_words]

		if len(filtered_words) != 0:
			all_fts_list = []
			for word in filtered_words:
				n_lemma = lemmatizer.lemmatize(word, pos='n')
				v_lemma = lemmatizer.lemmatize(n_lemma, pos='v')
				if v_lemma in glove_word_fts:
					all_fts_list.append(glove_word_fts[v_lemma])
			doc_fts = np.mean(np.array(all_fts_list),0)
			all_sentence_fts_list.append(doc_fts)
		else:
			print 'no meaningful words'
			all_sentence_fts_list.append(np.zeros(300,)+0.0)

	all_sentence_fts_list = np.array(all_sentence_fts_list)
	print np.shape(all_sentence_fts_list)
	return all_sentence_fts_list


def compute_sentence_similar(predict,content):
	predict_num = len(predict)
	content_num = len(content)
	all_sentence_list = predict+content
	all_fts = encoder_sentence_with_words(all_sentence_list)

	similarity = cosine_similarity(all_fts)
	max_list = []
	mean_list = []
	for i in range(predict_num):
		current_max = np.max(similarity[i,predict_num:])
		current_mean = np.mean(similarity[i,predict_num:])
		max_list.append(current_max)
		mean_list.append(current_mean)
	return np.mean(max_list),np.mean(mean_list)
	


i = 82
while i <= 6000:
	name = 'results/control_output_dict'+str(i)+'.pkl'
	print name

	if os.path.exists(name):

		content_score_file = open('./results/content_score_byGloveWord.txt','a')
		results = pkl.load(open(name,'r'))
		all_similarity_max_list= []
		all_similarity_mean_list = []

		count = 0
		for video in results:
			content_list = results[video]['content']
			predict_list = results[video]['predict']
			process_predict_list = []
			process_content_list = []

			for predict in predict_list:
				predict = predict.replace('<sos>','')
				predict = predict.replace('<eos>','')
				predict = predict.strip()
				if predict == '':
					predict = '.'
				process_predict_list.append(predict)

			for content in content_list:
				content = content.replace('<sos>','')
				content = content.replace('<eos>','')
				content = content.strip()
				if content == '':
					content = '.'
				process_content_list.append(content)


			max_num, mean_num = compute_sentence_similar(process_predict_list,process_content_list)
					
			all_similarity_max_list.append(max_num)
			all_similarity_mean_list.append(mean_num)


			print count
			count +=1


		max_similarity_score = np.mean(np.array(all_similarity_max_list))
		mean_similarity_score = np.mean(np.array(all_similarity_mean_list))
		content_score_file.write(name+'\n')
		content_score_file.write('mean max similariy score: '+str(max_similarity_score)+ '\n')
		content_score_file.write('mean mean similariy score: '+str(mean_similarity_score)+ '\n')
		content_score_file.close()

		print len(all_similarity_max_list)
		print len(all_similarity_mean_list)
		print max_similarity_score
		print mean_similarity_score
		print '***********************************************'

	i = i + 2
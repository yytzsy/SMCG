import numpy as np
import cPickle as pkl
import sys
sys.path.insert(0,'/DATA-NFS/yuanyitian/skip-thoughts')
import skipthoughts
import os
from sklearn.metrics.pairwise import cosine_similarity

skip_model = skipthoughts.load_model()
skip_encoder = skipthoughts.Encoder(skip_model)

def compute_sentence_similar(predict,content):
	predict_num = len(predict)
	content_num = len(content)
	all_fts = skip_encoder.encode(predict+content)

	# predict_fts = all_fts[0:predict_num,:]
	# content_fts = all_fts[predict_num:,:]
	
	similarity = cosine_similarity(all_fts)
	max_list = []
	mean_list = []
	for i in range(predict_num):
		current_max = np.max(similarity[i,predict_num:])
		current_mean = np.mean(similarity[i,predict_num:])
		max_list.append(current_max)
		mean_list.append(current_mean)
	return np.mean(max_list),np.mean(mean_list)
	


i = 0
while i <= 6000:
	name = 'results/control_output_dict'+str(i)+'.pkl'
	print name

	if os.path.exists(name):

		content_score_file = open('./results/content_score.txt','a')
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

	i = i + 50









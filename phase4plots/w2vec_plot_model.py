import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

with open("doc2vec_grid_search_results/hist_file_name.txt") as f:
    file_name_list = f.read().splitlines()

print("filenamelist:  ",len(file_name_list))
results = pd.read_csv("doc2vec_grid_search_results/results2.csv", header=0)
print("len(results)",len(results))   
print("aaaaaaaaaaaaaaaaaaa",len(file_name_list))

markList = ['-', '--', '-.']
colorList = ['black', 'green', 'red', 'yellow', 'pink']

#i=0 (b solid), i=1 black solid, i=1 b dash, i=3 greed solid
abs_count = 0;
max_hist = [0,0,0,0];
max_acc = 0;
max_acc_index = 0;
for i in [0,1,2,3]:
	counter=0;
	
	curr_max = 0;
	inner_count = 0;
	for file_name in file_name_list[i*15:(i+1)*15]:
		history = pd.read_csv("doc2vec_grid_search_results/"+file_name);
		print("count: ",abs_count)
		if abs_count == 45:
			print(history['val_accuracy'].values)
		if history['val_accuracy'].values[-1] > curr_max:
			print("ppppp", history['val_accuracy'].values[-1])
			curr_max = history['val_accuracy'].values[-1];
			max_hist[i] = abs_count;
			
			
		if history['val_accuracy'].values[-1] > max_acc:
			#print("ppppp", history['val_accuracy'].values[-1])
			max_acc = history['val_accuracy'].values[-1];
			max_acc_index = abs_count;
			
		#Plot training & validation accuracy values
		#plt.plot(history['accuracy'], color = "black", label="Bias")
		
		print("        ",counter // 8)
		print(counter % 8)
		
		plt.plot(history['val_accuracy'],color=colorList[counter % 5], linestyle=markList[counter // 5])
		plt.title(results['activation'][abs_count] + \
		"\nweighted=" + str(results['weighted'][abs_count]))
		plt.ylabel('Accuracy')
		plt.xlabel('Epoch')
		plt.legend(["- 1 layer", "-- 2 layers", "-. 3 layers", ": 4layers",\
		'black: 2 node', 'green: 4 nodes', 'red: 8 nodes', 'yellow: 16 nodes',\
		 'pink: 32 nodes', 'orange: 64 nodes', 'brown: 128 nodes', 'blue: 256 nodes'], loc='upper left')
		counter += 1;
		abs_count += 1;
	print('------------------------------------------------')	

	plt.savefig("w2vec_"+str(i),dpi = 200)	
	plt.figure()
	
	print(max_hist)

for i in max_hist:
	history = pd.read_csv("doc2vec_grid_search_results/"+file_name_list[i]);
	#+,nodes,activation,optimizer,loss_function
	print(results['activation'][i])
	plt.plot(history['val_accuracy'],label=results['activation'][i] + \
		"\nweighted=" + str(results['weighted'][i])+
		"\nnodes="+str(results['nodes'][i])+"\nlayers="+str(results['layers'][i])\
		+"\nloss_function="+str(results['loss_function'][i]))
	#plt.title('Testing Accuracy vs Epoch Using doc2vec')
	plt.ylabel('Testing Accuracy')
	plt.xlabel('Epoch')
	plt.legend(fontsize=5)
plt.savefig("doc2vec_summary",dpi = 600)	
plt.figure()













#starting tfidf stuff

with open("../round2/grid_search_results/hist_file_name.txt") as f:
    file_name_list = f.read().splitlines()

tfidf_results = pd.read_csv("../round2/grid_search_results/results2.csv", header=0)


markList = ['-', '--', '-.',':']
colorList = ['black', 'green', 'red', 'yellow', 'pink', 'orange', 'brown', 'blue']

#i=0 (b solid), i=1 black solid, i=1 b dash, i=3 greed solid
abs_count = 0;
max_hist = [0,0,0,0];
tfidf_max_acc = 0;
tfidf_max_acc_index = 0;
for i in [0,1,2,3]:
	counter=0;
	
	curr_max = 0;
	inner_count = 0;
	for file_name in file_name_list[i*32:(i+1)*32]:
		history = pd.read_csv("../round2/grid_search_results/"+file_name);
		
		if history['val_accuracy'].values[-1] > curr_max:
			print("ppppp", history['val_accuracy'].values[-1])
			curr_max = history['val_accuracy'].values[-1];
			max_hist[i] = abs_count;
			
		# Plot training & validation accuracy values
		#plt.plot(history['accuracy'], color = "black", label="Bias")
		
		print("        ",counter // 8)
		print(counter % 8)
		
		plt.plot(history['val_accuracy'],color=colorList[counter % 8], linestyle=markList[counter // 8])
		plt.title(tfidf_results['activation'][abs_count] + \
		"\nweighted=" + str(tfidf_results['weighted'][abs_count]))
		plt.ylabel('Accuracy')
		plt.xlabel('Epoch')
		plt.legend(["- 1 layer", "-- 2 layers", "-. 3 layers", ": 4layers",\
		'black: 2 node', 'green: 4 nodes', 'red: 8 nodes', 'yellow: 16 nodes',\
		 'pink: 32 nodes', 'orange: 64 nodes', 'brown: 128 nodes', 'blue: 256 nodes'], loc='upper left')
		counter += 1;
		abs_count += 1;
	print('------------------------------------------------')	

	plt.savefig("tfidf_"+str(i),dpi = 200)	
	plt.figure()
	
	print(max_hist)

for i in max_hist:
	history = pd.read_csv("../round2/grid_search_results/"+file_name_list[i]);
	#+,nodes,activation,optimizer,loss_function
	print(tfidf_results['activation'][i])
	plt.plot(history['val_accuracy'],label=tfidf_results['activation'][i] + \
		"\nweighted=" + str(tfidf_results['weighted'][i])+
		"\nnodes="+str(tfidf_results['nodes'][i])+"\nlayers="+str(tfidf_results['layers'][i])\
		+"\nloss_function="+str(tfidf_results['loss_function'][i]))
	#plt.title('Testing Accuracy vs Epoch Using tf-idf')
	plt.ylabel('Testing Accuracy')
	plt.xlabel('Epoch')
	plt.legend(fontsize=5)
plt.savefig("tfidf_summary",dpi = 600)
plt.figure()















#layers,nodes,activation,optimizer,loss_function,epochs,batch_size,weighted
file_name = tfidf_results['file_name'][tfidf_max_acc_index][:-3]+".history";	
history = pd.read_csv("../round2/grid_search_results/"+file_name);
plt.plot(history['val_accuracy'],label="tf-idf: "+tfidf_results['loss_function'][tfidf_max_acc_index]);
for i,row in tfidf_results.iterrows():
	if row['layers'] == tfidf_results['layers'][tfidf_max_acc_index] and  \
		row['nodes'] == tfidf_results['nodes'][tfidf_max_acc_index] and \
		row['weighted'] == tfidf_results['weighted'][tfidf_max_acc_index] and \
		row['loss_function'] != tfidf_results['loss_function'][tfidf_max_acc_index]:
		file_name = row['file_name'][:-3]+".history";	
		history = pd.read_csv("../round2/grid_search_results/"+file_name);
		plt.plot(history['val_accuracy'],label="tf-idf: "+row['loss_function']);
		print("i is: ",i)
plt.legend(fontsize=5)
	











doc2vec_results = results;
#layers,nodes,activation,optimizer,loss_function,epochs,batch_size,weighted
file_name = doc2vec_results['file_name'][max_acc_index][:-3]+".history";	
history = pd.read_csv("doc2vec_grid_search_results/"+file_name);
plt.plot(history['val_accuracy'],label="doc2vec: "+doc2vec_results['loss_function'][max_acc_index]);
for i,row in doc2vec_results.iterrows():
	if row['layers'] == doc2vec_results['layers'][max_acc_index] and  \
		row['nodes'] == doc2vec_results['nodes'][max_acc_index] and \
		row['weighted'] == doc2vec_results['weighted'][max_acc_index] and \
		row['loss_function'] != doc2vec_results['loss_function'][max_acc_index]:
		file_name = row['file_name'][:-3]+".history";	
		history = pd.read_csv("doc2vec_grid_search_results/"+file_name);
		plt.plot(history['val_accuracy'],label="doc2vec: "+row['loss_function']);
		print("i is: ",i)
plt.ylabel('Testing Accuracy')
plt.xlabel('Epoch')		
plt.legend()
plt.savefig("loss_functions",dpi = 600)	
		

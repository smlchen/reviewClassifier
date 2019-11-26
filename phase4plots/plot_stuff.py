import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import pandas as pd
import numpy as np

with open("doc2vec_grid_search_results/hist_file_name.txt") as f:
    d2v_file_name_list = f.read().splitlines()

d2v_results = pd.read_csv("doc2vec_grid_search_results/results2.csv", header=0)


#i=0 (b solid), i=1 black solid, i=1 b dash, i=3 greed solid
d2v_max_acc = 0;
d2v_max_acc_index = 0;
count = 0;
for file_name in d2v_file_name_list:
	history = pd.read_csv("doc2vec_grid_search_results/"+file_name);	
			
	if history['val_accuracy'].values[-1] > d2v_max_acc:
		d2v_max_acc = history['val_accuracy'].values[-1];
		d2v_max_acc_index = count;
	count += 1

#starting tfidf stuff
with open("../round2/grid_search_results/hist_file_name.txt") as f:
    tfidf_file_name_list = f.read().splitlines()

tfidf_results = pd.read_csv("../round2/grid_search_results/results2.csv", header=0)

tfidf_max_acc = 0;
tfidf_max_acc_index = 0;
count = 0;
for file_name in tfidf_file_name_list:
	
	history = pd.read_csv("../round2/grid_search_results/"+file_name);	
	if history['val_accuracy'].values[-1] > tfidf_max_acc:
		tfidf_max_acc = history['val_accuracy'].values[-1];
		tfidf_max_acc_index = count;
	count +=1

#layers,nodes,activation,optimizer,loss_function,epochs,batch_size,weighted
file_name = tfidf_results['file_name'][tfidf_max_acc_index][:-3]+".history";	
print(file_name)
history = pd.read_csv("../round2/grid_search_results/"+file_name);
plt.plot(history['val_accuracy'],label="tf-idf: "+tfidf_results['loss_function'][tfidf_max_acc_index]);
for i,row in tfidf_results.iterrows():
	if row['layers'] == tfidf_results['layers'][tfidf_max_acc_index] and  \
		row['nodes'] == tfidf_results['nodes'][tfidf_max_acc_index] and \
		row['weighted'] == tfidf_results['weighted'][tfidf_max_acc_index] and \
		row['optimizer'] == tfidf_results['optimizer'][tfidf_max_acc_index] and \
		row['activation'] == tfidf_results['activation'][tfidf_max_acc_index] and \
		row['loss_function'] != tfidf_results['loss_function'][tfidf_max_acc_index]:
		file_name = row['file_name'][:-3]+".history";	
		history = pd.read_csv("../round2/grid_search_results/"+file_name);
		print(file_name)

		plt.plot(history['val_accuracy'],label="tf-idf: "+row['loss_function']);
		print("i is: ",i)
	
	
#layers,nodes,activation,optimizer,loss_function,epochs,batch_size,weighted
file_name = d2v_results['file_name'][d2v_max_acc_index][:-3]+".history";	
history = pd.read_csv("doc2vec_grid_search_results/"+file_name);
print(file_name)
plt.plot(history['val_accuracy'],label="doc2vec: "+d2v_results['loss_function'][d2v_max_acc_index]);
for i,row in d2v_results.iterrows():
	if row['layers'] == d2v_results['layers'][d2v_max_acc_index] and  \
		row['nodes'] == d2v_results['nodes'][d2v_max_acc_index] and \
		row['weighted'] == d2v_results['weighted'][d2v_max_acc_index] and \
		row['optimizer'] == d2v_results['optimizer'][d2v_max_acc_index] and \
		row['activation'] == d2v_results['activation'][d2v_max_acc_index] and \
		row['loss_function'] != d2v_results['loss_function'][d2v_max_acc_index]:
		file_name = row['file_name'][:-3]+".history";	
		history = pd.read_csv("doc2vec_grid_search_results/"+file_name);
		plt.plot(history['val_accuracy'],label="doc2vec: "+row['loss_function']);
		print("i is: ",i)
plt.ylabel('Testing Accuracy')
plt.xlabel('Epoch')		
plt.legend()
plt.savefig("loss_functions",dpi = 600)	
plt.figure()
	
		
		
#Layer, Node Dependence.
		
		
#layers,nodes,activation,optimizer,loss_function,epochs,batch_size,weighted
file_name = tfidf_results['file_name'][tfidf_max_acc_index][:-3]+".history";	
print(file_name)
history = pd.read_csv("../round2/grid_search_results/"+file_name);
plt.plot(history['val_accuracy'],label="tf-idf: "+ str(tfidf_results['layers'][tfidf_max_acc_index]) + " layers");
for i,row in tfidf_results.iterrows():
	if 	row['nodes'] == tfidf_results['nodes'][tfidf_max_acc_index] and \
		row['weighted'] == tfidf_results['weighted'][tfidf_max_acc_index] and \
		row['loss_function'] == tfidf_results['loss_function'][tfidf_max_acc_index]and \
		row['layers'] != tfidf_results['layers'][tfidf_max_acc_index]:
		file_name = row['file_name'][:-3]+".history";	
		history = pd.read_csv("../round2/grid_search_results/"+file_name);
		print(file_name)

		plt.plot(history['val_accuracy'],label="tf-idf: "+str(row['layers']) + " layers");
		print("i is: ",i)
	
	
#layers,nodes,activation,optimizer,loss_function,epochs,batch_size,weighted
file_name = d2v_results['file_name'][d2v_max_acc_index][:-3]+".history";	
history = pd.read_csv("doc2vec_grid_search_results/"+file_name);
print(file_name)
plt.plot(history['val_accuracy'],label="dov2vec: "+str(d2v_results['layers'][d2v_max_acc_index]) + " layers");
for i,row in d2v_results.iterrows():
	if 	row['nodes'] == d2v_results['nodes'][d2v_max_acc_index] and \
		row['weighted'] == d2v_results['weighted'][d2v_max_acc_index] and \
		row['loss_function'] == d2v_results['loss_function'][d2v_max_acc_index] and\
		row['layers'] != d2v_results['layers'][d2v_max_acc_index]:
		file_name = row['file_name'][:-3]+".history";	
		history = pd.read_csv("doc2vec_grid_search_results/"+file_name);
		plt.plot(history['val_accuracy'],label="dov2vec: "+str(row['layers'])+ " layers");
		print("i is: ",i)
plt.ylabel('Testing Accuracy')
plt.xlabel('Epoch')		
plt.legend()
plt.savefig("layers",dpi = 600)	
plt.figure()

def plot_nodes(sample_type, max_acc_index, results, path):
	if sample_type == "doc2vec":
		colors = pl.cm.inferno(np.linspace(0,1,10))
	else:
		colors = pl.cm.jet(np.linspace(0,1,10))
	file_name = results['file_name'][max_acc_index][:-3]+".history";	
	history = pd.read_csv(path+"/"+file_name);
	print(file_name)
	count = 0;
	plt.plot(history['val_accuracy'],label=sample_type+": "+str(results['nodes'][max_acc_index]) + " nodes",color=colors[0]);
	for i,row in results.iterrows():
		if 	row['nodes'] != results['nodes'][max_acc_index] and \
			row['weighted'] == results['weighted'][max_acc_index] and \
			row['loss_function'] == results['loss_function'][max_acc_index] and\
			row['layers'] == results['layers'][max_acc_index]:
			file_name = row['file_name'][:-3]+".history";	
			history = pd.read_csv(path+"/"+file_name);
			plt.plot(history['val_accuracy'],label=sample_type+": "+str(row['nodes'])+ " nodes",color=colors[i+1]);
			print("i is: ",i)
			count += 1
	plt.ylabel('Testing Accuracy')
	plt.xlabel('Epoch')	
	
	
plot_nodes("doc2vec", d2v_max_acc_index, d2v_results, "doc2vec_grid_search_results")
plot_nodes("tf-idf",tfidf_max_acc_index,  tfidf_results, "../round2/grid_search_results")
plt.legend()
plt.savefig("nodes",dpi = 600)	
plt.figure()


def plot_weighted(sample_type, max_acc_index, results, path):
	if sample_type == "doc2vec":
		colors = pl.cm.Set1(np.linspace(0,1,10))
	else:
		colors = pl.cm.Set2(np.linspace(0,1,10))
	file_name = results['file_name'][max_acc_index][:-3]+".history";	
	history = pd.read_csv(path+"/"+file_name);
	print(file_name)
	count = 0;
	plt.plot(history['val_accuracy'],label=sample_type+": weighted="+str(results['weighted'][max_acc_index]),color=colors[0]);
	for i,row in results.iterrows():
		if 	row['nodes'] == results['nodes'][max_acc_index] and \
			row['weighted'] != results['weighted'][max_acc_index] and \
			row['loss_function'] == results['loss_function'][max_acc_index] and\
			row['layers'] == results['layers'][max_acc_index]:
				
			file_name = row['file_name'][:-3]+".history";	
			history = pd.read_csv(path+"/"+file_name);
			plt.plot(history['val_accuracy'],label=sample_type+": weighted="+str(row['weighted']),color=colors[1*3]);
			print("i is: ",i)
			count += 1
	plt.ylabel('Testing Accuracy')
	plt.xlabel('Epoch')	
	
	
plot_weighted("doc2vec", d2v_max_acc_index, d2v_results, "doc2vec_grid_search_results")
plot_weighted("tf-idf",tfidf_max_acc_index,  tfidf_results, "../round2/grid_search_results")
plt.legend()
plt.savefig("weighted",dpi = 600)	
plt.figure()


def plot_activation(sample_type, max_acc_index, results, path):
	print(results.loc[[max_acc_index]])
	if sample_type == "doc2vec":
		colors = pl.cm.Set1(np.linspace(0,1,10))
	else:
		colors = pl.cm.Set2(np.linspace(0,1,10))
	file_name = results['file_name'][max_acc_index][:-3]+".history";	
	history = pd.read_csv(path+"/"+file_name);
	print(file_name)
	count = 0;
	plt.plot(history['val_accuracy'],label=sample_type+": "+str(results['activation'][max_acc_index]),color=colors[0]);
	for i,row in results.iterrows():
		if 	row['nodes'] == results['nodes'][max_acc_index] and \
			row['weighted'] == results['weighted'][max_acc_index] and \
			row['loss_function'] == results['loss_function'][max_acc_index] and\
			row['activation'] != results['activation'][max_acc_index] and\
			row['layers'] == results['layers'][max_acc_index]:
				
			file_name = row['file_name'][:-3]+".history";	
			history = pd.read_csv(path+"/"+file_name);
			plt.plot(history['val_accuracy'],label=sample_type+": "+str(row['activation']),color=colors[(1+count)*3]);
			print("i is: ",i)
			count += 1
	plt.ylabel('Testing Accuracy')
	plt.xlabel('Epoch')	
	
	
plot_activation("doc2vec", d2v_max_acc_index, d2v_results, "doc2vec_grid_search_results")
plot_activation("tf-idf",tfidf_max_acc_index,  tfidf_results, "../round2/grid_search_results")
plt.legend()
plt.savefig("activation",dpi = 600)	
plt.figure()

### What is this for?
This program will use Artificial Neural Network (ANN) to predict overall star rating from Amazon review texts. Both tf-idf and Doc2Vec as feature selection methods will be compared against each other, to see if data sparsity affects classification accuracy.

### Installation
+ `pip3 install pandas`
+ `pip3 install numpy`
+ `pip3 install keras`
+ `pip3 install tensorflow`
+ `pip3 install gensim`
+ `pip3 install scikit-learn`

### How to use reviewClassifier
**Obtaining the master random data over 28 categories of Amazon reviews.** 
Download the zipped data files, except the Appliances category, of the reviews from https://nijianmo.github.io/amazon/index.html. Make sure to modify the directory reference to point to the locations where the zipped files are. Create another directory, `large_data` to hold particularly large files like the Books_5.json.gz because it's over 6GB and would take too long to process. 

``` ./get_data.py ```


**Text preprocessing.** Uses the master csv file. Each sample’s review text is then tokenized and lowercase. Removed stop words such as articles, pronouns, and prepositions, as well as non-English words and spelling typos. The remaining words—which consisted of mostly nouns, adjectives, and verbs—were then processed further with lemmatization and stemming. Numbers are removed and the text is rejoined to be passed into tf-idf or Doc2Vec.

``` ./clean_text.py ```

**Term Frequency-Inverse Document Frequency (TF-IDF).** Computes a TF-IDF matrix given a dataframe and outputs a CSV file, along with the overall ratings for each review text as a separate column. Using five-fold cross validation, every set of training and test data is outputted to a CSV file and saved in the corresponding folders.

``` ./tf_idf.py ```

split folder contains 5 sets of training and test data. They were produced by doing a 5-fold cross validation after shuffling the original dataset.

In this case, five models will be trained and evaluated with each fold given a chance to be the held out test set. An example is shown below: 

+ Model 1: Train on Fold1 + Fold2 + Fold3 + Fold4, Test on Fold5
+ Model 2: Train on Fold2 + Fold3 + Fold4 + Fold5, Test on Fold1
+ Model 3: Train on Fold3 + Fold4 + Fold5 + Fold1, Test on Fold2
+ Model 4: Train on Fold4 + Fold5 + Fold1 + Fold2, Test on Fold3
+ Model 5: Train on Fold5 + Fold1 + Fold2 + Fold3, Test on Fold4

**Doc2Vec.** Turns a body of text into a vector with a given number of dimensions.The final trained value of this feature is then saved as the vector representation of the document. Using five-fold cross validation, every set of training and test data is outputted to a CSV file and saved in the corresponding folders.

``` ./doc2vec.py ```

**Building the model.** Implement a Feed Forward Network on both the Tf-idf and Doc2Vec matrix and a gridsearch to determine the best activation function, nodes, and layer hyperparameters. Once the output was hot label encoded, the matrix were ran through a grid search sweep to determine the most optimal hyperparameters for our two different models. Models are saved to ./grid_search_results/ for TF-IDF models and ./doc2vec_grid_search_results for Doc2Vec models. 

``` ./run_grid_search.py ```

#### Data Visualization
**Clustering.** The reviews are clustered to see if reviews with the same rating would be placed into the same cluster. To visualize the clusters from K-means, compare two dimensionality reduction methods, Principal Component Analysis (PCA) and t-Distributed Stochastic Neighbor Embedding (t-SNE).

``` ./clustering.py ```

** ROC and PR curves.** Plots of the ROC and PR curves are generated for the ratings for the saved models using 5-fold cross validation. Plots are saved in ./roc_plots/ and ./pr_plots/

``` ./roc_curves.py ```
``` ./pr_curves.py ```



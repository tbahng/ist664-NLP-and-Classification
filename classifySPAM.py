"""Train and Test a SPAM Classifier (See Experiment 5 in 'thomas_bahng_final.ipynb')
This program reads email data for the spam classification problem and can be executed from an IDE or command line.
This program expects the following directory structure for proper execution:
- project (folder)
    - classifySPAM_thomas_bahng_final.py (file)
    - corpus (folder)
        - ham (folder)
            - emails (files)
        - spam (folder)
            - emails (files)
The feature set used to train the model consists of 3000 unigrams, 1000 bigrams, 4 pos tag groupings, and 3 corpus-related statistics.    
Corpus statistics include measures of lexical richness, character count, and mean word length.
The Linear SVC algorithm from Sci-kit Learn is fitted with the data, and both a cross-validation classification report and test set evaluation are generated.
"""

# open python and nltk packages needed for processing
import time
import os
import sys
import random
import nltk
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from nltk.corpus import stopwords
import re
import pandas as pd 
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'white')
sns.set(style = 'whitegrid', color_codes = True)

# define a feature definition function here
# [feature definition function: experiment 4] function to get document features
# param document: a list of strings representing a tokenized email
# param word_features: a list of strings against which the tokens in document are matched
# param bigram_features: a list of tuples where each element is a bigram
# returns a dictionary where each key value is either 'contains(keyword)', normalized frequencies by POS tags, or corpus statistics
def document_features(document, word_features, bigram_features):
    document_words = set(document) # unigrams
    document_bigrams = nltk.bigrams(document) # bigrams
    document_pos = [t[1] for t in nltk.pos_tag(document)] # pos tags
    features = {}
    # unigram features
    for word in word_features:
        features['V_{}'.format(word)] = (word in document_words)
    # bigram features
    for bigram in bigram_features:
        features['B_{}_{}'.format(bigram[0], bigram[1])] = (bigram in document_bigrams)
    # pos features
    noun_count, verb_count, adj_count, adv_count = 0, 0, 0, 0
    for tag in document_pos:
        if tag.startswith('N'): noun_count += 1
        if tag.startswith('V'): verb_count += 1
        if tag.startswith('J'): adj_count += 1
        if tag.startswith('R'): adv_count += 1
    features['noun_count_norm'] = noun_count / len(document_pos)
    features['verb_count_norm'] = verb_count / len(document_pos)
    features['adj_count_norm'] = adj_count / len(document_pos)
    features['adv_count_norm'] = adv_count / len(document_pos)
    # corpus statistics
    features['lexical_richness'] = len(document_words) / len(document) # lexical richness of email
    features['total_char_count'] = 0 # total character count of email
    for word in document:
        features['total_char_count'] += len(word)
    word_lengths = [len(word) for word in document]
    features['mean_word_length'] = sum(word_lengths) / len(word_lengths) # mean word length of email
    return features  


# function to read spam and ham files, train and test a classifier 
def processspamham():
    start = time.clock() # start time    
    # start lists for spam and ham email texts
    hamtexts = []
    spamtexts = []  
    # process all files in directory that end in .txt up to the limit
    #    assuming that the emails are sufficiently randomized
    # function to get absolute filepaths in a directory
    # param directory: absolute directory name
    # yields absolute file paths
    def absoluteFilePaths(directory):
        for dirpath,_,filenames in os.walk(directory):
            for f in filenames:
                yield os.path.abspath(os.path.join(dirpath, f))
    mygenerator = absoluteFilePaths('corpus/spam')
    filelistspam = []
    for f in mygenerator:
        filelistspam.append(f)
    for f in filelistspam:
        if (f.endswith(".txt")):
            # open file for reading and read entire file into a string
            with open(f, 'r', encoding = 'latin-1') as fin:
                spamtexts.append(fin.read())
    mygenerator = absoluteFilePaths('corpus/ham')
    filelistham = []
    for f in mygenerator:
        filelistham.append(f)
    for f in filelistham:
        if (f.endswith(".txt")):
            # open file for reading and read entire file into a string
            with open(f, 'r', encoding = 'latin-1') as fin:
                hamtexts.append(fin.read())
    # print number emails read
    print ("Number of spam files:",len(spamtexts))
    print ("Number of ham files:",len(hamtexts))
  
    # create list of mixed spam and ham email documents as (list of words, label)
    documents = []
    # add all the spam
    for spam in spamtexts:
        tokens = nltk.word_tokenize(spam)
        documents.append((tokens, 'spam'))
    # add all the regular emails
    for ham in hamtexts:
        tokens = nltk.word_tokenize(ham)
        documents.append((tokens, 'ham'))
    print("There are a total of {:d} documents".format(len(documents)))
    # randomly shuffle the documents for training and testing classifier
    random.seed(111)
    random.shuffle(documents)
    # possibly filter tokens
    from nltk.corpus import stopwords
    stopwords = stopwords.words('english')
    # function that identifies non-alphabetic tokens
    # param w: string word
    # returns true if word consists only of non-alphabetic characters 
    def alpha_filter(w):
        # pattern to match a word of non-alphabetical characters
        pattern = re.compile('^[^a-z]+$')
        if pattern.match(w):
            return True
        else:
            return False

    # continue as usual to get all words and create word features
    
    # function to get word features
    # param documents: a list of tuples where the first item of each tuple is the tokenized email text
    # param stopwords: a list of strings where each element is a stopword
    # returns a list of 3000 strings where each string is a word feature
    def getWordFeatures(documents, stopwords):    
        # lower-case conversion of complete document tokenization
        all_words_list = [word.lower() for (email, cat) in documents for word in email]
        # filter for alphabetic words
        all_words_list = [word for word in all_words_list if not alpha_filter(word)]
        # exclude stopwords
        keep_words = set(all_words_list) - set(stopwords)
        all_words_list = [word for word in all_words_list if word in keep_words]
        all_words = nltk.FreqDist(all_words_list)
        # get the 1500 most frequently appearing keywords in all words
        word_items = all_words.most_common(3000)
        word_features = [word for (word, count) in word_items]
        return word_features

    # function to get bigram features
    # param documents: a list of tuples where the first item of each tuple is the tokenized email text
    # param stopwords: a list of strings where each element is a stopword
    # returns a list of 1000 tuples where each element is a bigram feature
    def getBigramFeatures(documents, stopwords):
        # lower-case conversionof complete document tokenization
        all_words_list = [word.lower() for (email, cat) in documents for word in email]
        # Top 1000 bigram feature extraction
        measures = BigramAssocMeasures()
        finder = BigramCollocationFinder.from_words(all_words_list) # scorer
        finder.apply_word_filter(alpha_filter) # exclude non-alphabetic words
        finder.apply_word_filter(lambda w: w in stopwords) # exclude stop words    
        scored = finder.score_ngrams(measures.raw_freq)
        bigram_features = [s[0] for s in scored[:1000]]
        return bigram_features
    
    # feature sets from a feature definition function
    # function to get feature sets for modeling in Experiment 2
    def getFeatureSets(documents, stopwords):
        # get word features based on specified stopwords
        word_features = getWordFeatures(documents, stopwords)
        # get bigram features based on specified stopwords
        bigram_features = getBigramFeatures(documents, stopwords)
        featuresets = [(document_features(d, word_features, bigram_features), c) for (d, c) in documents]
        return featuresets
    print("\nRunning feature extraction...")
    featureset = getFeatureSets(documents, stopwords)
    features = [f for (f,c) in featureset]
    labels = [c for (f,c) in featureset]
    X = pd.DataFrame(features)
    y = np.array(labels)
    X_train = X.iloc[:3620, :]
    X_test = X.iloc[3620:, :]
    y_train = y[:3620]
    y_test = y[3620:]
    print("\nTraining set has {:d} observations".format(X_train.shape[0]))
    print("Test set has {:d} observations".format(X_test.shape[0]))
    print("Number of predictors: {:d}".format(X.shape[1]))
    # train classifier and show performance in cross-validation
    print("\nRunning 10-fold cross-validation...\n")
    classifier = LinearSVC(C=1, penalty='l1', dual=False, class_weight='balanced')
    np.random.seed(111)
    y_pred = cross_val_predict(classifier, X_train, y_train, cv=10)
    print("\nCross-validation classification report:")
    print(classification_report(y_train, y_pred))
    # train classifier and predict test set
    # function to compute precision, recall, and f1 for each label and for any number of labels
    # param gold: list of strings where each element is a gold label
    # param predicted: list of strings where each element is a predicted label (in same order)
    # output: prints precision, recall, f1 for each class
    def eval_measures(gold, predicted):
        # get a list of labels
        labels = list(set(gold))
        # initialize list of class-specific scores
        precision_list, recall_list, f1_list = [],[],[]
        for lab in labels:
            # for each label, compare gold and predicted lists and compute values
            TP = FP = FN = TN = 0
            for i, val in enumerate(gold):
                if val == lab and predicted[i] == lab:  TP += 1
                if val == lab and predicted[i] != lab:  FN += 1
                if val != lab and predicted[i] == lab:  FP += 1
                if val != lab and predicted[i] != lab:  TN += 1
            # formulas for precision, recall, and f1
            precision = TP / (TP + FN)
            recall = TP / (TP + FP)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append( 2 * (recall * precision) / (recall + precision))
        # the evaluation measures in a table with one row per label
        print('class\tPrecision\tRecall\tF1\n')
        # print measures for each label
        for i, lab in enumerate(labels):
            print(lab, '\t', "{:10.3f}".format(precision_list[i]), \
            "{:10.3f}".format(recall_list[i]), "{:10.3f}".format(f1_list[i]))
    # function to plot confusion matrix
    # param gold: list of strings where each element is a gold label
    # param predicted: list of strings where each element is a predicted label (in same order)
    # output: plots confusion matrix with sklearn
    def getCM(gold, predicted):
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(gold, predicted)
        # plot heatmap
        class_names=[0,1] # name  of classes
        fig, ax = plt.subplots()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names)
        plt.yticks(tick_marks, class_names)
        sns.heatmap(pd.DataFrame(cm), annot=True, cmap = "YlGnBu", fmt = 'g')
        ax.xaxis.set_label_position("top")
        plt.tight_layout()
        plt.title('Confusion matrix', y=1.1)
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.show()
    print("\nTest set prediction")
    svm = classifier.fit(X_train, y_train)
    preds = svm.predict(X_test)
    eval_measures(y_test, preds)
    end = time.clock() # end time
    processingTime = round(end - start, 1)
    print("Operation completed in {:.1f} seconds".format(processingTime))
    getCM(y_test, preds)    

processspamham()    
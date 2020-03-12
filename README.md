# ist664-NLP-and-Classification

Purpose: The objective of this project is to demonstrate classification of SPAM in a repository of emails using NLP techniques and toolkits

Team: Thomas Bahng tbahng@syr.edu

Course: IST664

Term: Winter 2020

Instructor: Michael Larche mlarche@syr.edu

Data: https://github.com/tbahng/ist664-NLP-and-Classification/tree/master/corpus

Notebook Filename: thomas_bahng_final.ipynb

Acknowledgement goes to the main instructor Nancy McCraken in providing baseline code "classifySPAM.py" and data. 
The data is for detecting Spam emails from the Enron public email corpus.
In addition to some small numbers of Spam already in the corpus, additional spam emails were introduced into each user’s email stream in order to have a sufficient number of spam examples to train a classifier. The non-Spam emails are labeled “ham”. (See this paper for details: http://www.aueb.gr/users/ion/docs/ceas2006_paper.pdf ) The dataset that we have was gleaned from their web site at http://www.aueb.gr/users/ion/data/enron-spam/.
Although there are 3 large directories of both Spam and Ham emails, only the first one is used here with 3,672 regular emails in the “ham” folder, and 1,500 emails in the “spam” folder.

The project notebook will include data extraction, exploration, and multiple modeling experiments. 

These experiments will apply pre-processing, feature engineering and classification specific to the scenario.
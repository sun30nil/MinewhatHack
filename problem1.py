import pandas as pd
import sys
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import os

_inputDataSetFile = 'products.csv'
_outputResultSetFile = 'problem1Sol.csv'

def predictBasedOnTheTrainingData(x_VariableList, y_VariableListOfLists, testSetList):
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn import preprocessing
    # print len(y_VariableListOfLists), len(x_VariableList)

    X_train = np.array(x_VariableList)
    y_train_text = y_VariableListOfLists

    X_test = np.array(testSetList)

    lb = preprocessing.MultiLabelBinarizer()  
    Y = lb.fit_transform(y_train_text)

    classifier = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', OneVsRestClassifier(LinearSVC()))])

    classifier.fit(X_train, Y)
    predicted = classifier.predict(X_test)
    all_labels = lb.inverse_transform(predicted)
    classifiedList = []
    for item, labels in zip(X_test, all_labels):
        classifiedList.append(labels)
        # print '%s => %s' % (item, ', '.join(labels))
    return classifiedList

def textMiningUsingNltk(rawTextToBeMined):
    # word_tokenize = RegexpTokenizer(r'\w+')
    extendedStopWords = ["'", "s", ".", ",", "@", ":", ";", "!", "#", "$", "&", "*", "(", ")", "+", "=", "|", "[", "]", "{", "}", "%", "?"]
    tokenizedWords = word_tokenize(rawTextToBeMined.lower())
    listAfterStopWordsRemoved = []
    corporaStopWordsList = stopwords.words('english')
    corporaStopWordsList.extend(extendedStopWords)
    for eachWord in tokenizedWords:
        if eachWord not in corporaStopWordsList:
            listAfterStopWordsRemoved.append(eachWord)

    porter_stemmer = PorterStemmer()
    counter = 0
    stemmedList = []
    for each in listAfterStopWordsRemoved:
        counter += 1
        stemmedList.append( porter_stemmer.stem(each) )
    return stemmedList
    pass

print 'reading input file...'
df = pd.read_csv(_inputDataSetFile, index_col=0)
df_problem1 = df[['name', 'description', 'cat']]
print 'splitting train and test....'
df_problem1_train = df_problem1[df_problem1.cat != '["uncategorised"]']
df_problem1_test = df_problem1[df_problem1.cat == '["uncategorised"]']
df_result_set = df_problem1_test.copy(deep=True)

X_train_list = []
Y_train_list = []
print 'extracting features for train set....'
for index, row in df_problem1_train.iterrows():
	trainStr = ' '.join(textMiningUsingNltk(unicode(row['name'] + row['description'], errors='ignore')))
	X_train_list.append(trainStr)
	Y_train_list.append(eval(row['cat']))
X_test_list = []
print 'extracting features for test set....'
for index, row in df_problem1_test.iterrows():
	testStr = ' '.join(textMiningUsingNltk(unicode(row['name'] + row['description'], errors='ignore')))
	X_test_list.append(testStr)

print "Training ...."
predictedLabels = predictBasedOnTheTrainingData(X_train_list, Y_train_list, X_test_list)
print "Predicted ...."
predictedLabels = [str(list(each)) for each in predictedLabels]
df_result_set.drop('description', axis=1, inplace=True)
df_result_set['cat'] = predictedLabels
df_result_set.to_csv(_outputResultSetFile, index=True)
print "Result Successfully Saved the predicted values to ", os.path.abspath(_outputResultSetFile)

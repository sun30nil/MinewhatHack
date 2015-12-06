import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

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

df = pd.read_csv('products.csv')
df_problem1 = df[['_id', 'name', 'description', 'cat']]
df_problem1_train = df_problem1[df_problem1.cat != '["uncategorised"]']
df_problem1_test = df_problem1[df_problem1.cat == '["uncategorised"]']

X_train_list = []
Y_train_list = []
for index, row in df_problem1_train.iterrows():
	trainStr = ' '.join(row['name'].strip().split() + textMiningUsingNltk(unicode(row['description'], 'utf-8',errors='ignore')))
	# this thie feature string 
	print trainStr
	X_train_list.append(trainStr.decode('utf-8'))
	Y_train_list.append(eval(row['cat']))

X_test_list = []
for index, row in df_problem1_test.iterrows():
	testStr = ' '.join(row['name'].strip().split() + textMiningUsingNltk(unicode(row['description'], 'utf-8',errors='ignore')))
	X_test_list.append(unicode(testStr, 'utf-8', errors='ignore'))

predictedLabels = predictBasedOnTheTrainingData(X_train_list, Y_train_list, X_test_list)
print predictedLabels
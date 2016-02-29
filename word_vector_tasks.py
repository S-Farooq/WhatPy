import nltk
from transcript import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn import neighbors
from sklearn import svm
import random


def corpus_vectorizer(corpus, n_grams):
    #Creates a dictionary where each unique unigram (single word) and bigram (2 consecutive words) is represented by an index in a vector
    #Converts all the training data into a set of vectors
    #Converts test data into vectors depending on vectorizer dictionary developed from training data
    #Return the vectorizer

    vectorizer = CountVectorizer(min_df=1, decode_error='replace', ngram_range=n_grams)
    train, test = train_test_split(corpus, test_size=0.10, random_state=42)
    train = vectorizer.fit_transform(train)
    test = vectorizer.transform(test)
    return vectorizer, train, test

def Classify(clf,X,y, Xt, yt, v,vt):
    clf.fit(X, y.toarray())
    Z = clf.predict(Xt)
    val = 0.0
    for i in range(len(Z)):
        p = np.argmax(Z[i])
        o = np.argmax(yt.toarray()[i])
        print vt.vocabulary_.keys()[vt.vocabulary_.values().index(p)]
        print vt.vocabulary_.keys()[vt.vocabulary_.values().index(o)]
        if p == o:
            val += 1
        print

    print val/len(Z)

def vector_classify(msg_corpus, speakers):
    # #Read CSV File row by row and send each msg to tagger
    # with open(chat_log, 'rb') as f:
    #     reader = csv.reader(f)
    #     train_data = list(reader)
    #
    # print len(train_data)


    #Vectorize
    v, X_train, X_test = corpus_vectorizer(msg_corpus[:], (1,2))
    vt, y_train, y_test = corpus_vectorizer(speakers[:], (1,1))
    print "Vectorized, shape of train data:"
    print X_train.shape
    print y_train.shape

    clf = neighbors.KNeighborsClassifier(14, weights='distance')
    #clfsvm = svm.SVC()
    Classify(clf,X_train,y_train,X_test,y_test,v,vt)

if __name__ == '__main__':

    chat_log = 'data/output_ham.csv'

    data = pd.read_csv(chat_log)
    spk1 = ['Shaham']
    spk2 = ['HammadMirza']
    data1 = data[data.Speaker.isin(spk1)]
    data2 = data[data.Speaker.isin(spk2)]

    minm = min([data1.shape[0],data2.shape[0]])
    print 'Minimum per speaker: ' + str(minm)

    #format data into msg and label
    msg_corpus = np.concatenate((list(data1.Text)[:minm],list(data2.Text)[:minm]))
    speakers = np.concatenate((['speaker1']*minm,['speaker2']*minm))

    #vector_classify(msg_corpus,speakers)

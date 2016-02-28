__author__ = 'Shaham'
import nltk
from transcript import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn import neighbors
from sklearn import svm
import random
from NLPlib import *
nlp_tag = NLPlib()

patternL = []
with open('Wordlists/laughs.txt', 'rb') as f:
    for word in f:
        if len(word)>1:
            w = word[:-2]
            patternL.append(w)
            patternL.append('$')
            patternL.append('|')
pattern = "".join(patternL[:-1])
#print pattern
laughs = re.compile(pattern,re.IGNORECASE)

patternL = []
with open('Wordlists/Slang2', 'rb') as f:
    for word in f:
        if len(word)>2:
            w = word[:-2]
            patternL.append('^')
            patternL.append(w)
            patternL.append('$')
            patternL.append('|')

pattern = "".join(patternL[:-1])
#print pattern
slang = re.compile(pattern,re.IGNORECASE)

rlg= re.compile('alhamdu|subhan|astag|^ia$', re.IGNORECASE) #Religious phrases
corr= re.compile('(.+)\*$', re.IGNORECASE) #Corrections
emojis = re.compile(u'^\\xf0') #emojis
def count_tags(msg):

def tag_msgs(msg):
    #If the msg is <media omitted> we count that as an entity in itself
    if re.search('^<Media\somitted>$',msg):
        tokenized_text = [(msg,'MEDIA')]
    else:
        #TODO: Deal with Names, Slang (including lol, lmao, rofl, etc.), Emoticons, ASCII characters, and maybe specific words we are interested in (iA,salam, etc.)
        #TODO: instead of just word tokenization, consider sentance tokenizer? maybe specific phrases, idk
        text = nltk.word_tokenize(msg) #separate msg by words
        tagged = nlp_tag.tag(text)
        #tokenized_text = nltk.pos_tag(text) #Tokenize each word using NLTK tokenizer
        tokenized_text = [('','')]*len(text)
        for i in range(len(text)):
            tokenized_text[i] = (text[i], tagged[i])

        #After tokenization changes
        #Laughs

        for i in range(len(tokenized_text)):
            if rlg.match(tokenized_text[i][0]):
                tokenized_text[i] = (tokenized_text[i][0],'RLG')
            elif laughs.match(tokenized_text[i][0]):
                tokenized_text[i] = (tokenized_text[i][0],'LOL')
            elif slang.match(tokenized_text[i][0]):
                tokenized_text[i] = (tokenized_text[i][0],'SLG')
            elif corr.match(tokenized_text[i][0]):
                tokenized_text[i] = (tokenized_text[i][0],'CORR')
            elif emojis.match(tokenized_text[i][0]):
                tokenized_text[i] = (tokenized_text[i][0],'EMJ')

        #print pattern




    return tokenized_text

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
    TRAIN_SIZE = 15
    X =np.zeros((TRAIN_SIZE),dtype=list)
    random.seed(42)
    inds = random.sample(xrange(len(msg_corpus)), TRAIN_SIZE)
    #print inds
    for ind in range(TRAIN_SIZE):
        X[ind] = tag_msgs(msg_corpus[inds[ind]])
        print X[ind]
    # with open(chat_log, 'rb') as csvfile:
    #     reader = csv.reader(csvfile)
    #     #i is temporary just for testing pusposes so we only look at a certain amount of messages
    #     i = 0
    #     for row in reader:
    #         date, time, speaker, msg = row
    #         train_corpus.append(msg)
    #         train_labels.append(speaker)
    #         #print tag_msgs(row[-1])
    #         i += 1
    #         if i == 10:
    #             break

    #print len(train_corpus)
    #print len(train_labels)
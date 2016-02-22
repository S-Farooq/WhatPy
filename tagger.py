__author__ = 'Shaham'
import nltk
from transcript import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn import neighbors, datasets

def tag_msgs(msg):
    #If the msg is <media omitted> we count that as an entity in itself
    if re.search('^<Media\somitted>$',msg):
        tokenized_text =  [(msg,'MEDIA')]
    else:
        #TODO: Deal with Names, Slang (including lol, lmao, rofl, etc.), Emoticons, ASCII characters, and maybe specific words we are interested in (iA,salam, etc.)
        #TODO: instead of just word tokenization, consider sentance tokenizer? maybe specific phrases, idk
        text = nltk.word_tokenize(msg) #separate msg by words
        tokenized_text = nltk.pos_tag(text) #Tokenize each word using NLTK tokenizer


    return tokenized_text

def corpus_vectorizer(corpus, n_grams):
    vectorizer = CountVectorizer(min_df=1, decode_error='replace', ngram_range=n_grams)
    train, test = train_test_split(corpus, test_size=0.10, random_state=42)
    train = vectorizer.fit_transform(train)
    test = vectorizer.transform(test)
    return vectorizer, train, test

if __name__ == '__main__':
    chat_log = 'data/output_ham.csv'
    # #Read CSV File row by row and send each msg to tagger
    # with open(chat_log, 'rb') as f:
    #     reader = csv.reader(f)
    #     train_data = list(reader)
    #
    # print len(train_data)
    data = pd.read_csv(chat_log)
    data = data[data.Speaker.isin(['Qasim','Shamil'])]
    msg_corpus = list(data.Text)
    speakers = list(data.Speaker)
    v, X_train, X_test = corpus_vectorizer(msg_corpus[:], (1,2))
    vt, y_train, y_test = corpus_vectorizer(speakers[:], (1,1))

    print X_train.shape
    print y_train.shape
    print vt.vocabulary_
    n_neighbors = 3
    weights = 'distance'
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X_train, y_train.toarray())
    Z = clf.predict(X_test)
    val = 0.0
    for i in range(len(Z)):
        p = np.argmax(Z[i])
        o = np.argmax(y_test.toarray()[i])
        print vt.vocabulary_.keys()[vt.vocabulary_.values().index(p)]
        print vt.vocabulary_.keys()[vt.vocabulary_.values().index(o)]
        if p == o:
            val += 1
        print

    print val/len(Z)
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
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

# count_tags(msg) determines the number of times each feature appears in a message.
# Parameters: A message dictionary containing the message as the key and the feature as the value.
#             Dictionary of tags that are to be counted.
# Returns an array containing the total count for each feature.
def count_tags(msg, tagsDict):

    counts = []

    maxValue = max(tagsDict.values())

    counts = np.zeros((maxValue + 1))

    for word in msg:
        if word[1] in tagsDict:
            counts[tagsDict[word[1]]] += 1

    return counts

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

    return tokenized_text

def get_data(TRAIN_SIZE, num_features, speakers):
    random.seed(42)

    chat_log = 'data/output_ham.csv'

    data = pd.read_csv(chat_log)
    X =np.zeros((TRAIN_SIZE*len(speakers),num_features))
    Y =np.zeros((TRAIN_SIZE*len(speakers)))
    for i in range(len(speakers)):
        data_sub = data[data.Speaker.isin(speakers[i])]
        if data_sub.shape[0]<TRAIN_SIZE:
            print "Sorry, you only have " + str(data_sub.shape[0]) + "data points for " + str(speakers[i])
        data_sub = list(data_sub.Text)
        np.random.shuffle(data_sub)
        data_sub = data_sub[:TRAIN_SIZE]
        for m in range(TRAIN_SIZE):
            tagged_array = tag_msgs(data_sub[m])
            X[(TRAIN_SIZE*i)+m,:] = count_tags(tagged_array)
            Y[(TRAIN_SIZE*i)+m] = i


    # spk1 = ['Shaham']
    # spk2 = ['HammadMirza']
    # data1 = data[data.Speaker.isin(spk1)]
    # data2 = data[data.Speaker.isin(spk2)]
    #
    # minm = min([data1.shape[0],data2.shape[0]])
    # print 'Minimum per speaker: ' + str(minm)
    #
    # data1 = list(data1.Text)
    # data2 = list(data2.Text)
    #
    #
    #
    # #inds0 = random.sample(xrange(data1.shape[0]), TRAIN_SIZE)
    # #inds1 = random.sample(xrange(data2.shape[0]), TRAIN_SIZE)
    #
    #
    # #format data into msg and label
    # msg_corpus = np.concatenate((np.random.shuffle(list(data1.Text))[:TRAIN_SIZE],np.random.shuffle(list(data2.Text))[:TRAIN_SIZE]))
    # speakers = np.concatenate(([0]*TRAIN_SIZE,[1]*TRAIN_SIZE))
    #
    #
    # #print inds
    # for i in range(X.shape[0]):
    #     tagged_array = tag_msgs(msg_corpus[i])
    #     X[i,:] = count_tags(tagged_array)
    #     Y[i] = speakers[i]

    return X, Y

def fit_features(clf, X,Y):
    clf.fit(X, Y)
    return

def classify_features(clf, Xt,Yt):
    Z = clf.predict(Xt)
    val = 0.0
    for i in range(len(Z)):
        print Z[i]
    print val/len(Z)

    return

if __name__ == '__main__':

    spk1 = ['Shaham']
    spk2 = ['HammadMirza']

    X, Y = get_data(20,6,[spk1,spk2])
    print X
    print Y
    clf = neighbors.KNeighborsClassifier(4, weights='distance')

    TRAIN_SIZE = 15
    X =np.zeros((TRAIN_SIZE),dtype=list)
    random.seed(42)
    inds = random.sample(xrange(len(msg_corpus)), TRAIN_SIZE)
    #print inds
    for ind in range(TRAIN_SIZE):
        X[ind] = tag_msgs(msg_corpus[inds[ind]])
        print X[ind]

        ## count how many times certain tags appear in a message.
        tags = {'NN': 0, 'IN': 1, 'UH': 2, 'CC': 3, 'RB': 4, 'PRP': 5, 'VB': 6, 'EMJ': 7}
        tagCount = count_tags( X[ind], tags)
        print tagCount

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
    #print len(train_lab els)

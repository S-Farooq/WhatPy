__author__ = 'Shaham'
import nltk
from nltk.corpus import wordnet
from transcript import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn import neighbors
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
import random
from NLPlib import *
nlp_tag = NLPlib()
import matplotlib.pyplot as plt
import word2vec
import gensim


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

rlg = re.compile('alhamdu|subhan|astag|masha|^ia$', re.IGNORECASE) #Religious phrases
corr = re.compile('(.+)\*$', re.IGNORECASE) #Corrections
#emojis = re.compile(u'^\\xf0') #emojis
emojis = re.compile(u'('
    u'\ud83c[\udf00-\udfff]|'
    u'\ud83d[\udc00-\ude4f\ude80-\udeff]|'
    u'[\u2600-\u26FF\u2700-\u27BF])+',
    re.UNICODE)
all_emojis = {}
k = 0

# count_tags(msg) determines the number of times each feature appears in a message.
# Parameters: A message dictionary containing the message as the key and the feature as the value.
#             Dictionary of tags that are to be counted.
# Returns an array containing the total count for each feature.
def count_tags(msg, tagsDict):

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
        text = nltk.word_tokenize(msg.decode('utf-8')) #separate msg by words
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

def get_data(TRAIN_SIZE, num_features, speakers, chat_log):
    random.seed(42)

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
            tags = {'NN': 0, 'IN': 1, 'UH': 2, 'LOL': 3, 'RLG': 4, 'SLG': 5, 'VB': 6, 'EMJ': 7}
            X[(TRAIN_SIZE*i)+m,:] = count_tags(tagged_array, tags)
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

def get_time(TRAIN_SIZE, speakers, chat_log):
    random.seed(42)

    data = pd.read_csv(chat_log)
    X =np.zeros((TRAIN_SIZE*len(speakers),1)) ## speaker vs the time of msg.
    Y =np.zeros((TRAIN_SIZE*len(speakers)))

    for i in range(len(speakers)):
        data_sub = data[data.Speaker.isin(speakers[i])]
        if data_sub.shape[0]<TRAIN_SIZE:
            print "Sorry, you only have " + str(data_sub.shape[0]) + "data points for " + str(speakers[i])
        data_sub = list(data_sub.Time)

        np.random.shuffle(data_sub)
        data_sub = data_sub[:TRAIN_SIZE]

        # numTwentyFourTime will store the 24 hour time of the message in decimal form.
        # e.g, 2:30 pm will be stored as 14.50
        numTwentyFourTime = np.zeros(TRAIN_SIZE)

        # count_times[n] will store the number of times a message is sent in the nth hour of the 24 hour clock
        # e.g, count_times[15] corresponds to 3pm.
        # Only for debugging purposes. count_times won't be used anywhere.
        count_times = np.zeros(24)
        for j in range(TRAIN_SIZE):
            twentyFourTime = ''
            minutes = ''
            flagSkip = 0
            flagMinutes = 0

            # TODO(hammad): move the following into its own function and find a less hacky way to do this...
            for digit in data_sub[j]:
                if digit != ':' and digit != 'A' and digit != 'P' and flagSkip == 0 and flagMinutes == 0:
                    twentyFourTime += digit
                elif digit == ':':
                    flagSkip = 1
                    flagMinutes = 1
                elif flagMinutes == 1 and digit != 'P' and digit != 'A' and digit != 'M':
                    minutes += digit
                elif digit == 'P':
                    numTwentyFourTime[j] = float(twentyFourTime)
                    if numTwentyFourTime[j] != 12:
                        numTwentyFourTime[j] += 12
                    numTwentyFourTime[j] += (float(minutes)/60.0)
                elif digit == 'A':
                    numTwentyFourTime[j] = float(twentyFourTime)
                    if numTwentyFourTime[j] == 12:
                        numTwentyFourTime[j] = 0
                    numTwentyFourTime[j] += (float(minutes)/60.0)

            # Incrememnt the count for the hour in which the current message was sent.
            count_times[numTwentyFourTime[j]] += 1

            # Training data
            X[(TRAIN_SIZE*i)+j,:] = numTwentyFourTime[j]
            Y[(TRAIN_SIZE*i)+j] = i
            #print numTwentyFourTime[j]
        #print count_times
    return X, Y

def get_capital_letters(TRAIN_SIZE, speakers, chat_log):
    random.seed(42)

    data = pd.read_csv(chat_log)
    X =np.zeros((TRAIN_SIZE*len(speakers),1))
    Y =np.zeros((TRAIN_SIZE*len(speakers)))
    for i in range(len(speakers)):
        data_sub = data[data.Speaker.isin(speakers[i])]
        if data_sub.shape[0]<TRAIN_SIZE:
            print "Sorry, you only have " + str(data_sub.shape[0]) + "data points for " + str(speakers[i])
        data_sub = list(data_sub.Text)
        np.random.shuffle(data_sub)
        data_sub = data_sub[:TRAIN_SIZE]
        for m in range(TRAIN_SIZE):
            capital_letters = sum(1 for c in data_sub[m] if c.isupper()) # Think about excluding letters that automatically get capitalized (e.g, 'I' and first letter of msg)
            X[(TRAIN_SIZE*i)+m,:] = capital_letters
            Y[(TRAIN_SIZE*i)+m] = i
    return X, Y

def get_capital_words(TRAIN_SIZE, speakers, chat_log, type):
    random.seed(42)

    data = pd.read_csv(chat_log)
    X =np.zeros((TRAIN_SIZE*len(speakers),1))
    Y =np.zeros((TRAIN_SIZE*len(speakers)))
    for i in range(len(speakers)):
       # print speakers[i][0] + '---------------------'
        data_sub = data[data.Speaker.isin(speakers[i])]
        if data_sub.shape[0]<TRAIN_SIZE:
            print "Sorry, you only have " + str(data_sub.shape[0]) + "data points for " + str(speakers[i])
        data_sub = list(data_sub.Text)
        np.random.shuffle(data_sub)
        data_sub = data_sub[:TRAIN_SIZE]

        all_capital_messages = 0
        for m in range(TRAIN_SIZE):
            capital_words = 0
            word_list = re.sub("[^\w]", " ",  data_sub[m]).split()
            word_count = len(word_list)

            for word in word_list:
                if word.isupper() and word_count != 1: ## Doesn't take into account one word messages.
                    capital_words += 1

            percentage = 0
            if word_count != 0: ## Because emojis do not count as words. Could divide by zero if a msg is only emojis.
                percentage = (capital_words / float(word_count))*100.0

            if data_sub[m].isupper():
                all_capital_messages += 1

            if type == "percentage":
                X[(TRAIN_SIZE*i)+m,:] = percentage
                Y[(TRAIN_SIZE*i)+m] = i

        if type == "total":
            X[(TRAIN_SIZE*i)+m,:] = all_capital_messages
            Y[(TRAIN_SIZE*i)+m] = i

        #print "all caps"
        #print all_capital_messages
    return X, Y


def get_msg_length(TRAIN_SIZE, speakers, chat_log):
    random.seed(42)

    data = pd.read_csv(chat_log)
    X =np.zeros((TRAIN_SIZE*len(speakers),1))
    Y =np.zeros((TRAIN_SIZE*len(speakers)))
    for i in range(len(speakers)):
       # print speakers[i][0] + '---------------------'
        data_sub = data[data.Speaker.isin(speakers[i])]
        if data_sub.shape[0]<TRAIN_SIZE:
            print "Sorry, you only have " + str(data_sub.shape[0]) + "data points for " + str(speakers[i])
        data_sub = list(data_sub.Text)
        np.random.shuffle(data_sub)
        data_sub = data_sub[:TRAIN_SIZE]

        for m in range(TRAIN_SIZE):
            capital_words = 0
            word_list = re.sub("[^\w]", " ",  data_sub[m]).split()
            word_count = len(word_list)
            # print data_sub[m]
            # print word_count
            X[(TRAIN_SIZE*i)+m,:] = word_count
            Y[(TRAIN_SIZE*i)+m] = i

    return X, Y

def get_name_refs(TRAIN_SIZE, speakers, chat_log):
    random.seed(42)

    data = pd.read_csv(chat_log)
    allSpeakers = ['Shaham', 'Shamil', 'Qasim', 'Abdullah', 'Hammad', 'Usamah', 'Mahmoud', 'Belal']
    X =np.zeros((TRAIN_SIZE*len(speakers),len(allSpeakers)))
    Y =np.zeros((TRAIN_SIZE*len(speakers)))
    for i in range(len(speakers)):
       # print speakers[i][0] + '---------------------'
        data_sub = data[data.Speaker.isin(speakers[i])]
        if data_sub.shape[0]<TRAIN_SIZE:
            print "Sorry, you only have " + str(data_sub.shape[0]) + "data points for " + str(speakers[i])
        data_sub = list(data_sub.Text)
        np.random.shuffle(data_sub)
        data_sub = data_sub[:TRAIN_SIZE]

        for m in range(TRAIN_SIZE):
            name_refs = 0
            ref_count = [0,0,0,0,0,0,0,0]
            word_list = re.sub("[^\w]", " ",  data_sub[m]).split()

            for word in word_list:
                for spkr in allSpeakers:
                    if spkr.lower() == word.lower():
                        # name_refs += 1
                        # print word
                        if spkr.lower() == 'shaham':
                            ref_count[0] += 1
                        if spkr.lower() == 'shamil':
                            ref_count[1] += 1
                        if spkr.lower() == 'qasim':
                            ref_count[2] += 1
                        if spkr.lower() == 'abdullah':
                            ref_count[3] += 1
                        if spkr.lower() == 'hammad':
                            ref_count[4] += 1
                        if spkr.lower() == 'usamah':
                            ref_count[5] += 1
                        if spkr.lower() == 'mahmooud':
                            ref_count[6] += 1
                        if spkr.lower() == 'belal':
                            ref_count[7] += 1

            X[(TRAIN_SIZE*i)+m,:] = ref_count
            Y[(TRAIN_SIZE*i)+m] = i

    return X, Y

def get_word_vectors(TRAIN_SIZE, speakers, chat_log, model):
    random.seed(42)

    data = pd.read_csv(chat_log)
    X =np.zeros((TRAIN_SIZE*len(speakers),1))
    Y =np.zeros((TRAIN_SIZE*len(speakers)))

    for i in range(len(speakers)):
        data_sub = data[data.Speaker.isin(speakers[i])]
        if data_sub.shape[0]<TRAIN_SIZE:
            print "Sorry, you only have " + str(data_sub.shape[0]) + "data points for " + str(speakers[i])
        data_sub = list(data_sub.Text)
        np.random.shuffle(data_sub)
        data_sub = data_sub[:TRAIN_SIZE]

        success = 0
        fail = 0

        for m in range(TRAIN_SIZE):
            word_list = re.sub("[^\w]", " ",  data_sub[m]).split()

            for word in word_list:
                if word.lower() in model:
                    indexes, metrics = model.cosine(word.lower())
                    success += 1
                else:
                    fail += 1

            X[(TRAIN_SIZE*i)+m,:] = 1
            Y[(TRAIN_SIZE*i)+m] = i

        print "% of words from this speaker that exist in the wikipedia corpus: "
        print (success / float(success + fail))*100.0


    return X, Y

# def get_emoji_data(TRAIN_SIZE, speakers, chat_log):
#
#
#     random.seed(42)
#
#
#     data = pd.read_csv(chat_log)
#
#     ## Obtain emoji data.
#     emojis = {}
#     emoji_index = 0
#     for i in range(len(speakers)):
#         data_sub = data[data.Speaker.isin(speakers[i])]
#         if data_sub.shape[0]<TRAIN_SIZE:
#             print "Sorry, you only have " + str(data_sub.shape[0]) + "data points for " + str(speakers[i])
#         data_sub = list(data_sub.Text)
#         np.random.shuffle(data_sub)
#         data_sub = data_sub[:TRAIN_SIZE]
#         for m in range(TRAIN_SIZE):
#             tagged_array = tag_msgs(data_sub[m])
#             for i in range(len(tagged_array)):
#                 if tagged_array[i][1] == "EMJ":
#                     current_emoji = tagged_array[i][0].encode('unicode_escape').split('\\')
#                     for j in range(1,len(current_emoji)):
#                         if current_emoji[j] not in emojis:
#
#                         # if current_emoji[j] in emojis:
#                         #     emojis[current_emoji[j]] += 1
#                         # else:
#                         #     emojis[current_emoji[j]] = 0
#
#         print emojis
#
#         X =np.zeros((TRAIN_SIZE*len(speakers),len(emojis)))
#         Y =np.zeros((TRAIN_SIZE*len(speakers)))
#
#             # current_emojis = tokenized_text[i][0].encode('unicode_escape').split('\\')
#             # for j in range (1,len(current_emojis)):
#             #     k += 1
#             #     if current_emojis[j] in all_emojis:
#             #         #all_emojis[current_emojis[j]] += 1
#             #         print "k"
#             #     else:
#             #         all_emojis[current_emojis[j]] = k#1
#             # print all_emojis
#
#
#     return X, Y
#
#
#
#
#
#
#     # random.seed(42)
#     # data = pd.read_csv(chat_log)
#     # X =np.zeros((TRAIN_SIZE*len(speakers),len(all_emojis)))
#     # Y =np.zeros((TRAIN_SIZE*len(speakers)))
#     #
#     # ## Fill emoji array.
#     # if emojis.match(tokenized_text[i][0]):
#     #     tokenized_text[i] = (tokenized_text[i][0],'EMJ')
#     #     current_emojis = tokenized_text[i][0].encode('unicode_escape').split('\\')
#     #     for j in range (1,len(current_emojis)):
#     #         k += 1
#     #         if current_emojis[j] in all_emojis:
#     #             #all_emojis[current_emojis[j]] += 1
#     #             print "k"
#     #         else:
#     #             all_emojis[current_emojis[j]] = k#1
#     #     print all_emojis
#     #
#     # for i in range(len(speakers)):
#     #     emojis = all_emojis
#     #     emoji_count = np.zeros(len(all_emojis))
#     #     data_sub = data[data.Speaker.isin(speakers[i])]
#     #     if data_sub.shape[0]<TRAIN_SIZE:
#     #         print "Sorry, you only have " + str(data_sub.shape[0]) + "data points for " + str(speakers[i])
#     #     data_sub = list(data_sub.Text)
#     #     np.random.shuffle(data_sub)
#     #     data_sub = data_sub[:TRAIN_SIZE]
#     #     for m in range(TRAIN_SIZE):
#     #         tagged_array = tag_msgs(data_sub[m])
#     #         #print tagged_array
#     #         for t in tagged_array:
#     #             if t[1] == "EMJ":
#     #                 split_emojis = t[0].encode('unicode_escape').split('\\')
#     #                 for j in range(1,len(split_emojis)):
#     #                     if split_emojis[j] in emojis:
#     #                         emojis[split_emojis[j]] += 1
#     #                     else:
#     #                         emojis[split_emojis[j]] = 1
#     #         print emojis
#     #         for e in range(len(emojis)):
#     #             print emojis(e)
#     #             emoji_count[e] = emojis[e]
#     #         X[(TRAIN_SIZE*i)+m,:] = emoji_count#count_tags(tagged_array, tags)
#     #         Y[(TRAIN_SIZE*i)+m] = i
#     #     print emojis
#     # return X, Y


def fit_features(clf, X,Y):
    clf.fit(X, Y)
    return

def classify_features(clf, Xt,Yt):
    Z = clf.predict(Xt)
    val = 0.0
    for i in range(len(Z)):
        if Z[i] == Yt[i]:
            val +=1

    print val/len(Z)

    return

def min_occurrences( speakers, chat_log ):
    data = pd.read_csv(chat_log)
    dataX = np.zeros(len(speakers))
    for i in range(len(speakers)):
        dataX[i] = len(data[data.Speaker.isin(speakers[i])])

    return int(min(dataX))

if __name__ == '__main__':

    chat_log = 'data/output_ham.csv'
    spk1 = ['HammadMirza']
    spk2 = ['Shaham']
    # spk3 = ['Shaham']
    # spk4 = ['BelalSaleem']
    # spk5 = ['Mahmoud']
    # spk6 = ['HammadMirza']
    # spk7 = ['Abdullah']
    # spk8 = ['UsamahWadud']
    min_train_size = min_occurrences( [spk1, spk2], chat_log )
    print "Min train size " + str(min_train_size)

    print "Classifying tag counts"
    X, y = get_data(min_train_size,8,[spk1, spk2], chat_log)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    clf = neighbors.KNeighborsClassifier(4, weights='distance')
    fit_features(clf,X_train, y_train)
    classify_features(clf, X_test, y_test)

    # TRAIN_SIZE = 15
    # X =np.zeros((TRAIN_SIZE),dtype=list)
    # random.seed(42)
    # inds = random.sample(xrange(len(msg_corpus)), TRAIN_SIZE)
    # #print inds
    # for ind in range(TRAIN_SIZE):
    #     X[ind] = tag_msgs(msg_corpus[inds[ind]])
    #     print X[ind]
    #
    #     ## count how many times certain tags appear in a message.
    #     tags = {'NN': 0, 'IN': 1, 'UH': 2, 'CC': 3, 'RB': 4, 'PRP': 5, 'VB': 6, 'EMJ': 7}
    #     tagCount = count_tags( X[ind], tags)
    #     print tagCount

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

    ## TIME FEATURE
    print "Classifying time"
    X, y = get_time(min_train_size,[spk1, spk2], chat_log)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    clf = neighbors.KNeighborsClassifier(4, weights='distance')
    fit_features(clf,X_train, y_train)
    classify_features(clf, X_test, y_test)

    # plt.scatter(y, X)
    # plt.ylabel('hours')
    # plt.xlabel('0 = '+spk1[0]+', 1 = '+spk2[0])
    # plt.xlim(-0.5,1.5)
    # plt.ylim(-1,25)
    # plt.show()

    # clf = GaussianNB()
    # fit_features(clf, X_train,y_train)
    # classify_features(clf, X_test, y_test)


    ## Number of capital letters in a message feature
    print "Classifying Number of Capital Letters"
    X, y = get_capital_letters(min_train_size,[spk1, spk2], chat_log)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    clf = neighbors.KNeighborsClassifier(4, weights='distance')
    fit_features(clf,X_train, y_train)
    classify_features(clf, X_test, y_test)

    # plt.scatter(y, X)
    # plt.ylabel('hours')
    # plt.xlabel('0 = '+spk1[0]+', 1 = '+spk2[0])
    # plt.xlim(-0.5,1.5)
    # plt.ylim(-1,20)
    # plt.show()


    ## Number of capital words in a message feature
    ## TODO(hammad): fix the all type of get_capital_words
    print "Classifying number of capital words in a message"
    X, y = get_capital_words(min_train_size,[spk1, spk2], chat_log, "percentage")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    clf = neighbors.KNeighborsClassifier(4, weights='distance')
    fit_features(clf,X_train, y_train)
    classify_features(clf, X_test, y_test)

    print "Classifying length of message"
    X, y = get_msg_length(min_train_size,[spk1, spk2], chat_log)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    clf = neighbors.KNeighborsClassifier(4, weights='distance')
    fit_features(clf,X_train, y_train)
    classify_features(clf, X_test, y_test)

    print "Classifying other speaker name references"
    X, y = get_name_refs(min_train_size,[spk1, spk2], chat_log)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    clf = neighbors.KNeighborsClassifier(4, weights='distance')
    fit_features(clf,X_train, y_train)
    classify_features(clf, X_test, y_test)

    ## Word2Vec
    model = word2vec.load('WordVecTest/text8.bin')
    get_word_vectors(50,[spk1, spk2], chat_log, model)

    # Gensim word2vec
    g_model = gensim.models.Word2Vec.load_word2vec_format( 'WordVecTest/text8.bin', binary = True )


    # plt.scatter(y, X)
    # plt.ylabel('hours')
    # plt.xlabel('0 = '+spk1[0]+', 1 = '+spk2[0])
    # plt.xlim(-0.5,1.5)
    # plt.ylim(-1,100)
    # plt.show()

    # X, y = get_emoji_data(min_train_size,[spk1,spk2], chat_log)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    # clf = neighbors.KNeighborsClassifier(4, weights='distance')
    # fit_features(clf,X_train, y_train)
    # classify_features(clf, X_test, y_test)


    # w1 = wordnet.synset('hello.n.01')
    # w2 = wordnet.synset('goodbye.n.01')
    # print w1.wup_similarity(w2)


    print "done"
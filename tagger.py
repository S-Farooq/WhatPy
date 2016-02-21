__author__ = 'Shaham'
import numpy as np
import nltk
import csv
import re

def tag_msgs(msg):
    #If the msg is <media omitted> we count that as an entity in itself
    if re.search('^<Media\somitted>$',msg):
        return [(msg,'MEDIA')]
    #TODO: Deal with Names, Slang (including lol, lmao, rofl, etc.), Emoticons, ASCII characters, and maybe specific words we are interested in (alhamdulillah, etc.)
    #TODO: instead of just word tokenization, consider sentance tokenizer? maybe specific phrases, idk
    text = nltk.word_tokenize(msg) #separate msg by words
    tokenized_text = nltk.pos_tag(text) #Tokenize each word using NLTK tokenizer




    return tokenized_text

if __name__ == '__main__':
    chat_log = 'data/output.csv'
    #Read CSV File row by row and send each msg to taggar
    with open(chat_log, 'rb') as csvfile:
        spamreader = csv.reader(csvfile)
        #i is temporary just for testing pusposes so we only look at a certain amount of messages
        i = 0
        for row in spamreader:
            print tag_msgs(row[-1])
            i += 1
            if i == 50:
                break

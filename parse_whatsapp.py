#The purpose of this code is to 
# Read input from whatsapp chat
# Parse the debate transcript into the following fields:
# 1. Sentence No. 2. Paragraph No. 3. Speaker 4. Conversation Text

import csv
from os import path
from transcript import *

def parse_whatsapp(chat,output):
    c = Transcript(chat, output)
    c.open_file()

    c.feed_lists()

    c.write_transcript()

    # Print all the unique speakers to clean up any unwanted sentences and only keep speakers
    data = pd.read_csv(output)
    print (('Unique Speakers: ', sorted(list(data.Speaker.unique()))))

if __name__ == "__main__":
    whatsapp_chat = 'whatsapp_chat.txt'
    output = 'output.csv'
    parse_whatsapp(whatsapp_chat,output)
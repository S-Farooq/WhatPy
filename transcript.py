#Include all the subtleties that are required to read a whatsapp chat transcript


import sys
import csv
import pandas as pd
import numpy as np
import codecs
import re

class Transcript():
	def __init__(self, inputFileName,outputFileName):
		self.inputFileName = inputFileName
		self.outputFileName = outputFileName
		self.raw_messages = []
		self.speakerlist = []
		self.messagelist = []

		self.datelist = []
		self.timelist = []

	def open_file(self):
		arq = codecs.open(self.inputFileName, "r", "utf-8-sig")
		content = arq.read()
		arq.close()
		lines = content.split("\n")
		lines = [l for l in lines if len(l) > 4]
		for l in lines:
			self.raw_messages.append(l.encode("utf-8", errors='replace'))
	def valid_date(self,date_str):
		valid = True
		separator="/"
		try:
			year, month, day = map(int, date_str.split(separator))
		except ValueError:
			valid = False
		return valid
	def feed_lists(self):
		lineNo = 0
		seqNo = 0
		for l in self.raw_messages:
			l = l.rstrip()
			partition = re.match('^([^,]+),\s*(\d+:\d+\s+\w+)\s+\-\s+([^:]+):\s(.+)$',l)
			if partition:
				msg_date, time, speaker, message = [partition.group(1),partition.group(2),partition.group(3), partition.group(4)]
				lineNo += 1

				self.datelist.append(msg_date)
				self.timelist.append(time)
				self.speakerlist.append(speaker)
				self.messagelist.append(message)
				# store the previous speaker so that you can use it to print when there is only a line
				prevSender = speaker
				prevRawDate = msg_date
				prevTime = time
				seqNo +=1
			else:
				partition2 = re.match('^([^,]+),\s*(\d+:\d+\s+\w+)\s+\-',l)
				if not partition2:
					print l
					self.datelist.append(prevRawDate)
					self.timelist.append(prevTime)
					self.speakerlist.append(prevSender)
					self.messagelist[-1] = self.messagelist[-1] + ' ' + l

	def write_transcript(self, end=0):
		if end == 0:
			end = len(self.messagelist)
		writer = csv.writer(open(self.outputFileName, 'w'))
		writer.writerow(["Date","Time","Speaker","Text"])
		for i in range(len(self.messagelist[:end])):
			writer.writerow([self.datelist[i], self.timelist[i],self.speakerlist[i], self.messagelist[i]])

	def get_speakers(self):
		speakers_set = set(self.speakerlist)
		return [e for e in speakers_set]

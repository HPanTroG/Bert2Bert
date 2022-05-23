#!/usr/bin/python3

'''
        Code: Concept Extraction
        AUTHOR: Koustav Rudra
'''

import sys
import os
from operator import itemgetter
import math
from textblob import *
import re
import time
import gzip
from collections import Counter
import times
import codecs
from datetime import datetime
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import itertools
import numpy as np
from numpy import linalg as LA

lmtzr = WordNetLemmatizer()
url = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
WORD = re.compile(r'\w+')

cachedstopwords = stopwords.words("english")
Tagger_Path = 'ark-tweet-nlp-0.3.2/' #Download and set Twitter pos-tagger path accordingly

AUX = ['be','can','cannot','could','am','has','had','is','are','may','might','dare','do','did','have','must','need','ought','shall','should','will','would','shud','cud','don\'t','didn\'t','shouldn\'t','couldn\'t','wouldn\'t']
NEGATE = ["aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt",
              "ain't", "aren't", "can't", "couldn't", "daren't", "didn't", "doesn't",
              "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt", "neither",
              "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't",
              "neednt", "needn't", "never", "none", "nope", "nor", "not", "nothing", "nowhere",
              "oughtnt", "shant", "shouldnt", "uhuh", "wasnt", "werent",
              "oughtn't", "shan't", "shouldn't", "uh-uh", "wasn't", "weren't",
              "without", "wont", "wouldnt", "won't", "wouldn't", "rarely", "seldom", "despite"]

def process_tweet(ifname,placefile,ofname):

    ### Extract Places ###
    PLACE = {}
    fp = open(placefile,'r')
    for l in fp:
        w = l.strip(' #\t\n\r').lower()
        if PLACE.__contains__(w)==False:
            PLACE[w] = 1
    fp.close()

    fp = open(ifname,'r')
    fo = open('temp.txt','w')
    Tweet = []
    for i,l in enumerate(fp):
        if i>=0:
            wl = l.split('\t')
            Tweet.append(wl)
            fo.write(wl[2].strip(' \t\n\r') + '\n')
    fp.close()
    fo.close()
    print('Data size {}'.format(len(Tweet)))

    command = Tagger_Path + './runTagger.sh --output-format conll temp.txt > tag.txt'
    os.system(command)

    RPL = ['+','-',',','+91']
    TAGREJECT = ['~','G','U','E','@',',','#']

    
    fo = open(ofname,'w')
    fp = open('tag.txt','r')
    index = 0
    count = 0
    L = 0
    temp = set([])
    All = set([])
    MAP = {}
    r = 0
    for l in fp:
        wl = l.split('\t')
        if len(wl)>1:
            word = wl[0].strip(' #\t\n\r').lower()
            tag = wl[1].strip(' \t\n\r')
            if tag not in TAGREJECT:
                L+=1
                if PLACE.__contains__(word)==True:
                    s = word + '_P'
                    temp.add(s)
                    All.add(s)
                elif tag=='$':
                    s = word
                    try:
                        Q = s
                        for x in RPL:
                            Q = s.replace(x,'')
                            s = Q
                        w = str(numToWord(int(s)))
                        if len(w.split())>1: # like 67
                            w = s
                    except Exception as e:
                        w = str(s)
                    word = w.lstrip('0')
                    s = word + '_S'
                    if wl[0].startswith('#')==False:
                        temp.add(s)
                        All.add(s)
                elif tag=='^' and word not in cachedstopwords and word not in NEGATE:
                    w = lmtzr.lemmatize(word)
                    s = w + '_PN'
                    if len(w)>2:
                        All.add(s)
                        temp.add(s)
                        MAP[word] = w
                elif tag=='N' and word not in cachedstopwords and word not in NEGATE:
                    w = lmtzr.lemmatize(word)
                    s = w + '_CN'
                    if len(w)>2:
                        All.add(s)
                        temp.add(s)
                        MAP[word] = w
                elif tag=='V':
                    try:
                        w = Word(word)
                        x = w.lemmatize("v")
                    except Exception as e:
                        x = word
                    if x not in AUX and x not in cachedstopwords and x not in NEGATE:
                        s = x + '_V'
                        MAP[word] = x
                    if len(x)>2:
                        All.add(s)
                        temp.add(s)
                else:
                    if word not in cachedstopwords:
                        s = word + '_A'
                        All.add(s)
                        #pass
        else:
            s = ''
            for x in temp:
                if len(x) > 0:
                    try:
                        s = s + x + ' '
                    except Exception as e:
                        pass
            
            z = ''
            for x in All:
                if len(x) > 0:
                    try:
                        z = z + x + ' '
                    except Exception as e:
                        pass
            temp1 = Tweet[index]
            rationale = temp1[4].split()
            R = ''
            for x in rationale:
                if MAP.__contains__(x)==True:
                    R = R + MAP[x].strip(' \t\n\r') + ' '
                else:
                    R = R + x.strip(' \t\n\r') + ' '
            try:
                p = str(count) + '\t' + temp1[1].strip(' \t\n\r') + '\t' + temp1[0].strip(' \t\n\r') + '\t' + temp1[2].strip(' \t\n\r') + '\t' + s.strip(' \t') + '\t' + z.strip(' \t\n\r') + '\t' + R.strip(' \t\n\r') + '\t' + str(L)
                fo.write(p)
                fo.write('\n')
                count+=1
            except Exception as e:
                print(temp1)
                print(e)
                sys.exit(0)
                r+=1
                pass
            index+=1
            temp = set([])
            All = set([])
            MAP = {}
            L = 0
    fp.close()
    fo.close()
    print('Reject: ',r)

def numToWord(number):
    word = []
    if number < 0 or number > 999999:
        return number
        # raise ValueError("You must type a number between 0 and 999999")
    ones = ["","one","two","three","four","five","six","seven","eight","nine","ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen","eighteen","nineteen"]
    if number == 0: return "zero"
    if number > 9 and number < 20:
        return ones[number]
    tens = ["","ten","twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety"]
    word.append(ones[int(str(number)[-1])])
    if number >= 10:
        word.append(tens[int(str(number)[-2])])
    if number >= 100:
        word.append("hundred")
        word.append(ones[int(str(number)[-3])])
    if number >= 1000 and number < 1000000:
        word.append("thousand")
        word.append(numToWord(int(str(number)[:-3])))
    for i,value in enumerate(word):
        if value == '':
            word.pop(i)
    return ' '.join(word[::-1])

def main():

    try:
        _, ifname, placefile, ofname = sys.argv
    except Exception as e:
        print(e)
        sys.exit(0)

    process_tweet(ifname,placefile,ofname)

if __name__=='__main__':
    main()

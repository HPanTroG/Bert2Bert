#!/usr/bin/python3
'''
	Code: Summarization --- EXCRIS
	AUTHOR: Koustav Rudra
'''


import sys
from collections import Counter
import re
from gurobipy import *
import gzip
from textblob import *
import os
import time
import codecs
import math
import codecs
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.corpus import wordnet_ic, genesis
import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from itertools import cycle
from operator import itemgetter
import string


WORD = re.compile(r'\w+')
cachedstopwords = stopwords.words("english")
punc = string.punctuation

AUX = ['be','can','cannot','could','am','has','had','is','are','may','might','dare','do','did','have','must','need','ought','shall','should','will','would','shud','cud','don\'t','didn\'t','shouldn\'t','couldn\'t','wouldn\'t']
NEGATE = ["aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt",
              "ain't", "aren't", "can't", "couldn't", "daren't", "didn't", "doesn't",
              "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt", "neither",
              "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't",
              "neednt", "needn't", "never", "none", "nope", "nor", "not", "nothing", "nowhere",
              "oughtnt", "shant", "shouldnt", "uhuh", "wasnt", "werent",
              "oughtn't", "shan't", "shouldn't", "uh-uh", "wasn't", "weren't",
              "without", "wont", "wouldnt", "won't", "wouldn't", "rarely", "seldom", "despite"]


def compute_summary(ifname,placefile,ofname):

    cachedstopwords.append(u'us')
    cachedstopwords.append(u'u.s.')

    PLACE = {}
    fp = codecs.open(placefile,'r','utf-8')
    for l in fp:
        w_org = l.strip(' \t\n\r')
        w_lower = w_org.lower()
        if PLACE.__contains__(w_org)==False:
            PLACE[w_org] = 1
        if PLACE.__contains__(w_lower)==False and w_lower not in cachedstopwords:
            PLACE[w_lower] = 1
    fp.close()

    fp = codecs.open(ifname,'r','utf-8')
    T = {}
    COT = {}
    TW = {}
    M = {}
    P = []
    index = 0
    count = 0
    word = {}
    coword = {}
    window = 0
    t0 = time.time()
	
    for l in fp:
        count+=1
        wl = l.split('\t')
        temp_I = wl[4].split()
        All_I = wl[5].split()
        Rationale_I = wl[6].split()
        text = wl[3].strip(' \t\n\r')
        Length = int(wl[7])
        Vscore = 1
        tid = wl[2].strip(' \t\n\r')
	
        temp = []
        for x in temp_I:
            x_0 = x.split('_')[0].strip(' \t\n\r')
            x_1 = x.split('_')[1].strip(' \t\n\r')
            if x_1=='PN':
                s = x_0 + '_CN'
                temp.append(s)
            else:
                temp.append(x)

        All = []
        for x in All_I:
            x_0 = x.split('_')[0].strip(' \t\n\r')
            x_1 = x.split('_')[1].strip(' \t\n\r')
            if x_1=='PN':
                s = x_0 + '_CN'
                All.append(s)
            else:
                All.append(x)

        Rationale = []
        for x in Rationale_I:
            WR = x.strip(' \t\n\r')
            if punc.__contains__(WR)==False and WR not in cachedstopwords and WR not in AUX and WR not in NEGATE and len(WR)>1:
                Rationale.append(WR)

        for x in temp:
            W = x.split('_')[0].strip(' \t\n\r')
            I = x.split('_')[1].strip(' \t\n\r')
            if I=='S':
                Rationale.append(W)

        ################### Update word dictionary  ###################################
        for x in Rationale:
            if word.__contains__(x)==True:
                v = word[x]
                v+=1
                word[x] = v
            else:
                word[x] = 1
		
        ################## Is it duplicate tweet ######################################

        k = compute_selection_criteria(All,TW)
        if k==1:
            T[index] = Rationale
            COT[index] = temp
            TW[index] = [tid,text,Rationale,All,Length,Vscore]
            index+=1
    fp.close()

    print('Total tweets: {} Selected tweets: {} %selection: {}'.format(count,index,round((index*100)/count,2)))
    	
    L = len(T.keys())
    print('L: ',L)
    weight = compute_rationale_weight_tf(word,count,PLACE)
    tweet_cur_window = {}
    check = set([])
    for i in range(0,L,1):
        temp = TW[i]
        tweet_cur_window[str(i)] = [temp[1],temp[4],set(temp[2]),temp[5],temp[0]]   ### Text, Length, Content words ###
        for x in temp[2]:
            check.add(x)
    print('Content: ',len(check))
    ##################### Finally apply cowts ################################
    optimize(tweet_cur_window,weight,ofname,200)
    print('Summarization done: ',ofname)
    window+=1
    t1 = time.time()
    print('Time Elapsed: ',t1-t0)


def compute_selection_criteria(current,T):
    for k,v in T.items():
        new = set(current).intersection(set(v[3]))
        if len(new)==len(current):
            return 0
    return 1

def optimize(tweet,weight,ofname,L):


    ################################ Extract Tweets and Content Words ##############################
    word = {}
    tweet_word = {}
    tweet_index = 1
    for  k,v in tweet.items():
        set_of_words = v[2]
        for x in set_of_words:
            if word.__contains__(x)==False:
                if weight.__contains__(x)==True:
                    p1 = round(weight[x],4)
                else:
                    p1 = 0.0
                word[x] = p1

        tweet_word[tweet_index] = [v[1],set_of_words,v[0],v[3],v[4]]  #Length of tweet, set of content words present in the tweet, tweet itself, validity score
        tweet_index+=1

    ############################### Make a List of Tweets ###########################################
    sen = list(tweet_word.keys())
    sen.sort()
    entities = list(word.keys())
    print(len(sen),len(entities))

    ################### Define the Model #############################################################

    m = Model("sol1")

    ############ First Add tweet variables ############################################################

    sen_var = []
    for i in range(0,len(sen),1):
        sen_var.append(m.addVar(vtype=GRB.BINARY, name="x%d" % (i+1)))

    ############ Add entities variables ################################################################

    con_var = []
    for i in range(0,len(entities),1):
        con_var.append(m.addVar(vtype=GRB.BINARY, name="y%d" % (i+1)))

    ########### Integrate Variables ####################################################################
    m.update()

    P = LinExpr() # Contains objective function
    C1 = LinExpr()  # Summary Length constraint
    C4 = LinExpr()  # Summary Length constraint
    C2 = [] # If a tweet is selected then the content words are also selected
    counter = -1
    for i in range(0,len(sen),1):
        P += sen_var[i] #Validity score * tweet index
        C1 += tweet_word[i+1][0] * sen_var[i]
        v = tweet_word[i+1][1] # Entities present in tweet i+1
        C = LinExpr()
        flag = 0
        for j in range(0,len(entities),1):
            if entities[j] in v:
                flag+=1
                C += con_var[j]
            if flag>0:
                counter+=1
                m.addConstr(C, GRB.GREATER_EQUAL, flag * sen_var[i], "c%d" % (counter))

    C3 = [] # If a content word is selected then at least one tweet is selected which contains this word
    for i in range(0,len(entities),1):
        P += word[entities[i]] * con_var[i]
        C = LinExpr()
        flag = 0
        for j in range(0,len(sen),1):
            v = tweet_word[j+1][1]
            if entities[i] in v:
                flag = 1
                C += sen_var[j]
            if flag==1:
                counter+=1
                m.addConstr(C,GRB.GREATER_EQUAL,con_var[i], "c%d" % (counter))
    
    counter+=1
    m.addConstr(C1,GRB.LESS_EQUAL,L, "c%d" % (counter))


    ################ Set Objective Function #################################
    m.setObjective(P, GRB.MAXIMIZE)

    ############### Set Constraints ##########################################

    fo = codecs.open(ofname,'w','utf-8')
    try:
        m.optimize()
        for v in m.getVars():
            if v.x==1:
                temp = v.varName.split('x')
                if len(temp)==2:
                    fo.write(tweet_word[int(temp[1])][4] + '\t' + tweet_word[int(temp[1])][2])
                    fo.write('\n')
    except GurobiError as e:
        print(e)
        sys.exit(0)

    fo.close()

def compute_rationale_weight_tf(word,tweet_count,PLACE):
    score = {}

    discard = []
    THR = 2
    N = tweet_count + 4.0 - 4.0
    for k,v in word.items():
        D_w = k.strip(' \t\n\r')
        if D_w not in discard:
            tf = v
            w = 1 + math.log(tf,2)
            score[k] = w
    return score

def main():
    try:
        _, ifname, placefile, ofname = sys.argv
    except Exception as e:
        print(e)
        sys.exit(0)
    compute_summary(ifname,placefile,ofname)

if __name__=='__main__':
    main()

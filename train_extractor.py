import math
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from textblob import TextBlob
from collections import Counter
from gensim.parsing.preprocessing import STOPWORDS
from spellchecker import SpellChecker
from nltk.corpus import wordnet
import textstat
import re
import language_tool

from preprocess import preprocessdata

#part of Speech Tagging
from test import main_cosine
from train_test.TFIDF import calculate_tfidf
from train_test.extractor import calculate_score


def PosTagging(essay_processed):
    """
    create a list of Pos tag and return
    :param essay_processed:
    :return: list_pos_tag
    """
    no_of_pos = {}
    list_pos_tag = []
    for i in range(0, len(essay_processed)):
        essay = TextBlob(essay_processed[i])
        no_of_pos[i] = len(essay.tags)
        list_pos_tag.append((essay.tags))
    return list_pos_tag

# Noun  Extraction
def NN_Extraction(list_of_pos_tags):
    """
    count noun in a script
    :param list_of_pos_tags:
    :return:
    """
    count_of_NN = ["NN"]
    i=0
    for f in list_of_pos_tags:
        sum =0
        for e,t in enumerate(f):
            if t[1] == 'NN':
                sum = sum + 1
        count_of_NN.append(sum)
        i = i+1
    return count_of_NN

#Proper Noun Extraction
def NNP_Extraction(list_of_pos_tags):
    """
    count proper noun in a script
    :param list_of_pos_tags:
    :return:
    """
    count_of_NNP = ["NNP"]
    i=0
    for f in list_of_pos_tags:
        sum =0
        for e,t in enumerate(f):
            if t[1] == 'NNP':
                sum = sum + 1
        count_of_NNP.append(sum)
        i = i+1
    return count_of_NNP

#VERB Extraction
def VERB_Extraction(list_of_pos_tag):
    """
    count verb in a script
    :param list_of_pos_tag:
    :return:
    """
    count_of_VERB = ["VERB"]
    i=0
    for f in list_of_pos_tag:
        sum =0
        for e,t in enumerate(f):
            if t[1] == 'VBP':
                sum = sum + 1
        count_of_VERB.append(sum)
        i = i+1
    return count_of_VERB

#ADVERB Extraction
def ADVERB_Extraction(list_of_pos_tag):
    """
    count adverb in a script
    :param list_of_pos_tag:
    :return:
    """
    count_of_RB = ["ADVERB"]
    i=0
    for f in list_of_pos_tag:
        sum =0
        for e,t in enumerate(f):
            if t[1] == 'RB':
                sum = sum + 1
        count_of_RB.append(sum)
        i = i+1
    return count_of_RB

#ADJECTIVE Extraction
def ADJECTIVE_Extraction(list_of_pos_tag):
    """
    count adjective in a script
    :param list_of_pos_tag:
    :return:
    """
    count_of_JJ = ["ADJECTIVE"]
    i=0
    for f in list_of_pos_tag:
        sum =0
        for e,t in enumerate(f):
            if t[1] == 'JJ':
                sum = sum + 1
        count_of_JJ.append(sum)
        i = i+1
    return count_of_JJ

#Determiners Extraction
def DETERMINERS_Extraction(list_of_pos_tag):
    """
    count Determiners in a script
    :param list_of_pos_tag:
    :return: count_of_DT
    """
    count_of_DT = ["DETERMINER"]
    i=0
    for f in list_of_pos_tag:
        sum =0
        for e,t in enumerate(f):
            if t[1] == 'JJ':
                sum = sum + 1
        count_of_DT.append(sum)
        i = i+1
    return count_of_DT

#average length of word
def no_of_words(essay_data):
    """
    count number of words , number of sentences in a script
    from above two parameter we calculate average wordlength in a script
    :param essay_data:
    :return: average_word_count,no_of_words,no_of_sentence
    """
    no_of_words = ["no of words"]
    no_of_sentence = ["no of sentences"]
    average_word_count =["average word count"]
    for k,v in essay_data.items():
        doc = TextBlob(str(v))
        no_of_words.append(len(doc.words))
        no_of_sentence.append(len(doc.sentences))
        len_of_word = []
        for i in range(len(doc.words)):
            len_of_word.append(len(doc.words[i]))
        average_word_count.append(round(sum(len_of_word)/len(len_of_word),2))
    return average_word_count,no_of_words,no_of_sentence

#check Spelling Mistake
def Spelling_mistake(essay):
    """
    Count number of spelling mistake in a script
    :param essay:
    :return: mispelled_list
    """
    spell = SpellChecker()
    mispelled_list = ["mistake"]
    for v in essay:
        doc = TextBlob(str(v))
        mispelled = spell.unknown(doc.words)
        mispelled_list.append(len(mispelled))
    return mispelled_list

#Flesch score
def calculate_Flesch_Score(essay_data):
    """
    Flesch score is Readability score. It is done on unpreprocessed data.
    :param essay_data:
    :return: list_flesch_score
    """
    list_flesch_score = ["FLESCH Score"]
    for v in essay_data:
        list_flesch_score.append(textstat.flesch_reading_ease(str(v)))
    return list_flesch_score

#seven test using textstat for content richness
def seven_test(processed_essay):
    """
    score which is assigned to every script in on the basis of some predifened fomulas
    These scores are known as readability score.
    flesch_score,gunning_index,kincaid_grade,liau_index,automated_readability_index,dale_readability_score,difficult_word,linsear_write
    :param processed_essay:
    :return:flesch_score,gunning_index,kincaid_grade,liau_index,automated_readability_index,dale_readability_score,difficult_word,linsear_write
    """
    flesch_score = ["FS"]
    gunning_index = ["GI"]
    kincaid_grade = ["KG"]
    liau_index = ["LI"]
    automated_readability_index = ["ARI"]
    dale_readability_score = ["DLS"]
    difficult_word = ["DW"]
    linsear_write = ["LW"]
    for v in processed_essay:
        flesch_score.append(textstat.flesch_reading_ease(str(v)))
        gunning_index.append(textstat.gunning_fog(str(v)))
        kincaid_grade.append(textstat.flesch_kincaid_grade(str(v)))
        liau_index.append(textstat.coleman_liau_index(str(v)))
        automated_readability_index.append(textstat.automated_readability_index(str(v)))
        dale_readability_score.append(textstat.dale_chall_readability_score(str(v)))
        difficult_word.append(textstat.difficult_words(str(v)))
        linsear_write.append(textstat.linsear_write_formula(str(v)))
    return flesch_score,gunning_index,kincaid_grade,liau_index,automated_readability_index,dale_readability_score,difficult_word,linsear_write

#grammer error check
def Check_grammer_error(essay_data):
    """
    count grammer error per in essay data for every ID
    :param essay_data:
    :return: grammer_error_list
    """
    grammer_error_list = []
    lang_tool = language_tool.LanguageTool("en-US")
    for v,k in essay_data.items():
        matches = lang_tool.check(str(k))
        grammer_error_list.append(matches)
    return grammer_error_list

def Clauseword(essay_data):
    """
    count no of "what","why","which","who","where" used in essay
    :param essay_data:
    :return: count_clause_word
    """
    count_clause_word = ["clause word"]
    clause_word_list = ["what","why","which","who","where"]
    for k in essay_data:
        num=0
        doc = TextBlob(str(k).lower())
        list_of_word = doc.words
        for word in list_of_word:
            if word in clause_word_list:
                num += 1
        count_clause_word.append(num)
    return count_clause_word

def clarity_list(train_essay):
    """

    :param train_essay:
    :return: list_clarity
    """
    list_clarity = ["clarity"]
    arr = ['worst','average','above_average','excellent']
    for v,k in train_essay['clarity'].items():
        if k in arr:
            list_clarity.append(arr.index(k)+1)
        else:
            list_clarity.append(0)
    return list_clarity

def coherant_list(train_essay):
    """

    :param train_essay:
    :return: list_coherent
    """
    list_coherent = ['coherent']
    arr = ['worst','average','above_average','excellent']
    for v,k in train_essay['coherent'].items():
        if k in arr:
            list_coherent.append(arr.index(k)+1)
        else:
            list_coherent.append(0)
    return list_coherent

        
def normal_score(score):
    """

    :param score:
    :return:final_score
    """
    final_Score = ['score']
    for i in score:
        if i == 'nan':
            final_Score.append(0)
        else:
            final_Score.append(round(i))
    return final_Score

def store_csv(complete_data,c):
    """
    create csv file for every essayset
    :param complete_data:
    :param c:
    :return:
    """
    my_df = pd.DataFrame(complete_data)
    my_df.T.to_csv("TrainData/datatrain{}.csv".format(c),index=False, header=False)

def DataTrain():
    """
    :param train_essay:
    :return: Create csv file for every set
    """
    train_essay = pd.read_csv("/mnt/1f2870f0-1578-4534-b33f-0817be64aade/projects/Hackerearth/incedo_nlpcadad7d/incedo_participant/train_dataset.csv")
    essay_Set = [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]
    c = 0
    for i in essay_Set:
        if i > 0 :
            c = c+1
            set_filter = (train_essay.Essayset == i)
            id_list = ['ID']
            id_list.extend(list(train_essay[set_filter]['ID']))
            average_word_length,no_of_word,no_of_sentence= no_of_words(train_essay[set_filter]['EssayText'])  #10
            processed_essay = preprocessdata(train_essay[set_filter]['EssayText'])     #preprocessed Essay
            flesch_score,gunning_index,kincaid_grade,liau_index,automated_readability_index,dale_readability_score,difficult_word,linsear_write = seven_test(processed_essay)  #18
            count_misspell = Spelling_mistake(processed_essay)     #misspell word #9
            Flesch_score_list = calculate_Flesch_Score(processed_essay)    #Flesch Score #8
            count_clause_word = Clauseword(processed_essay)  #7
            list_of_pos_tag= PosTagging(processed_essay) #6
            count_of_NN = NN_Extraction(list_of_pos_tag)   #1
            count_of_NNP = NNP_Extraction(list_of_pos_tag)   #2
            count_of_verb = VERB_Extraction(list_of_pos_tag)  #3
            count_of_adverb = ADVERB_Extraction(list_of_pos_tag)  #4
            count_of_adjective = ADJECTIVE_Extraction(list_of_pos_tag)  #5
            count_of_deteminers = DETERMINERS_Extraction(list_of_pos_tag)   #20
            clarity = clarity_list(train_essay[set_filter])   #21
            coherant = coherant_list(train_essay[set_filter])   #22
            score = calculate_score(train_essay,i)
            normalized_score = normal_score(score)
            tfidf_score = calculate_tfidf(train_essay,i)
            #count_grammer_error = Check_grammer_error(train_essay[set_filter]['EssayText'])
            complete_data = []
            list_column = [count_of_NN,count_of_NNP,count_of_verb,count_of_adverb,flesch_score,
                           count_of_adjective,count_misspell,clarity,coherant,
                           Flesch_score_list,count_clause_word,gunning_index,
                           dale_readability_score,linsear_write,
                           average_word_length,no_of_word,no_of_sentence,normalized_score]     #cosine_data not required for test set
            for i in list_column:
                complete_data.append(i)
            store_csv(complete_data,c)

if __name__ == "__main__":
    DataTrain()







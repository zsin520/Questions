#! /usr/bin/env python3
import nltk
import sys
import os
import math
import string

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Calculate IDF values across files
    files = load_files('/Users/zsin/Code/questions/corpus')
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    dictionary={}
    for foldername, subfolders, files in os.walk(directory):
        for file in files:
            path=os.path.join(foldername,file)
            upload=open(path)
            text=upload.read()
            dictionary[file]=text
    return dictionary

def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    allWords=[]
    punc=string.punctuation
    stopWords=nltk.corpus.stopwords.words('english')
    
    tokens=nltk.word_tokenize(document.lower())
    for item in tokens:
        if item not in punc and item not in stopWords: 
            allWords.append(item)
    return allWords 


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    dic={}
    words=set()
    length=len(documents)
    
    for key in documents.keys():
        for val in documents[key]:
            words.add(val)

    for word in words:
        docFreq=0
        for text in documents.values():
            if word in text:
                docFreq+=1
        idf=math.log(length/docFreq)
        dic[word]=idf

    return dic 

def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    fileVal={}

    for filename,text in files.items():
        tf_idf=0
        for word in query:
            tf_idf += text.count(word)*idfs[word]
        fileVal[filename]=tf_idf

    fileL=[]
    for i in range(n):
        topFile=(max(fileVal,key=fileVal.get))
        fileL.append(topFile)
        del fileVal[topFile]
    return fileL

def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    sentVal={}

    for sentence,text in sentences.items():
        matchingWords=query.intersection(text)

        idf=0
        for item in matchingWords:
            idf += idfs[item]

        matches=0
        for item in text:
            for word in query:
                if item == word:
                    matches+=1
        qtd=matches/len(text)
        
        sentVal[sentence]={'idf':idf, 'qtd':qtd}

    sortedDic=sorted(sentVal.items(), key=lambda item: (item[1]['idf'],item[1]['qtd']), reverse=True)
    finalDic=[sent[0] for sent in sortedDic]

    return finalDic[:n]

if __name__ == "__main__":
    main()

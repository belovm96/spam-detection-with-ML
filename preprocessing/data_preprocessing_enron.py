"""
Enron1 dataset preprocessing script

@author: Mike Belov - belovm96
"""
import pandas as pd
import nltk
import re
import string
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer 

df = pd.read_csv('spam_ham_dataset.csv')

df.drop('Unnamed: 0', axis=1, inplace = True)
df.columns = ['class', 'mail', 'label']

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
def init_helpers():
    sw = set(stopwords.words('english'))
    ps = PorterStemmer() 
    wnl = WordNetLemmatizer()
    p = string.punctuation
    not_needed = ['subject', 'ect', 'enron', 'hou']
    return sw, ps, wnl, p, not_needed
    

def lemmatize(text):
    stop_words, ps, wnl, puncts, not_needed = init_helpers()

    splitted = re.split(': | , |\r\|\n|\s', text)
    lower = [word.lower() for word in splitted]
    filtered = list(filter(lambda x: x.isdigit() != True and x not in puncts and x not in not_needed and x not in stop_words, lower))
    tagged = nltk.tag.pos_tag(filtered)
    
    wordnet_tagged = [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in tagged]
    lemmatized = [wnl.lemmatize(word, tag) for word, tag in wordnet_tagged]
    back_to_str = ' '.join(lemmatized)
    return back_to_str

def stem(text):
    stop_words, ps, wnl, puncts, not_needed = init_helpers()
    splitted = re.split(': | , |\r\|\n|\s', text)
    lower = [word.lower() for word in splitted]
    filtered = list(filter(lambda x: x.isdigit() != True and x not in puncts and x not in not_needed and x not in stop_words, lower))
    stemmed = [ps.stem(word) for word in filtered]
    back_to_str = ' '.join(stemmed)
    return back_to_str

df['lemmatized'] = df.mail.apply(lambda x: lemmatize(x))
df['stemmed'] = df.mail.apply(lambda x: stem(x))
df['class'].loc[df['class'] == 'ham'] = 'real'
df.to_csv('spam_dataset_preprocessed.csv')


import re
import string
import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# import sys
# sys.path.insert(0, '/home/prajendiran/riskfusion')
# from scripts.common.utils import get_text_from_html

from bs4 import BeautifulSoup as Soup

def get_text_from_html(html_str):
    soup = Soup(html_str, features="html.parser")
    text = soup.get_text(strip=True)
    return text


# as per recommendation from @freylis, compile once only
CLEANR = re.compile('<.*?>') 

def clean_html(raw_html):
    cleantext = re.sub(CLEANR, '', raw_html)
    return cleantext


def cleanse_text(text):
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)
    # text = re.sub(r'\[[0-9]*\]',' ', text)
    text = re.sub(r'[0-9]',' ', text)
    text = text.replace('\n', ' ').replace('\r', ' ').strip().lower()
    text = re.sub(r'\s+', ' ', text) #\s matches any whitespace, \s+ matches multiple whitespace, \S matches non-whitespace 
    
    return text

def remove_stopwords(text):
    en_stopwords = stopwords.words('english')
    required_stopwords = ['of', 'not', 'no', 'further']
    for rsw in required_stopwords:
        en_stopwords.remove(rsw)
    unwanted_words = [
        'solace', 'pulse', 'netcore', 'mq', 'rtm', 'qid',
        'activematrix', 'confluent', 'pubsub', 'businessevents', 'asa', 'window', 
        'richfaces', 'therefore', 
    ]
    en_stopwords.extend(unwanted_words)
    non_stopwords = [i for i in text.split() if i not in en_stopwords]
    months = [
        'january', 'february', 'march', 'april', 'june', 'july', 'august', 'september', 'october', 'november', 'december',
        'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
    ]
    vendors = [
        'adobe', 'apache', 'atlassian', 'jetbrains', 'jira', 'businessworks', 'jre', 'confluence', 'mozilla', 
        'tibco', 'joomla', 'teamcity', 'realplayer', 'vermillion', 'microsoft', 'paloalto', 'dhcp', 'talos',
        'rabbitmq', 'dotnet', 'freebsd', 'ubuntu', 'opensuse', 'euleros', 'fedora'
    ]
    output = []
    for word in non_stopwords:
        if word in months:
            output.append('month')
        elif word in vendors:
            output.append('vendor')
        else:
            output.append(word)

    return ' '.join(output)

def get_wordnet_pos(tag):
    """
    This is a helper function to map NTLK part of speech tags.
    Full list is available here: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
    """
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

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

def lemmatize(text):
    word_pos_tags = nltk.pos_tag(word_tokenize(text))
    lemmatized_words = [
        lemmatizer.lemmatize(word, get_wordnet_pos(pos)) 
        for word, pos in word_pos_tags
    ]
    return ' '.join(lemmatized_words)

def pre_process(text):
    text = get_text_from_html(text.replace('<', '. <'))
    text = cleanse_text(text)
    text = remove_stopwords(text)
    text = lemmatize(text)
    return text

def cleanse_title(title):
    if 'EOL/' in title:
        title = title.replace('EOL/', '')
    return title

def show_metrics(y_actual, y_predicted):
    conf_mat = confusion_matrix(y_actual, y_predicted)
    tn, fp, fn, tp = conf_mat.flatten()
    print('True Positive : {:>7}\t\tFalse Positive : {:>7}\nTrue Negative : {:>7}\t\tFalse Negative : {:>7}'.format(tp,fp,tn,fn))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    acc = (tp + tn) / np.sum(conf_mat)
    f1 = 2 * precision * recall / (precision + recall)
    # f1_func = f1_score(y_actual, y_predicted)
    print(
        'Precision : {0:.4}\nRecall    : {1:.4}\nAccuracy  : {2:.4%}\nF1 score  : {3:.4}'.format(
            precision, recall, acc, f1
        )
    )

    group_names = ['True -ve', 'False +ve', 'False -ve', 'True +ve']
    group_counts = [tn, fp, fn, tp]
    group_percentages = ['{0:.2%}'.format(i) for i in conf_mat.flatten()/np.sum(conf_mat)]
    labels = [
        f'{n}\n{c}\n{p}' for n, c, p in zip(
            group_names, group_counts, group_percentages
        )
    ]
    labels = np.asarray(labels).reshape(2, 2)

    ax = sns.heatmap(conf_mat, annot=labels, fmt='', cmap='Greens', linewidths=1, linecolor='Black')
    for _, spine in ax.spines.items():
        spine.set_visible(True)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values')
    ax.xaxis.set_ticklabels(['False', 'True'])
    ax.yaxis.set_ticklabels(['False', 'True'])
    plt.show()

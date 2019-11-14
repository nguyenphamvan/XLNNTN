import os

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DOCUMENT = os.path.join(DIR_PATH, 'document.txt')
STOP_WORDS = os.path.join(DIR_PATH, 'vietnamese_stopwords.txt')
SPECIAL_CHARACTER = '0123456789%@$.,=+-!–;/()*"&^:”“#|\n\t\''
DICTIONARY_PATH = 'dictionary.txt'
import re
from nltk.util import ngrams
from textrank.preprocessing import FileReader, NLP
from textrank import settings
# s = "Natural-language processing (NLP) is an area of computer science " \
#     "and artificial intelligence concerned with the interactions " \
#     "between computers and human (natural) languages."

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def bigram(doc):
    tokens = [token.strip(settings.SPECIAL_CHARACTER).lower() for token in doc.split()]
    tokens = [token for token in tokens if token != '']
    return list(ngrams(tokens, 2))

title, reference_summary = FileReader('reference_Summary.txt').read_file()
title_sum ,system_summary = FileReader('summary.txt').read_file()
output1 = bigram(reference_summary)
output2 = bigram(system_summary)
output3 = intersection(output2, output1)
#
print(output1)
print(output3)
#
print(len(output3)/len(output1))
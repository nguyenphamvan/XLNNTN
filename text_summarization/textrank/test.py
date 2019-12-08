import re
from nltk.util import ngrams
from text_summarization.textrank.readfile import FileReader
from text_summarization.textrank.preprocessing import NLP
from text_summarization.textrank import settings
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

title, reference_summary = FileReader('/home/nguyenpham/PycharmProjects/XLNNTN/text_summarization/textrank/model/00000_model.txt').read_file()
title_sum ,system_summary = FileReader('/home/nguyenpham/PycharmProjects/XLNNTN/text_summarization/textrank/system/00000_system.txt').read_file()
reference_summary = reference_summary.strip()
system_summary = system_summary.strip()
output1 = bigram(reference_summary)
output2 = bigram(system_summary)
output3 = intersection(output2, output1)
#
print(output1)
print(output3)
#
print(len(output3)/len(output2))







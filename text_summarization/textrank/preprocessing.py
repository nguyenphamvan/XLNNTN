from pyvi import ViTokenizer
from textrank.readfile import FileReader
from textrank import settings
import nltk

class NLP(object):
    def __init__(self, doc = None):
        self.doc = doc
        self.__set_stopwords()

    def __set_stopwords(self):
        self.stopwords = FileReader(settings.STOP_WORDS).read_stopwords()

    def sentence_segmentation(self):
        sents_tokened = nltk.sent_tokenize(self.doc)
        return [sent_tokened for sent_tokened in sents_tokened if (sent_tokened not in settings.SPECIAL_CHARACTER and sent_tokened !='.')]

    def word_segmentation(self, sent):
        return ViTokenizer.tokenize(sent)

    def split_words(self, sent):
        sent = self.word_segmentation(sent)
        try:
            step_1 = [x.strip(settings.SPECIAL_CHARACTER).lower() for x in sent.split()]
            return [x for x in step_1 if x != '']
        except TypeError:
            return []

    def get_words_feature(self,sent):
        split_words = self.split_words(sent)
        words = []
        for word in split_words:
            if word not in self.stopwords:
                words.append(word)
        return words

    def doc_parsed(self):
        doc_parsed = []
        sentences = self.sentence_segmentation()
        for sent in sentences:
            sent_parsed = ' '.join(self.get_words_feature(sent))+"."
            if sent_parsed != '.':
                doc_parsed.append(sent_parsed)
        return doc_parsed

#Trường hợp câu quá ngắn, và tất cả các từ đều thuộc stopwords sẽ bị loại bỏ
title, doc = FileReader(settings.DOCUMENT).read_file()
sents = NLP(doc).sentence_segmentation()
sentences = NLP(doc).doc_parsed()
print(sents)
print(sentences)
for i,sent in enumerate(sentences):
    if sent == '':
        print(i)
        print(sents[i])









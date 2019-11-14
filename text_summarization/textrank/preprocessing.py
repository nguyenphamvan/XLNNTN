from pyvi import ViTokenizer
import settings
import nltk

class FileReader(object):
    def __init__(self, file_path):
        self.file_path = file_path

    def read_file(self):
        rows = []
        with open(self.file_path, 'r', encoding='utf-8') as f:
            file = f.readlines()
        for i, row in enumerate(file):
            row = row.replace('\n', '')
            if i == 0:
                title = row
                pass
            else:
                rows.append(row)
        doc_content = ' '.join(rows)
        return title, doc_content

    def read_stopwords(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            stopwords = list(set([w.strip().replace(' ', '_') for w in f.readlines()]))
        return stopwords

class NLP(object):
    def __init__(self, doc = None):
        self.doc = doc
        self.__set_stopwords()

    def __set_stopwords(self):
        self.stopwords = FileReader(settings.STOP_WORDS).read_stopwords()

    def sentence_segmentation(self):
        sent_tokened = nltk.sent_tokenize(self.doc)
        return sent_tokened

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
            sent_parsed = ' '.join(self.get_words_feature(sent)) + '.'
            doc_parsed.append(sent_parsed)
        return doc_parsed

# title, doc = FileReader(settings.DOCUMENT).read_file()
# print(title)
# print(doc)








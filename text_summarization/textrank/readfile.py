import nltk
from textrank import settings
import os

class FileReader(object):
    def __init__(self, file_path):
        self.file_path = file_path

    def read_file(self):
        rows = []
        with open(self.file_path, 'r', encoding='utf-8',errors='ignore') as f:
            file = f.readlines()
        for i, row in enumerate(file):
            if row != '\n':
                row = row.replace('\n', '')
                if row.endswith('.') == False:
                    row = row+'.'
                if i == 0:
                    title = row
                rows.append(row)
        doc_content = ' '.join(rows)
        return title, doc_content

    def read_stopwords(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            stopwords = list(set([w.strip().replace(' ', '_') for w in f.readlines()]))
        return stopwords


# doc = FileReader(os.path.join(settings.DIR_PATH, 'Plaintext','Van Hoa','VH02.txt')).read_file()
# print(doc)
# sentences = nltk.sent_tokenize(doc)
# print(sentences)

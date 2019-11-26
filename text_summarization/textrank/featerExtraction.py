from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

class FeaterExtraction(object):
    def __init__(self, doc):
        self.doc = doc

    def tf_idf(self):
        tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0.0)
        tfidf_matrix = tf.fit_transform(self.doc)
        return tfidf_matrix


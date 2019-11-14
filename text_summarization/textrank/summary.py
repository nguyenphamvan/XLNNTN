from nltk.cluster.util import cosine_distance
import nltk
import numpy as np
from operator import itemgetter
from textrank import featerExtraction, preprocessing, settings

class TextRank(object):

    def sentence_similarity(self,vector1, vector2):
        return 1 - cosine_distance(vector1, vector2)

    def build_similarity_matrix(self,sentences_matrix):
        # tạo một ma trận tương đồng rỗng toàn 0
        # kích thước là (số câu trong văn bản x số câu trong văn bản)
        # phần tử hang i, cột j lưu giá trị độ tương đồng của câu i và câu j
        S = np.zeros((sentences_matrix.shape[0], sentences_matrix.shape[0]))

        for idx1 in range(sentences_matrix.shape[0]):
            for idx2 in range(sentences_matrix.shape[0]):
                if idx1 == idx2:
                    #các phần tử trên đường chéo vẫn bằng 0
                    continue

                S[idx1][idx2] = self.sentence_similarity(sentences_matrix[idx1], sentences_matrix[idx2])

        # chuẩn hóa ma trận theo hàng
        for idx in range(len(S)):
            S[idx] /= S[idx].sum()

        return S

    def pagerank(self,A, eps=0.001, d=0.85):
        P = np.ones(len(A)) / len(A)
        while True:
            new_p = np.ones(len(A)) * (1 - d) / len(A) + d * A.T.dot(P)
            delta = abs(new_p - P).sum()
            if delta <= eps:
                return new_p
            P = new_p


    def textrank(self,sentences, tf_idf_matrix):
        """
        sentences = a list of sentences [[w11, w12, ...], [w21, w22, ...], ...]
        top_n = how may sentences the summary should contain
        stopwords = a list of stopwords
        """
        top_n = len(sentences) // 4
        S = self.build_similarity_matrix(tf_idf_matrix)
        sentence_ranks = self.pagerank(S)

        # Sort the sentence ranks
        ranked_sentence_indexes = [item[0] for item in sorted(enumerate(sentence_ranks), key=lambda item: -item[1])]
        selected_sentences = sorted(ranked_sentence_indexes[:top_n])
        summary = itemgetter(*selected_sentences)(sentences)
        return summary


    def print_result(self):
        title, doc = preprocessing.FileReader(settings.DOCUMENT).read_file()
        sentences = nltk.sent_tokenize(doc)
        doc_parsed = preprocessing.NLP(doc).doc_parsed()
        tf_idf_matrix = featerExtraction.FeaterExtraction(doc_parsed).tf_idf()
        tf_idf_matrix = tf_idf_matrix.toarray()

        summary = self.textrank(sentences, tf_idf_matrix)
        summary = ' '.join(summary)
        summary = title+'\n'+summary
        file = open('summary.txt', 'w', encoding='utf-8')
        file.write(summary)

        print('use tf-idf model')
        for idx, sentence in enumerate(self.textrank(sentences, tf_idf_matrix)):
            print("%s. %s" % ((idx + 1), sentence))




a = TextRank()
a.print_result()
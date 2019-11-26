from nltk.cluster.util import cosine_distance
import nltk
import numpy as np
from operator import itemgetter
from textrank import featerExtraction, preprocessing, settings, readfile
import os

class TextRank(object):
    def sentence_similarity(self,vector1, vector2):
        """
        :param vector1: vector sau khi encoder dạng tf-idf
        :param vector2: vector sau khi encoder dạng tf-idf
        :return: khoảng cách giữa 2 vector
        """
        return 1 - cosine_distance(vector1, vector2)

    def build_similarity_matrix(self,sentences_matrix):
        """
        :param sentences_matrix: ma trận encoder các câu dưới dạng vector
        :return: ma trận tương đồng câu
        """
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
            if S[idx].sum() == 0:
                S[idx] = np.add(S[idx], np.ones(len(S))/(len(S)*10))

            S[idx] /= S[idx].sum()
        return S

    def pagerank(self,A, eps=0.001, d=0.85):
        """
        :param A: input là ma trận tương đồng của các câu
        :param eps:
        :param d:
        :return: một mảng điểm xếp hạng các câu theo thứ tự
        """
        P = np.ones(len(A)) / len(A)
        while True:
            new_p = np.ones(len(A)) * (1 - d) / len(A) + d * A.T.dot(P)
            delta = abs(new_p - P).sum()
            if delta <= eps:
                return new_p
            P = new_p


    def textrank(self,sentences, tf_idf_matrix):
        """
        :param sentences: một danh sách các câu [[w11, w12, ...], [w21, w22, ...], ...]
        :param tf_idf_matrix: ma trận mã hóa các câu dạng vector theo tf-idf
        :return: một tuple chứa các câu được xếp hạng giảm dần theo thứ tự
        """
        top_n = len(sentences) // 4 #số câu tóm tắt muốn đưa vào văn bản tóm tắt
        S = self.build_similarity_matrix(tf_idf_matrix)
        sentence_ranks = self.pagerank(S)

        # Sắp xếp thứ hạng các câu
        ranked_sentence_indexes = [item[0] for item in sorted(enumerate(sentence_ranks), key=lambda item: -item[1])]
        selected_sentences = sorted(ranked_sentence_indexes[:top_n])
        summary = itemgetter(*selected_sentences)(sentences)
        return summary

    def summary(self, file_path):
        doc = readfile.FileReader(settings.DOCUMENT).read_file()
        sentences = nltk.sent_tokenize(doc)
        doc_parsed = preprocessing.NLP(doc).doc_parsed()
        tf_idf_matrix = featerExtraction.FeaterExtraction(doc_parsed).tf_idf().toarray()

        summary = self.textrank(sentences, tf_idf_matrix)
        summary = ' '.join(summary)
        return summary

    def main(self):
        dirs = os.listdir(settings.PLAINTEXT_PATH)
        for folder in dirs:
            file_paths = os.listdir(os.path.join(settings.PLAINTEXT_PATH,folder))
            for file_path in file_paths:
                print(folder + " - "+file_path)
                title,doc = readfile.FileReader(os.path.join(settings.PLAINTEXT_PATH,folder,file_path)).read_file()
                sentences = preprocessing.NLP(doc).sentence_segmentation()
                doc_parsed = preprocessing.NLP(doc).doc_parsed()
                tf_idf_matrix = featerExtraction.FeaterExtraction(doc_parsed).tf_idf().toarray()
                summary = self.textrank(sentences, tf_idf_matrix)
                result = ""
                for _, sent_sum in enumerate(summary):
                    result = result + sent_sum + "\n"
                file = open(os.path.join(settings.SUMMARY_SYSTEM_PATH,folder,file_path), 'w', encoding='utf-8')
                file.write(result)
                file.close()

        # title, doc = readfile.FileReader(settings.DOCUMENT).read_file()
        # sentences = preprocessing.NLP(doc).sentence_segmentation()
        # doc_parsed = preprocessing.NLP(doc).doc_parsed()
        # tf_idf_matrix = featerExtraction.FeaterExtraction(doc_parsed).tf_idf().toarray()
        # print(doc_parsed)
        # for i, tf_idf_row in enumerate(tf_idf_matrix):
        #     if np.sum(tf_idf_row)==0:
        #         print(i)
        #
        # summary = self.textrank(sentences, tf_idf_matrix)
        # result = title + "\n"
        # for _, sent_sum in enumerate(summary):
        #     result = result + sent_sum + "\n"
        # # file = open(os.path.join(settings.SUMMARY_SYSTEM_PATH, 'boKHCN', 'summary.txt'), 'w', encoding='utf-8')
        # file = open('summary.txt', 'w', encoding='utf-8')
        # file.write(result)
        #
        # print('use tf-idf model')
        # for idx, sentence in enumerate(self.textrank(sentences, tf_idf_matrix)):
        #     print("%s. %s" % ((idx + 1), sentence))

if __name__ == '__main__':
    TextRank().main()

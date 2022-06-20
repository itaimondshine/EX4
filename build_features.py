import spacy
FILE_DATA = 'Corpus.TRAIN.txt'
spacyTagger = spacy.load('en_core_web_sm')


class Data:
    def __init__(self):
        self.corpus = self.createCorpus()


    def createData(self):
        for id, sent in self.corpus.items():
            chunk_sent = spacyTagger(sent)
            print(chunk_sent)


    def createCorpus(self):
        sentences = {}
        file_data = open(FILE_DATA).read()
        for s in file_data.split('\n'):
            sent_array = s.strip().split('\t')
            sent_id = sent_array[0]
            sent = ''.join(sent_array[1:])

            sentences[sent_id] = sent
        return sentences



if __name__ == '__main__':
    d = Data()
    d.createData()











#
# class FeatureBuilding:
#     def __init__(self):
#         pass
#
#
#
#
#     def build_features(self, first_chunk, second_chunk, sentence):
#         word_before_entitiy_1 = word_before_entity_1()
#         word_before_entitiy_2 =
#         between_entitiy_bow =
#
#         # NER types
#         m1_type = get_ner(first_chunk)
#         m2_type = get_ner(second_chunk)

# Itai Mondshine 207814724
# Itay Lotan 308453935

import codecs
from collections import defaultdict

import spacy

from config import SPACY_MODEL_NAME, MODEL_PREDICTED_LABELS, NO_RELATION_TAG
from data_classes import NLPSentenceWordData, NLPChunkData

nlp = spacy.load(SPACY_MODEL_NAME)
ROOT_IDX = 0


class NLPDataParser:
    @classmethod
    def parse(cls, corpus_file, annotations_file=None):
        corpus = cls._read_corpus(corpus_file)
        annotations = cls.read_annotations(annotations_file) if annotations_file else None

        dataset = []
        for sent_id, sentence in corpus.items():
            dataset.extend(cls._parse_sentence(sent_id, sentence, annotations))

        return dataset

    @staticmethod
    def _clean_raw_text(text):
        return text.rstrip(".")

    @classmethod
    def _parse_sentence(cls, sent_id, sentence, annotations):
        parsed_sentence = nlp(sentence)

        raw_sentence, sentence_data, sentence_index_to_word_index = \
            cls._parse_sentence_nlp_data(parsed_sentence)
        chunks_data = cls._convert_to_chunks_data(parsed_sentence, sent_id)

        for chunk_data in chunks_data:
            cls._add_global_sentence_data(chunk_data, raw_sentence, sentence_data, sentence_index_to_word_index)

        annotations = annotations or defaultdict(list)
        for parsed_ne1 in chunks_data:
            for parsed_ne2 in chunks_data:
                if parsed_ne1.text != parsed_ne2.text:
                    arg1 = parsed_ne1
                    arg2 = parsed_ne2

                    found_any_relation = False
                    for anno_relation_type, anno_chunk1_text, anno_chunk2_text in annotations[sent_id]:
                        if arg1.text == anno_chunk1_text and arg2.text == anno_chunk2_text and anno_relation_type in MODEL_PREDICTED_LABELS:
                            yield ((arg1, arg2, (sentence_data)), anno_relation_type)
                            found_any_relation = True

                    if not found_any_relation:
                        yield ((arg1, arg2, (sentence_data)), NO_RELATION_TAG)

    @classmethod
    def _add_global_sentence_data(cls, chunk_data, raw_sentence, sentence_data, sentence_index_to_word_index):
        firstSentenceIndex = raw_sentence.find(chunk_data.originalText)
        firstWordIndex = sentence_index_to_word_index[
            firstSentenceIndex] if firstSentenceIndex in sentence_index_to_word_index else 0
        lastSentenceIndex = firstSentenceIndex + len(chunk_data.originalText) + 1
        lastWordIndex = sentence_index_to_word_index[lastSentenceIndex] - 1 \
            if lastSentenceIndex in sentence_index_to_word_index else 0
        depIndex = raw_sentence.find(chunk_data.rootDep)
        depWordIndex = sentence_index_to_word_index[depIndex] if depIndex in sentence_index_to_word_index else 0
        headIndex = raw_sentence.find(chunk_data.rootHead)
        headWordIndex = sentence_index_to_word_index[headIndex] if headIndex in sentence_index_to_word_index else 0
        chunk_data.firstWordIndex = firstWordIndex
        chunk_data.lastWordIndex = lastWordIndex
        chunk_data.headWordTag = sentence_data[headWordIndex].tag
        chunk_data.depWordIndex = depWordIndex

    @classmethod
    def _convert_to_chunks_data(cls, parsed_sentence, sent_id):
        text_to_chunk_data = {}
        for entity in parsed_sentence.ents:
            cleanText = cls._clean_raw_text(entity.text)
            chunk_data = NLPChunkData()
            chunk_data.text = cleanText
            chunk_data.originalText = entity.text
            chunk_data.entType = entity.root.ent_type_
            chunk_data.rootText = entity.root.text
            chunk_data.rootDep = entity.root.dep_
            chunk_data.rootHead = entity.root.head.text
            chunk_data.id = sent_id
            text_to_chunk_data[cleanText] = chunk_data

        for chunk in parsed_sentence.noun_chunks:
            cleanText = cls._clean_raw_text(chunk.text)
            if cleanText not in text_to_chunk_data:
                chunk_data = NLPChunkData()
                chunk_data.text = cleanText
                chunk_data.originalText = chunk.text
                chunk_data.entType = 'UNKNOWN'
                chunk_data.rootText = chunk.root.text
                chunk_data.rootDep = chunk.root.dep_
                chunk_data.rootHead = chunk.root.head.text
                chunk_data.id = sent_id
                text_to_chunk_data[cleanText] = chunk_data

        chunks_data = text_to_chunk_data.values()
        return chunks_data

    @classmethod
    def _parse_sentence_nlp_data(cls, parsed_sentence):
        sentence_data = []
        raw_sentence = ""
        index_in_sent = 0
        sentence_index_to_word_index = {}
        for i, word in enumerate(parsed_sentence):
            head_id = word.head.i + 1  # we want ids to be 1 based
            if word == word.head:  # and the ROOT to be 0.
                assert (word.dep_ == "ROOT"), word.dep_
                head_id = ROOT_IDX

            word_data = NLPSentenceWordData()
            word_data.id = word.i + 1
            word_data.word = word.text
            word_data.lemma = word.lemma_
            word_data.pos = word.pos_
            word_data.tag = word.tag_
            word_data.parent = head_id
            word_data.dependency = word.dep_
            word_data.bio = word.ent_iob_
            word_data.ner = word.ent_type_
            sentence_data.append(word_data)

            raw_sentence += " " + word.text
            sentence_index_to_word_index[index_in_sent + 1] = i
            index_in_sent += 1 + len(word.text)
        return raw_sentence, sentence_data, sentence_index_to_word_index

    @classmethod
    def read_annotations(cls, annotations_file):
        with open(annotations_file) as f:
            annotations_data = {
                row for row in f.read().split('\n')
                if row != ''
            }

        annotations = defaultdict(list)
        for annotation in annotations_data:
            id_, chunk1, connection_type, chunk2, _ = annotation.split('\t')
            chunk1 = cls._clean_raw_text(chunk1)
            chunk2 = cls._clean_raw_text(chunk2)
            if connection_type in MODEL_PREDICTED_LABELS:
                annotations[id_].append((connection_type, chunk1, chunk2))
        return annotations

    @staticmethod
    def _read_corpus(corpus_file):
        sentences = {}
        for line in codecs.open(corpus_file, encoding="utf8"):
            sent_id, sent = line.strip().split("\t")
            sent = sent.replace("-LRB-", "(").replace("-RRB-", ")")
            sentences[sent_id] = sent
        return sentences

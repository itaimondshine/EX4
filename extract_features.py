import numpy as np

from scipy.sparse import csr_matrix
from external_data_sources import COUNTRIES, CITIES

START_MARK = "#S#"
END_MARK = "#E#"

PARENT_MARK = "#P#"
CHILD_MARK = "#C#"
ROOT_MARK = "#R#"


def extract_features(dataset, feature_name_to_id, allow_map_new_features):
    feature_rows = []
    tags = []
    for ((chunk1, chunk2, sentence), tag) in dataset:
        features_row = []
        for feature in create_features(chunk1, chunk2, sentence):
            feature_is_known = feature in feature_name_to_id
            if not feature_is_known and allow_map_new_features:
                # todo - refactor this method, its easy to recognise again
                feature_name_to_id[feature] = len(feature_name_to_id)

            if feature in feature_name_to_id:
                features_row.append(feature_name_to_id[feature])

        feature_rows.append(features_row)
        tags.append(tag)

    X = csr_matrix([
        convert_dense_to_sparse_array(dense, feature_name_to_id)
        for dense in feature_rows]
    )
    Y = np.array(tags)
    return X, Y


def create_features(chunk1, chunk2, sentence):
    return [
        _chunk_entity_type_f(chunk1, 1),
        _chunk_entity_type_f(chunk2, 2),
        _chunk_text_f(chunk1),
        _chunk_text_f(chunk2),
        _chunks_concated_entity_types_f(chunk1, chunk2),
        _is_location_f(chunk1, 1),
        _is_location_f(chunk2, 2),
        _chunk_text_before_f(1, chunk1, sentence),
        _chunk_text_after_f(2, chunk2, sentence),
        _forward_tag_f(chunk1, sentence, 1),
        _chunk_head_f(chunk1, 1),
        _chunk_head_f(chunk2, 2),
        *_bag_of_words_f(chunk1, chunk2, sentence),
        *_calc_dependency_tag_features_list(chunk1, chunk2, sentence),
        *_calc_dependency_word_list_features_list(chunk1, chunk2, sentence),
        *_calc_dependency_type_list_features_list(chunk1, chunk2, sentence),
    ]


# features implementation
def _chunk_text_f(chunk):
    return f"WordsInChunk({chunk.text})"


def _chunk_text_before_f(chunk_id, chunk, sentence):
    return f"WordBeforeChunk{chunk_id}({_get_prev_word(chunk, sentence)})"


def _chunk_text_after_f(chunk_id, chunk, sentence):
    return f"WordAfterChunk{chunk_id}({_get_next_word(chunk, sentence)})"


def _chunk_entity_type_f(chunk, chunk_id):
    return f"Type{chunk_id}({chunk.entType})"


def _chunks_concated_entity_types_f(chunk1, chunk2):
    return f"TypeConcat({chunk1.entType + chunk2.entType})"


def _dependency_tag_f(chunk_id, sentence):
    tag = sentence[chunk_id - 1].tag
    return f"DependencyTag1({tag})"


def _dependency_type_f(direction, chunk_id, sentence):
    tag = sent_word_data(chunk_id, sentence).dependency
    return f"DependencyType{direction}({tag})"


def to_string_dependency_word(chunk_id, sentence):
    tag = sent_word_data(chunk_id, sentence).lemma
    return f"DependencyWord1({tag})"


def _forward_tag_f(chunk, sentence, chunk_id):
    return f"ForwardTag{chunk_id}({get_forward_tag(chunk, sentence)})"


def _chunk_head_f(chunk, chunk_id):
    return f"HeadTag{chunk_id}({chunk.headWordTag})"


def _bag_of_words_f(first_chunk, second_chunk, sentence):
    first, second = get_first_and_second_chunk(first_chunk, second_chunk)
    between_words = sentence[first.lastWordIndex:second.firstWordIndex - 1]

    return [f"BagOfWords{word.lemma}" for word in between_words]


def _is_location_f(chunk, chunk_id):
    text = chunk.text.lower()
    return f"IsLocation{chunk_id}({'T' if text in COUNTRIES or text in CITIES else 'F'})"


# helpers
def get_first_and_second_chunk(first_chunk, second_chunk):
    if first_chunk.firstWordIndex < second_chunk.firstWordIndex:
        return first_chunk, second_chunk
    return second_chunk, first_chunk


def get_forward_tag(chunk, sentence):
    id_first_word_in_chunk = chunk.lastWordIndex
    if id_first_word_in_chunk < len(sentence):
        return sentence[id_first_word_in_chunk].pos
    return END_MARK


def find_dependency_route(chunk, sentence):
    firstWord = sentence[chunk.firstWordIndex]
    parent = firstWord.parent
    current_id = firstWord.id
    while chunk.firstWordIndex <= parent - 1 <= chunk.lastWordIndex:
        current_id = parent
        parent = sentence[parent - 1].parent
    path = [current_id]
    while True:
        path.append(parent)
        if parent == 0:
            break
        parent = sentence[parent - 1].parent
    return path


def dispose_overlapping(first_route, second_route):
    overlapping = -1
    while overlapping > -len(first_route) and overlapping > -len(second_route) and first_route[overlapping] == \
            second_route[overlapping]:
        overlapping -= 1
    if overlapping == -1:
        return first_route, second_route
    return first_route[0:overlapping + 1], second_route[0:overlapping + 1]


def find_dependency_graph(first_chunk, second_chunk, sentence):
    first, second = find_dependency_routes(first_chunk, second_chunk, sentence)
    return first + list(reversed(second))


def find_dependency_routes(first_chunk, second_chunk, sentence):
    first_route = find_dependency_route(first_chunk, sentence)
    second_route = find_dependency_route(second_chunk, sentence)
    first, second = dispose_overlapping(first_route, second_route)
    return first, second


def convert_dense_to_sparse_array(dense, feature_name_to_id):
    sparse = np.zeros(len(feature_name_to_id))
    for i in dense:
        sparse[i] = 1
    return sparse


def _get_prev_word(chunk, sentence):
    id_first_word_in_chunk = chunk.firstWordIndex
    if id_first_word_in_chunk - 2 >= 0:
        return sentence[id_first_word_in_chunk - 2].word
    return START_MARK


def _get_next_word(chunk, sentence):
    id_last_word_in_chunk = chunk.lastWordIndex
    if id_last_word_in_chunk < len(sentence):
        return sentence[id_last_word_in_chunk].word
    return START_MARK


def _calc_dependency_tag_features_list(first_chunk, second_chunk, sentence):
    graph = find_dependency_graph(first_chunk, second_chunk, sentence)
    all_dependency_tags = []
    i = 0
    while graph[i] != graph[i + 1]:
        all_dependency_tags.append(_dependency_tag_f(graph[i], sentence))
        i += 1
        if i + 1 >= len(graph):
            return all_dependency_tags
    i += 1
    all_dependency_tags.append(_dependency_tag_f(graph[i], sentence))
    i += 1
    while i < len(graph):
        all_dependency_tags.append(_dependency_tag_f(graph[i], sentence))
        i += 1
    return all_dependency_tags


def _calc_dependency_word_list_features_list(first_chunk, second_chunk, sentence):
    graph = find_dependency_graph(first_chunk, second_chunk, sentence)
    all_dependency_tags = []
    i = 0
    while graph[i] != graph[i + 1]:
        all_dependency_tags.append(to_string_dependency_word(graph[i], sentence))
        i += 1
        if i + 1 >= len(graph):
            return all_dependency_tags
    i += 1
    all_dependency_tags.append(to_string_dependency_word(graph[i], sentence))
    i += 1
    while i < len(graph):
        all_dependency_tags.append(to_string_dependency_word(graph[i], sentence))
        i += 1
    return all_dependency_tags


def _calc_dependency_type_list_features_list(first_chunk, second_chunk, sentence):
    graph = find_dependency_graph(first_chunk, second_chunk, sentence)
    all_dependency_tags = []
    i = 0
    while graph[i] != graph[i + 1]:
        all_dependency_tags.append(_dependency_type_f(PARENT_MARK, graph[i], sentence))
        i += 1
        if i + 1 >= len(graph):
            return all_dependency_tags
    i += 1
    all_dependency_tags.append(_dependency_type_f(ROOT_MARK, graph[i], sentence))
    i += 1
    while i < len(graph):
        all_dependency_tags.append(_dependency_type_f(CHILD_MARK, graph[i], sentence))
        i += 1
    return all_dependency_tags


def sent_word_data(chunk_id, sentence):
    return sentence[chunk_id - 1]

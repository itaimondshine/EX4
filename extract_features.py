import numpy as np

from scipy.sparse import csr_matrix
from external_data_sources import COUNTRIES, CITIES

START_MARK = "#S#"
END_MARK = "#E#"

PARENT_MARK = "#P#"
CHILD_MARK = "#C#"
ROOT_MARK = "#R#"


def extract_features_for_train(train_dataset):
    feature_name_to_idx = dict()
    train_features, train_tags = _map_and_extract_features(
        train_dataset, feature_name_to_idx, allow_map_new_features=True)
    return train_features, train_tags, feature_name_to_idx


def extract_features_for_predict(dataset, feature_name_to_idx):
    features, *_ = _map_and_extract_features(
        dataset, feature_name_to_idx, allow_map_new_features=False)
    return features


def _map_and_extract_features(dataset, feature_name_to_idx, allow_map_new_features):
    positive_feature_rows = []
    actual_tags = []

    for ((chunk1, chunk2, sentence), actual_tag) in dataset:
        positive_feature_ids_row = []
        positive_features = create_features(chunk1, chunk2, sentence)
        for positive_feature in positive_features:
            feature_is_known = positive_feature in feature_name_to_idx
            if feature_is_known:
                feature_id = feature_name_to_idx[positive_feature]
                positive_feature_ids_row.append(feature_id)
            elif allow_map_new_features:
                new_feature_id = len(feature_name_to_idx)
                feature_name_to_idx[positive_feature] = new_feature_id
                positive_feature_ids_row.append(new_feature_id)

        positive_feature_rows.append(positive_feature_ids_row)
        actual_tags.append(actual_tag)

    X = to_sparse_matrix(positive_feature_rows, feature_name_to_idx)
    Y = np.array(actual_tags)
    return X, Y


def to_sparse_matrix(positive_feature_rows, feature_name_to_idx):
    def _positive_features_to_sparse_array(dense, feature_name_to_id):
        sparse = np.zeros(len(feature_name_to_id))
        for i in dense:
            sparse[i] = 1
        return sparse

    return csr_matrix([
        _positive_features_to_sparse_array(positive_feature_ids, feature_name_to_idx)
        for positive_feature_ids in positive_feature_rows
    ])


def create_features(chunk1, chunk2, sentence):
    positive_features = [
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
    return positive_features


# features implementation
def _chunk_text_f(chunk):
    return f'1_wic({chunk.text})'


def _chunk_text_before_f(chunk_id, chunk, sentence):
    return f'2_wbc{chunk_id}({_get_prev_word(chunk, sentence)})'


def _chunk_text_after_f(chunk_id, chunk, sentence):
    return f'3_wac{chunk_id}({_get_next_word(chunk, sentence)})'


def _chunk_entity_type_f(chunk, chunk_id):
    return f'4_et{chunk_id}({chunk.entType})'


def _chunks_concated_entity_types_f(chunk1, chunk2):
    return f'5_cet({chunk1.entType + chunk2.entType})'


def _dependency_tag_f(chunk_id, sentence):
    tag = sentence[chunk_id - 1].tag
    return f'6_dt({tag})'


def _dependency_type_f(direction, chunk_id, sentence):
    tag = sent_word_data(chunk_id, sentence).dependency
    return f'7_dt{direction}({tag})'


def _string_dependency_word_f(chunk_id, sentence):
    tag = sent_word_data(chunk_id, sentence).lemma
    return f'8_dw({tag})'


def _forward_tag_f(chunk, sentence, chunk_id):
    return f'9_ft{chunk_id}({get_forward_tag(chunk, sentence)})'


def _chunk_head_f(chunk, chunk_id):
    return f'10_ht{chunk_id}({chunk.headWordTag})'


def _bag_of_words_f(first_chunk, second_chunk, sentence):
    first, second = get_first_and_second_chunk(first_chunk, second_chunk)
    between_words = sentence[first.lastWordIndex:second.firstWordIndex - 1]

    return [f'11_bow{word.lemma}' for word in between_words]


def _is_location_f(chunk, chunk_id):
    text = chunk.text.lower()
    return f"12_il{chunk_id}({'T' if text in COUNTRIES or text in CITIES else 'F'})"


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
        all_dependency_tags.append(_string_dependency_word_f(graph[i], sentence))
        i += 1
        if i + 1 >= len(graph):
            return all_dependency_tags
    i += 1
    all_dependency_tags.append(_string_dependency_word_f(graph[i], sentence))
    i += 1
    while i < len(graph):
        all_dependency_tags.append(_string_dependency_word_f(graph[i], sentence))
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

# Itai Mondshine 207814724
# Itay Lotan 308453935

import sys
from datetime import datetime, timedelta
from time import time

from clf_model import ClfModel
from config import OUTPUT_TAG
from eval import _evaluate_predictions
from nlp_data_parser import NLPDataParser


def main():
    train_corpus_file, train_annotations_file, test_corpus_file = _read_program_arguments()

    log_info('starting the run')
    run_start_time = time()

    train_dataset = NLPDataParser.parse(train_corpus_file, train_annotations_file)
    log_info('parse train dataset finished')
    test_dataset = NLPDataParser.parse(test_corpus_file)
    log_info('parse test dataset finished')

    model = ClfModel()
    model.train(train_dataset)

    log_info('train finished')

    train_predictions = model.predict(train_dataset)
    test_predictions = model.predict(test_dataset)
    log_info('predict finished')

    write_final_annotations_to_file(train_dataset, train_predictions, './TRAIN.annotations.predicted.txt')
    write_final_annotations_to_file(test_dataset, test_predictions, './TEST.annotations.predicted.txt')

    run_duration_seconds = round(time() - run_start_time)
    log_info(f'all done. run duration: {timedelta(seconds=run_duration_seconds)}')

    # todo - delete
    _validate_after_refactor()


def _read_program_arguments():
    train_corpus_file, train_annotations_file, test_corpus_file = sys.argv[1], sys.argv[2], sys.argv[3]
    if _corpus_is_in_processed_format(train_corpus_file):
        print('the provided train corpus file is in the .processed format, please provide '
              'the raw unprocessed corpus file instead')
        sys.exit(1)

    if _corpus_is_in_processed_format(test_corpus_file):
        print('the provided test corpus file is in the .processed format, please provide '
              'the raw unprocessed corpus file instead')
        sys.exit(1)

    return train_corpus_file, train_annotations_file, test_corpus_file


def _corpus_is_in_processed_format(corpus_file):
    with open(corpus_file) as f:
        corpus_file_first_char = f.read(1)

    is_in_processed_format = corpus_file_first_char == '#'
    return is_in_processed_format


def log_info(message):
    print(f'{datetime.now().strftime("%H:%M:%S")} - {message}')


def write_final_annotations_to_file(dataset, predicted, output_annotations_file_path):
    all_predicted_annotations_ordered = []
    for i, prediction in enumerate(predicted):
        chunk1, chunk2, _ = dataset[i][0]
        annotation_text = f'{chunk1.id}\t{chunk1.text}\t{prediction}\t{chunk2.text}\t'
        all_predicted_annotations_ordered.append((prediction, annotation_text))

    output_lines = [
        annotation_text for prediction, annotation_text in all_predicted_annotations_ordered
        if prediction == OUTPUT_TAG
    ]
    with open(output_annotations_file_path, 'w') as output_file:
        output_file.write('\n'.join(output_lines))


# todo - delete
def _validate_after_refactor():
    import json

    def _compare_metric_dicts(dict1, dict2, decimals=4):
        import json

        dict1_rounded = {
            outer_key: {inner_key: round(val, decimals) for inner_key, val in inner_dict.items()}
            for outer_key, inner_dict in dict1.items()
        }
        dict2_rounded = {
            outer_key: {inner_key: round(val, decimals) for inner_key, val in inner_dict.items()}
            for outer_key, inner_dict in dict2.items()
        }
        dict1_rounded_json = json.dumps(dict1_rounded)
        dict2_rounded_json = json.dumps(dict2_rounded)
        return dict1_rounded_json == dict2_rounded_json

    actual_test_metrics = _evaluate_predictions(
        gold_file='/Users/itayl/git/toar2/toar2/nlp/ass4/resources/data/DEV.annotations',
        predictions_file='./TEST.annotations.predicted.txt'
    )
    actual_train_metrics = _evaluate_predictions(
        gold_file='/Users/itayl/git/toar2/toar2/nlp/ass4/resources/data/TRAIN.annotations',
        predictions_file='./TRAIN.annotations.predicted.txt'
    )

    print(f"actual_test_metrics:\n{json.dumps(actual_test_metrics, indent=4)}\n\nactual_train_metrics:\n{json.dumps(actual_train_metrics, indent=4)}\n")
    expected_test_metrics = {
        "Live_In": {
            "Precision": 0.532258064516129,
            "Recall": 0.5409836065573771,
            "F1": 0.5365853658536586
        }
    }
    expected_train_metrics = {
        "Live_In": {
            "Precision": 0.7664233576642335,
            "Recall": 0.8267716535433071,
            "F1": 0.7954545454545454
        }
    }

    assert _compare_metric_dicts(actual_test_metrics, expected_test_metrics) is True
    assert _compare_metric_dicts(actual_train_metrics, expected_train_metrics) is True
    print('_validate_after_refactor() - all good')


if __name__ == "__main__":
    # todo - delete
    sys.argv.extend(['./Corpus.TRAIN.txt', './TRAIN.annotations.txt', './Corpus.DEV.txt'])
    main()

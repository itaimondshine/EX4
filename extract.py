# Itai Mondshine 207814724
# Itay Lotan 308453935

import sys
from datetime import datetime, timedelta
from time import time

from clf_model import ClfModel
from config import OUTPUT_TAG
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


if __name__ == "__main__":
    main()

from datetime import datetime, timedelta
from time import time

from clf_model import ClfModel
from config import OUTPUT_TAG
from eval import evaluate_predictions
from nlp_data_parser import NLPDataParser


def main(trainCorpusFile, devCorpusFile, trainAnnotationsFile, devAnnotationsFile):
    log_info('starting the run')
    run_start_time = time()

    train_data_set = NLPDataParser.parse(trainCorpusFile, trainAnnotationsFile)
    log_info('parse train dataset finished')
    dev_data_set = NLPDataParser.parse(devCorpusFile, devAnnotationsFile)
    log_info('parse dev dataset finished')

    model = ClfModel()
    model.train(train_data_set)

    log_info('train finished')

    train_predictions = model.predict(train_data_set)
    dev_predictions = model.predict(dev_data_set)
    log_info('predict finished')

    write_final_annotations_to_file(dev_data_set, dev_predictions, './DEV.annotations.Pred')
    write_final_annotations_to_file(train_data_set, train_predictions, './TRAIN.annotations.Pred')

    run_duration_seconds = round(time() - run_start_time)
    log_info(f'all done. run duration: {timedelta(seconds=run_duration_seconds)}')

    # todo - delete
    _validate_after_refactor()


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

    actual_dev_metrics = evaluate_predictions(
        gold_file='/Users/itayl/git/toar2/toar2/nlp/ass4/resources/data/DEV.annotations',
        predictions_file='DEV.annotations.Pred'
    )
    actual_train_metrics = evaluate_predictions(
        gold_file='/Users/itayl/git/toar2/toar2/nlp/ass4/resources/data/TRAIN.annotations',
        predictions_file='TRAIN.annotations.Pred'
    )

    print(f"actual_dev_metrics:\n{json.dumps(actual_dev_metrics, indent=4)}\n\nactual_train_metrics:\n{json.dumps(actual_train_metrics, indent=4)}\n")

    expected_dev_metrics = {
        "Live_In": {
            "precision": 0.5824175824175825,
            "recall": 0.4344262295081967,
            "f1": 0.4976525821596243
        }
    }
    expected_train_metrics = {
        "Live_In": {
            "precision": 1.0,
            "recall": 0.8267716535433071,
            "f1": 0.9051724137931035
        }
    }

    assert _compare_metric_dicts(actual_dev_metrics, expected_dev_metrics) is True
    assert _compare_metric_dicts(actual_train_metrics, expected_train_metrics) is True
    print('_validate_after_refactor() - all good')


if __name__ == "__main__":
    main(
        trainCorpusFile='./Corpus.TRAIN.txt',
        devCorpusFile='./Corpus.DEV.txt',
        trainAnnotationsFile='./TRAIN.annotations.txt',
        devAnnotationsFile='./DEV.annotations.txt'
    )

# Itai Mondshine 207814724
# Itay Lotan 308453935

import sys
from collections import defaultdict


def main():
    gold_file, predictions_file = _read_program_args()

    gold_annotations = _parse_annotations_file(gold_file)
    predicted_annotations = _parse_annotations_file(predictions_file)
    gold_relations = set(gold_annotations.keys())
    predicted_relations = set(predicted_annotations.keys())

    all_relations = gold_relations.intersection(predicted_relations)
    eval_metrics = _calc_eval_metrics(all_relations, gold_annotations, predicted_annotations)

    display_rows = []
    for relation_type, eval_metrics in eval_metrics.items():
        eval_metrics_r = _round_metrics(eval_metrics)
        row = f"{relation_type}\tPrecision: {eval_metrics_r['Precision']}\tRecall: {eval_metrics_r['Recall']}\tF1: {eval_metrics_r['F1']}"
        display_rows.append(row)
    print('\n'.join(display_rows))


def _read_program_args():
    gold_file, predictions_file = sys.argv[1], sys.argv[2]
    return gold_file, predictions_file


def _parse_annotations_file(annotation_file_path):
    annotations = defaultdict(set)
    with open(annotation_file_path) as annotations_file:
        for line in annotations_file.readlines():
            sent_id, first_chunk, annotation, second_chunk, *_ = line.split('\t')
            anno_data = (sent_id, first_chunk.rstrip('.'), second_chunk.rstrip('.'))
            annotations[annotation].add(anno_data)

        return annotations


def _calc_eval_metrics(all_relations, gold_annotations, predicted_annotations):
    eval_metrics = {}
    for relation in all_relations:
        rel_predicted = predicted_annotations[relation]
        rel_gold = gold_annotations[relation]

        correctly_annotated = rel_gold.intersection(rel_predicted)
        precision = len(correctly_annotated) / len(rel_predicted)
        recall = len(correctly_annotated) / len(rel_gold)
        f1 = (2 * precision * recall) / (precision + recall)

        eval_metrics[relation] = {'Precision': precision, 'Recall': recall, 'F1': f1}
    return eval_metrics


def _round_metrics(metrics_dict, digits=3):
    rounded_metrics_dict = {
        key: round(val, digits) if key in ('Precision', 'Recall', 'F1') else val
        for key, val in metrics_dict.items()
    }
    return rounded_metrics_dict


if __name__ == "__main__":
    main()

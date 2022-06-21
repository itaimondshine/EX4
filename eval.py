import sys


def parse_annotations_file(annotation_file_path):
    with open(annotation_file_path) as annotations_file:
        connections = {}
        relations = set()
        for line in annotations_file.readlines():
            tokens = line.split('\t')
            id = tokens[0]
            first_chunk = tokens[1]
            connection = tokens[2]
            second_chunk = tokens[3]
            relations.add(connection)
            if connection not in connections:
                connections[connection] = set()
            connections[connection].add((id, first_chunk.rstrip('.'), second_chunk.rstrip('.')))

        return connections, relations


def metrics(relations, gold_annotations, predicted_annotations):
    metrics = {}
    for relation in relations:
        correctly_annotated = gold_annotations[relation].intersection(predicted_annotations[relation])
        precision = len(correctly_annotated) / float(len(predicted_annotations[relation]))
        recall = len(correctly_annotated) / float(len(gold_annotations[relation]))
        f1 = 2 * precision * recall / float(precision + recall)
        metrics[relation] = {'precision': precision, 'recall': recall, 'f1': f1}
    return metrics


def print_metrics(metrics):
    for relation, metrics_r in metrics.items():
        print("%s\tPrecision: %s\tRecall: %s\tF1: %s" % (
            relation, metrics_r['precision'], metrics_r['recall'], metrics_r['f1']))


def main_func(params):
    res = evaluate_predictions(params[0], params[1])
    print_metrics(res)


def evaluate_predictions(gold_file, predictions_file):
    gold_annotations, gold_relations = parse_annotations_file(gold_file)
    predicted_annotations, predicted_relations = parse_annotations_file(predictions_file)
    res = metrics(gold_relations.intersection(predicted_relations), gold_annotations, predicted_annotations)
    return res


if __name__ == "__main__":
    main_func(sys.argv[1:])

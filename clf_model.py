import numpy as np
from catboost import CatBoostClassifier

from config import NO_RELATION_TAG, CATBOOST_LOGGING_LEVEL, MODEL_PREDICTED_LABELS, CATBOOST_POSITIVE_CLASS_BOOST_FACTOR
from extract_features import extract_features


class ClfModel:
    def __init__(self):
        self._feature_name_to_id = None
        self._clf = None

    def train(self, train_dataset):
        self._feature_name_to_id = {}
        self._clf = self._init_catboost_classifier()
        features, tags = extract_features(train_dataset, self._feature_name_to_id, allow_map_new_features=True)
        self._clf.fit(features, tags)

    def predict(self, dataset):
        features, tags = extract_features(dataset, self._feature_name_to_id, allow_map_new_features=False)
        raw_predictions = self._clf.predict(features)
        predictions = np.array([row_preds[0] for row_preds in raw_predictions])
        return predictions

    @staticmethod
    def _init_catboost_classifier():
        class_weights = {
            pos_label: CATBOOST_POSITIVE_CLASS_BOOST_FACTOR
            for pos_label in MODEL_PREDICTED_LABELS
        }
        class_weights[NO_RELATION_TAG] = 1
        clf = CatBoostClassifier(loss_function='MultiClass', class_weights=class_weights,
                                 logging_level=CATBOOST_LOGGING_LEVEL)
        return clf

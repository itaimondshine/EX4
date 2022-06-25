import numpy as np
from catboost import CatBoostClassifier

from config import NO_RELATION_TAG, CATBOOST_LOGGING_LEVEL, MODEL_PREDICTED_LABELS, CATBOOST_POSITIVE_CLASS_BOOST_FACTOR
from extract_features import extract_features_for_train, extract_features_for_predict


class ClfModel:
    def __init__(self):
        self._feature_name_to_idx = None
        self._clf = None

    def train(self, train_dataset):
        train_features, train_tags, feature_name_to_idx = extract_features_for_train(train_dataset)
        self._clf = self._init_catboost_classifier()
        self._feature_name_to_idx = feature_name_to_idx
        self._clf.fit(train_features, train_tags)

    def predict(self, dataset):
        features = extract_features_for_predict(dataset, self._feature_name_to_idx)
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

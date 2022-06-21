from sklearn import svm

from extract_features import extract_features


class ClfModel:
    def __init__(self):
        self.feature_name_to_id = None
        self.clf = svm.LinearSVC()

    def train(self, dataset):
        self.feature_name_to_id = {}
        features, tags = extract_features(dataset, self.feature_name_to_id, allow_map_new_features=True)
        self.clf.fit(features, tags)

    def predict(self, data):
        features, tags = extract_features(data, self.feature_name_to_id, allow_map_new_features=False)
        return self.clf.predict(features)

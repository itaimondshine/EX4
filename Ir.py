import pickle
import spacy
import codecs

import numpy as np
from scipy import sparse
from sklearn import metrics, svm, ensemble
from feature_builders import FeatureBuilders
from datetime import datetime

startTime = datetime.now()

TAGS = {'Live_In', 'Work_For'}

nlp = spacy.load('en')
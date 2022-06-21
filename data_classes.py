class NLPChunkData:
    def __init__(self):
        self.text = None
        self.originalText = None
        self.entType = None
        self.rootText = None
        self.rootDep = None
        self.rootHead = None
        self.id = None
        self.firstWordIndex = None
        self.lastWordIndex = None
        self.headWordTag = None
        self.depWordIndex = None


class NLPSentenceWordData:
    def __init__(self):
        self.id = None
        self.word = None
        self.lemma = None
        self.pos = None
        self.tag = None
        self.parent = None
        self.dependency = None
        self.bio = None
        self.ner = None

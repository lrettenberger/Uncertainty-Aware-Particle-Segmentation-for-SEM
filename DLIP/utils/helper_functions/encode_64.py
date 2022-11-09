import codecs
import pickle

def encodeb64(config: dict):
    return codecs.encode(pickle.dumps(config), "base64").decode()
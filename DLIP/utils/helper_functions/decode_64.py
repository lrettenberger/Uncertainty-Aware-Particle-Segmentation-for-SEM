import pickle
import codecs

def decodeb64(base64):
    return pickle.loads(codecs.decode(base64.encode(), "base64"))

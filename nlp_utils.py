import pickle
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import cv2
import editdistance
from path import Path
import sys
sys.path.append('SimpleHTR/src')

from SimpleHTR.src.DataLoaderIAM import DataLoaderIAM, Batch
from SimpleHTR.src.Model import Model, DecoderType
from SimpleHTR.src.SamplePreprocessor import preprocess

class NoteParser:
    # note parser loads up and owns all the models
    w2v_model_path = "models/glove-twitter-25.model"
    pca_model_path = "models/norm_pca_params.pkl"
    htr_char_list_path = "SimpleHTR/model/charList.txt"
    htr = None
    w2v = None
    pca = None

    def __init__(self):
        self.load_htr_model()
        self.load_w2v_model()
        self.load_pca_model()


    def load_htr_model(self):
        self.htr = Model(open(self.htr_char_list_path).read(), decoderType=0, mustRestore=True, dump=False)
        
    def load_w2v_model(self):
        self.w2v = KeyedVectors.load(self.w2v_model_path)
        self.w2v.init_sims()

    def load_pca_model(self):
        with open(self.pca_model_path, 'rb') as infile:
            self.pca = pickle.load(infile)
    
    def photo2txt(self, photo):
        img = preprocess(cv2.imread(photo, cv2.IMREAD_GRAYSCALE), Model.imgSize)
        batch = Batch(None, [img])
        (recognized, probability) = self.htr.inferBatch(batch, True)
        print(f'Recognized: "{recognized[0]}"')
        print(f'Probability: {probability[0]}')
        return recognized[0].lower()
    
    def txt2embedding(self,word):
        return self.w2v.word_vec(word, use_norm=True)
    
    def txt2pca(self,word):
        return self.embedding2pca(self.txt2embedding(word))

    def txt2localpca(self, wordset):
        vectors = [self.txt2embedding(w) for w in wordset]
        localpca = PCA(n_components=2)
        localpca.fit(vectors)
        return localpca.transform(vectors)

    def embedding2pca(self,embedding):
        return self.pca.transform([embedding])[0]

    def photo2embedding(self,photo):
        return self.txt2embedding(self.photo2txt(photo))
    
    def photo2pca(self,photo):
        return self.embedding2pca(self.txt2embedding(self.photo2txt(photo)))

    def txt2clusters(self, wordset, k=3):
        vectors = [self.txt2embedding(w) for w in wordset]
        clusters = KMeans(n_clusters=k).fit(vectors).labels_
        clustered_words = [[] for _ in range(k)]
        print(clusters)
        for i,c in enumerate(clusters):
            clustered_words[c].append(wordset[i])
        print(clustered_words)
        return clustered_words
        


if __name__=='__main__':
    parser = NoteParser()
    print(f"hello: {parser.photo2pca('SimpleHTR/data/test.png')}")
    print(f"hi: {parser.txt2pca('hi')}")
    print(f"cat: {parser.txt2pca('cat')}")
    print(f"feline: {parser.txt2pca('feline')}")
    print(f"car: {parser.txt2pca('car')}")
    print(f"auto: {parser.txt2pca('auto')}")
    

    
    

    



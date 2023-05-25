import os
import sys
import pickle
import random
import numpy as np
import operator

from sklearn.cluster import MiniBatchKMeans

class Discriminator:
    def __init__(self, num_of_htables=28):
        self.num_of_htables = num_of_htables
        self.h_rams = [dict() for x in range(num_of_htables)]
        self.times_trained = 0

    def train(self,X):
        for i in range(0, self.num_of_htables):
            key = X[i]
            if key in self.h_rams[i].keys():
                self.h_rams[i][key] += 1
            else:
                self.h_rams[i][key] = 1
        self.times_trained += 1

    def classify(self,X):
        votes = 0
        for i in range(0, self.num_of_htables):
            key = X[i]
            if key in self.h_rams[i].keys():
                votes += 1

        return (votes, self.times_trained)

    def get_num_trainings(self):
        return self.times_trained

    def get_mental_image(self):
        addresses = []
        for i in range(0,self.num_of_htables):
            addresses.append(list(self.h_rams[i].items()))

        return addresses

class Wisard:
    def __init__(self, num_of_htables, input_addr_length, mapping=None, rank_type="ranks"):
        self.discs = {}
        self.input_addr_length = input_addr_length
        self.num_of_htables = num_of_htables
        self.d_votes = {}
        self.d_times_trained = {}
        self.d_relevance = {}
        if mapping is not None:
            self.mapping = mapping
        else:
            self.mapping = list(range(0, self.num_of_htables*self.input_addr_length))
            random.shuffle(self.mapping)
        self.last_rank = 0
        self.rank_table = {}        
        self.rank_type = rank_type

        if rank_type == "kmeans":
            self.kmeans = MiniBatchKMeans(n_clusters=num_of_htables)        
    
    def ranks(self,tmp):
        addresses = []
        for i in range(0, len(tmp), self.input_addr_length):
            tuples = sorted(list(zip(tmp[i:i+self.input_addr_length],list(range(0,self.input_addr_length)))))
            x,t = zip(*tuples)
            if str(t) not in self.rank_table.keys():
                self.rank_table[str(t)] = self.last_rank
                self.last_rank += 1                
            addresses.append(self.rank_table[str(t)])
        return addresses

    #work in progress: similar n-tuples patterns should produce the same rank
    def kmeans_ranks(self, tmp):
        X = []
        for i in range(0, len(tmp), self.input_addr_length):
            ntuple = list(tmp[i:i+self.input_addr_length])
            X.append(ntuple)        
        return self.kmeans.fit_predict(X)                  

    def train(self, X, label):
        if label not in self.discs:
            self.discs[label] = Discriminator(self.num_of_htables)
            self.d_votes[label] = 0
            self.d_times_trained[label] = 0
            self.d_relevance[label] = 0

        tmp = X[self.mapping]

        if self.rank_type == "kmeans": 
            addresses = self.kmeans_ranks(tmp)    
        else:
            addresses = self.ranks(tmp)            

        self.discs[label].train(addresses)

    def classify(self, X):
        tmp = X[self.mapping] 
        biggest = -1
        secon_biggest = -1
        label = -1
        d_label = None

        if self.rank_type == "kmeans": 
            addresses = self.kmeans_ranks(tmp)    
        else:
            addresses = self.ranks(tmp) 
        
        for label in self.discs.keys():
            self.d_votes[label], self.d_times_trained[label] = self.discs[label].classify(addresses)

        votes = list(sorted(self.d_votes.values()))            
        label = max(self.d_votes, key=self.d_votes.get)

        biggest = votes[-1]
        secon_biggest = votes[-2]

        conf = 0
        if biggest > 0:
            conf = (biggest - secon_biggest)/biggest
        
        return (label, biggest/self.num_of_htables, conf)

    def get_mental_addresses(self, label):
        return self.discs[label].get_mental_image()

    def save(self, path):

        if os.path.isdir(path):
            with open(path + 'discriminators.pkl', 'wb') as output:
                pickle.dump(self.discs, output)

            with open(path + 'mapping.pkl', 'wb') as output:
                pickle.dump(self.mapping, output)
        else:
            os.makedirs(path)
            with open(path + 'discriminators.pkl', 'wb') as output:
                pickle.dump(self.discs, output)

            with open(path + 'mapping.pkl', 'wb') as output:
                pickle.dump(self.mapping, output)

    def load(self, path):
        if os.path.isdir(path):
            with open(path + 'discriminators.pkl', 'rb') as inputs:
                self.discs = pickle.load(inputs)

            with open(path + 'mapping.pkl', 'rb') as inputs:
                self.mapping = pickle.load(inputs)
        else:
            raise FileNotFoundError


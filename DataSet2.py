# -*- Encoding:UTF-8 -*-

import numpy as np
import sys


class DataSet2(object):
    def __init__(self, fileName, cv, teacherPred):
        ##
        self.data, self.train, self.test, self.shape, self.groundTruth, self.ratedItems, self.teacherTopK  = self.getTrainTestData(fileName, cv, teacherPred)
        self.trainDict = self.getTrainDict()



    def getTrainTestData(self, dataName, cv, teacherPred):
        print("Loading " + dataName + " data set...")
        data = []
        train = []
        test = []
        groundTruth = {}
        ##
        ratedItems = {}
        
        trainPath = './Data/' + dataName + '/u' + str(cv) + '.base'
        # trainPath = './Data/' + dataName + '/u' + str(cv) + '_zero_0.4.base'
        testPath = './Data/' + dataName + '/u' + str(cv) + '.test'
        u = 0
        i = 0
        maxr = 0.0
        print('opening ' + trainPath + ' file...')
        with open(trainPath, 'r') as f:    # read a training file
            for line in f:
                if line:
                    if dataName != 'watcha':
                        lines = line[:-1].split('\t')
                    else:
                        lines = line[:-1].split()
                    user = int(lines[0])
                    movie = int(lines[1])
                    score = float(lines[2])

                    ## get user-item combination in trainset
                    if user-1 not in ratedItems.keys():
                        ratedItems[user-1] = []   
                    ratedItems[user-1].append(movie-1)
                    ##

                    ##
                    data.append((user-1, movie-1, score))
                    train.append((user-1, movie-1, score))
                    if user > u:
                        u = user
                    if movie > i:
                        i = movie
                    if score > maxr:
                        maxr = score
        print('opening ' + testPath + ' file...')
        with open(testPath, 'r') as f:    # read a test file
            for line in f:
                if line:
                    if dataName != 'watcha':
                        lines = line[:-1].split('\t')
                    else:
                        lines = line[:-1].split()
                    user = int(lines[0])
                    movie = int(lines[1])
                    score = float(lines[2])

                    if user-1 not in groundTruth.keys():
                        groundTruth[user-1] = []
                    if score == 5:
                        groundTruth[user-1].append(movie-1)
                    
                    ## get user-item combination in testset
                    if user-1 not in ratedItems.keys():
                        ratedItems[user-1] = []   
                    ratedItems[user-1].append(movie-1)
                    ##

                    ##
                    data.append((user-1, movie-1, score))
                    test.append((user-1, movie-1, score))
                    
                    if user > u:
                        u = user
                    if movie > i:
                        i = movie
                    if score > maxr:
                        maxr = score
        self.maxRate = maxr

        print("Loading Success!\n"
              "Data Info:\n"
              "\tUser Num: {}\n"
              "\tItem Num: {}\n"
              "\tData Size: {}".format(u, i, len(data)))

        ##get teacher's prediction file
        p = open('./_Knowledge/' + dataName + '/' + teacherPred + '.txt')
        teacherTopK = p.readlines()
        for k in range(len(teacherTopK)):
            teacherTopK[k] = teacherTopK[k].strip().split(sep = '\t')
            teacherTopK[k] = list(map(int, teacherTopK[k]))
        p.close()
        ##
   
        return data, train, test, [u, i], groundTruth, ratedItems, teacherTopK
        

    def getTrainDict(self):
        dataDict = {}
        for i in self.train:
            dataDict[(i[0], i[1])] = i[2]
        return dataDict

    def getEmbedding(self, predScore):
        train_matrix = np.zeros([self.shape[0], self.shape[1]], dtype=np.float32)
        for i in self.train:
            user = i[0]
            movie = i[1]
            rating = i[2]
            train_matrix[user][movie] = rating

        ##
        for i in range(len(self.teacherTopK)):
            for j in range(len(self.teacherTopK[i])):
                movie = self.teacherTopK[i][j]
                train_matrix[i][movie] = predScore
        ##
        
        return np.array(train_matrix)

    def getInstances(self, data, negNum):
        user = []
        item = []
        rate = []
        for i in data:
            user.append(i[0])
            item.append(i[1])
            rate.append(i[2])
            for t in range(negNum):
                j = np.random.randint(self.shape[1])
                while ((i[0], j) in self.trainDict) or (j in self.teacherTopK[i[0]]):
                    j = np.random.randint(self.shape[1])
                user.append(i[0])
                item.append(j)
                rate.append(0.0)
        return np.array(user), np.array(item), np.array(rate)

    def getTestNeg(self, testData, negNum):
        user = []
        item = []
        old_u = -1
        for s in testData:
            tmp_user = []
            tmp_item = []
            u = s[0]

            if old_u == u:
                continue

            for t in range(self.shape[1]):
                if (u, t) not in self.trainDict:    # Caution: excluding interesting items
                    tmp_user.append(u)
                    tmp_item.append(t)
            old_u = u
            user.append(tmp_user)
            item.append(tmp_item)

        return [np.array(user), np.array(item)]

    ##
    def getTopKNeg(self, data, negNum):
        user = []
        item = []
        old_u = []
        for s in data:
            tmp_user = []
            tmp_item = []
            u = s[0]

            if u in old_u:
              continue
            else:    
                for t in range(self.shape[1]):
                    tmp_user.append(u)
                    tmp_item.append(t)
            old_u.append(u)
            user.append(tmp_user)
            item.append(tmp_item)

        return [np.array(user), np.array(item)]
    ##

    ##
    def getUnratedItems(self):
        Items = self.getTopKNeg(self.data, 100)[1]
        itemDict = {i: Items[i] for i in range(len(Items))}

        unratedItems = {}
        
        for i in range(len(itemDict)):
            unratedItems[i] = list(set(itemDict[i]) - set(self.ratedItems[i]))

        return unratedItems
    ##






# -*- Encoding:UTF-8 -*-

import tensorflow as tf
import numpy as np
import argparse
from DataSet2 import DataSet2
import sys
import os
import heapq
import math

def main():
    parser = argparse.ArgumentParser(description="Options")
    
    parser.add_argument('-dataName', action='store', dest='dataName', default='ml-100k')
    parser.add_argument('-negNum', action='store', dest='negNum', default=7, type=int) 

    ### smaller than teacher?
    parser.add_argument('-userLayer', action='store', dest='userLayer', default=[128,16])
    parser.add_argument('-itemLayer', action='store', dest='itemLayer', default=[256, 16])
    # parser.add_argument('-reg', action='store', dest='reg', default=1e-3)
    parser.add_argument('-lr', action='store', dest='lr', default=0.0001)  
    parser.add_argument('-maxEpochs', action='store', dest='maxEpochs', default=10, type=int) 
    parser.add_argument('-batchSize', action='store', dest='batchSize', default=256, type=int)
    parser.add_argument('-earlyStop', action='store', dest='earlyStop', default=5)
    parser.add_argument('-checkPoint', action='store', dest='checkPoint', default='./checkPoint/')
    parser.add_argument('-topK', action='store', dest='topK', default=10)
    parser.add_argument('-cv', action='store', dest='cv', default=1, type=int)


    ## teacher's topk prediction
    parser.add_argument('-teacherPred', action='store', dest='teacherPred', default='predictions')
    ## here alpha=1.0 stands for equal weight for ranking loss and distillation loss
    parser.add_argument('-teach_alpha', action='store', dest='teach_alpha', default=0.5)
    ## hyperparameter for tuning the sharpness of position importance weight in Eq.(8)
    parser.add_argument('-lamda', action='store', dest='lamda', default=1)
    ## hyperparameter for tuning the sharpness of ranking discrepancy weight in Eq.(9)
    parser.add_argument('-mu', action='store', dest='mu', default=0.1)
    ## number of samples used for estimating student's rank in Eq.(9)
    parser.add_argument('-num_dynamic_samples', action='store', dest='num_dynamic_samples', default=100)
    ## number of iteration to start using hybrid of two different weights
    ##10->1
    parser.add_argument('-dynamic_start_epoch', action='store', dest='dynamic_start_epoch', default=5)
    ## number of teacher predictions' items
    parser.add_argument('-K', action='store', dest='K', default=10)
    ## score of teacher predictions' items
    parser.add_argument('-predScore', action='store', dest='predScore', default=5.0)
    ## weight scheme option (1.uniform 2.position 3.discrepancy 4.hybrid)
    parser.add_argument('-optionW', action='store', dest='optionW', default=4)
    
    args = parser.parse_args()
    
    classifier = DistilledModel(args)

    classifier.run()

class DistilledModel:
    def __init__(self, args):
        self.dataName = args.dataName
        self.cv = args.cv
        ##
        self.teacherPred = args.teacherPred
        self.dataSet = DataSet2(self.dataName, self.cv, self.teacherPred)
        ##
        self.shape = self.dataSet.shape
        self.maxRate = self.dataSet.maxRate

        ##
        self.teach_alpha = args.teach_alpha
        self.lamda = args.lamda
        self.mu = args.mu
        self.num_dynamic_samples = args.num_dynamic_samples
        self.dynamic_start_epoch = args.dynamic_start_epoch
        self.K = args.K
        self.predScore = args.predScore
        self.optionW = args.optionW
        ##

        ##get unratedItems
        self.unratedItems = self.dataSet.getUnratedItems()
        ##

        self.train = self.dataSet.train
        self.test = self.dataSet.test
        self.groundTruth = self.dataSet.groundTruth  # newly added variable

        ##
        self.teacherTopK = self.dataSet.teacherTopK

        self.negNum = args.negNum
        self.testNeg = self.dataSet.getTestNeg(self.test, 99)
        
        self.add_embedding_matrix()

        self.add_placeholders()

        self.userLayer = args.userLayer
        self.itemLayer = args.itemLayer
        self.add_model()

        ##
        self.add_loss1()
        self.add_loss2()
        ##

        self.lr = args.lr
        self.add_train_step()

        self.checkPoint = args.checkPoint
        self.init_sess()

        self.maxEpochs = args.maxEpochs
        self.batchSize = args.batchSize

        self.topK = args.topK
        self.earlyStop = args.earlyStop

    def add_placeholders(self):
        self.user = tf.placeholder(tf.int32)
        self.item = tf.placeholder(tf.int32)
        self.rate = tf.placeholder(tf.float32)
        ##
        self.weight = tf.placeholder(tf.float32)
        self.drop = tf.placeholder(tf.float32)

    def add_embedding_matrix(self):
        self.user_item_embedding = tf.convert_to_tensor(self.dataSet.getEmbedding(self.predScore))
        self.item_user_embedding = tf.transpose(self.user_item_embedding)
        

    def add_model(self):
        user_input = tf.nn.embedding_lookup(self.user_item_embedding, self.user)
        item_input = tf.nn.embedding_lookup(self.item_user_embedding, self.item)

        def init_variable(shape, name):
            return tf.Variable(tf.truncated_normal(shape=shape, dtype=tf.float32, stddev=0.01), name=name)

        with tf.name_scope("User_Layer"):
            user_W1 = init_variable([self.shape[1], self.userLayer[0]], "user_W1")
            user_out = tf.matmul(user_input, user_W1)
            for i in range(0, len(self.userLayer) - 1):
                W = init_variable([self.userLayer[i], self.userLayer[i + 1]], "user_W" + str(i + 2))
                b = init_variable([self.userLayer[i + 1]], "user_b" + str(i + 2))
                user_out = tf.nn.relu(tf.add(tf.matmul(user_out, W), b))

        with tf.name_scope("Item_Layer"):
            item_W1 = init_variable([self.shape[0], self.itemLayer[0]], "item_W1")
            item_out = tf.matmul(item_input, item_W1)
            for i in range(0, len(self.itemLayer) - 1):
                W = init_variable([self.itemLayer[i], self.itemLayer[i + 1]], "item_W" + str(i + 2))
                b = init_variable([self.itemLayer[i + 1]], "item_b" + str(i + 2))
                item_out = tf.nn.relu(tf.add(tf.matmul(item_out, W), b))

        norm_user_output = tf.sqrt(tf.reduce_sum(tf.square(user_out), axis=1))
        norm_item_output = tf.sqrt(tf.reduce_sum(tf.square(item_out), axis=1))
        self.y_ = tf.reduce_sum(tf.multiply(user_out, item_out), axis=1, keep_dims=False) / (
                    norm_item_output * norm_user_output)
        self.y_ = tf.maximum(1e-6, self.y_)   
      
    def add_loss1(self):
       regRate = self.rate / self.maxRate
       losses = regRate * tf.log(self.y_) + (1 - regRate) * tf.log(1 - self.y_)
       loss = -tf.reduce_sum(losses)
       # regLoss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
       # self.loss = loss + self.reg * regLoss
       self.loss1 = loss * (1 - self.teach_alpha)

    ##
    def add_loss2(self):
        losses = self.weight * tf.log_sigmoid(self.y_)
        loss = -tf.reduce_sum(losses)

        self.loss2 = loss * self.teach_alpha
    ##
    
        
    def add_train_step(self):
        '''
        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.lr = tf.train.exponential_decay(self.lr, global_step,
                                             self.decay_steps, self.decay_rate, staircase=True)
        '''
        optimizer = tf.train.AdamOptimizer(self.lr)
        ##
        self.train_step = optimizer.minimize(self.loss1 + self.loss2)
                
        
    def init_sess(self):
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.config.allow_soft_placement = True
        self.sess = tf.Session(config=self.config)
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()
        if os.path.exists(self.checkPoint):
            [os.remove(f) for f in os.listdir(self.checkPoint)]
        else:
            os.mkdir(self.checkPoint)

    def run(self):
        # best_hr = -1
        # best_NDCG = -1

        best_precisions = np.zeros(4)
        best_recalls = np.zeros(4)
        best_ndcgs = np.zeros(4)
        best_mrrs = np.zeros(4)
        best_totalMrrs = 0

        best_epoch = -1

        ##
        self.epochDynamic = 0
        
        print("Start Training!")
        for epoch in range(self.maxEpochs):
            print("=" * 20 + "Epoch ", epoch, "=" * 20)
            self.run_epoch(self.sess)
            ##
            self.epochDynamic += 1
            print('=' * 50)
            print("Start Evaluation!")
            # hr, NDCG = self.evaluate(self.sess, self.topK)
            precisions, recalls, ndcgs, mrrs, total_mrrs = self.evaluate(self.sess, self.topK)

            print("Epoch ", epoch,
                  "Precision: {}, Recall: {}, NDCG: {}, MRR: {}".format(precisions[0], recalls[0], ndcgs[0], mrrs[0]))
            if precisions[0] > best_precisions[0] or recalls[0] > best_recalls[0] or ndcgs[0] > best_ndcgs[0] or mrrs[
                0] > best_mrrs[0]:
                for idx in range(4):
                    best_precisions[idx] = precisions[idx]
                    best_recalls[idx] = recalls[idx]
                    best_ndcgs[idx] = ndcgs[idx]
                    best_mrrs[idx] = mrrs[idx]
                    best_totalMrrs = total_mrrs

                best_epoch = epoch
                self.saver.save(self.sess, self.checkPoint)

            if epoch - best_epoch > self.earlyStop:
                print("Normal Early stop!")
                break
            print("=" * 20 + "Epoch ", epoch, "End" + "=" * 20)

        print("Best Precision@5: {}, Recall@5: {}, NDCG@5: {}, MRR@5: {} At Epoch {}".format(best_precisions[0],
                                                                                             best_recalls[0],
                                                                                             best_ndcgs[0],
                                                                                             best_mrrs[0], best_epoch))
        print("Training complete!")

        for idx in range(4):
            fout = open('./Result/' + self.dataName + '/result_u' + str(self.cv) + '.txt', 'a')
            # fout = open('./Result/' + self.dataName + '/result_u' + str(self.cv) + '_zero_0.4.txt', 'a')
            resultLine = str(best_precisions[idx]) + '\t' + str(best_recalls[idx]) + '\t' + str(best_ndcgs[idx]) + '\t' \
                         + str(best_mrrs[idx]) + '\t' + str(best_totalMrrs) + '\n'
            fout.write(resultLine)
            fout.close()

    def run_epoch(self, sess, verbose=10):
        train_u, train_i, train_r = self.dataSet.getInstances(self.train, self.negNum)
        train_len = len(train_u)
        shuffled_idx = np.random.permutation(np.arange(train_len))
        train_u = train_u[shuffled_idx]
        train_i = train_i[shuffled_idx]
        train_r = train_r[shuffled_idx]

        num_batches = len(train_u) // self.batchSize + 1
        # num_batches = 2501

        ##
        User, Item, Rate, Weight = [], [], [], []
        for i in range(len(self.teacherTopK)):
            User = np.append(User, np.ones(self.K, dtype = int) * i)
            Item = np.append(Item, np.array(self.teacherTopK[i]))
            Rate = np.append(Rate, np.ones(self.K, dtype = float) * self.predScore)

            if self.optionW == 1:
                Weight = np.append(Weight, np.ones(self.K) / self.K)
                
            elif self.optionW == 2 or (self.optionW == 4 and self.epochDynamic < self.dynamic_start_epoch):
                W = np.array(range(1, self.K + 1), dtype = np.float32)
                W = np.exp(-W / self.lamda)
                Weight = np.append(Weight, W / np.sum(W))

            elif self.optionW == 3:
                shuffled_idx = np.random.permutation(np.arange(len(self.unratedItems[i])))[0:self.num_dynamic_samples - self.K]
                dynamic_samples = np.array(self.unratedItems[i])[shuffled_idx]
                dynamic_samples = np.append(self.teacherTopK[i], dynamic_samples)

                feed_dict = self.create_feed_dict(np.ones(self.num_dynamic_samples)*i, dynamic_samples)
                predict = sess.run(self.y_, feed_dict=feed_dict)
                predict_rank = [sorted(predict).index(x) for x in predict][0:self.K]

                for k in range(self.K):
                    predict_rank[k] = predict_rank[k] - k   

                W = np.tanh([self.mu * x for x in predict_rank])
                W = np.maximum(W, np.zeros(self.K, dtype = float))
                Weight = np.append(Weight, W)

            elif self.optionW == 4 and self.epochDynamic >= self.dynamic_start_epoch:
                weight_static = np.array(range(1, self.K + 1), dtype = np.float32)
                weight_static = np.exp(-weight_static / self.lamda)
                weight_static = weight_static / np.sum(weight_static)

                shuffled_idx = np.random.permutation(np.arange(len(self.unratedItems[i])))[0:self.num_dynamic_samples - self.K]
                dynamic_samples = np.array(self.unratedItems[i])[shuffled_idx]
                dynamic_samples = np.append(self.teacherTopK[i], dynamic_samples)

                feed_dict = self.create_feed_dict(np.ones(self.num_dynamic_samples)*i, dynamic_samples)
                predict = sess.run(self.y_, feed_dict=feed_dict)
                predict_rank = [sorted(predict).index(x) for x in predict][0:self.K]

                for k in range(self.K):
                    predict_rank[k] = predict_rank[k] - k   

                weight_dynamic = np.tanh([self.mu * x for x in predict_rank])
                weight_dynamic = np.maximum(weight_dynamic, np.zeros(self.K, dtype = float))

                Weight = np.append(Weight, weight_static * weight_dynamic)
        ##            


        ##
        topk_len = len(User)
        shuffled_idx = np.random.permutation(np.arange(topk_len))
        topk_u = User[shuffled_idx]
        topk_i = Item[shuffled_idx]
        topk_r = Rate[shuffled_idx]
        topk_w = Weight[shuffled_idx]
        ##
        
        losses = []
        for i in range(num_batches):
            min_idx = i * self.batchSize
            max_idx = np.min([train_len, (i + 1) * self.batchSize])
            train_u_batch = train_u[min_idx: max_idx]
            train_i_batch = train_i[min_idx: max_idx]
            train_r_batch = train_r[min_idx: max_idx]

            ##
            train_w_batch = -np.zeros(len(train_u_batch), dtype = np.float32)

            min_idx = i * self.batchSize
            max_idx = np.min([topk_len, (i + 1) * self.batchSize])
            topk_u_batch = topk_u[min_idx: max_idx]
            topk_i_batch = topk_i[min_idx: max_idx]
            topk_r_batch = topk_r[min_idx: max_idx]
            topk_w_batch = topk_w[min_idx: max_idx]

            U = np.append(train_u_batch, topk_u_batch)
            I = np.append(train_i_batch, topk_i_batch)
            R = np.append(train_r_batch, topk_r_batch)
            W = np.append(train_w_batch, topk_w_batch)


            feed_dict = self.create_feed_dict(U, I, R, W)
            _, tmp_loss1, tmp_loss2 = sess.run([self.train_step, self.loss1, self.loss2], feed_dict=feed_dict)
            ##
            
      
            losses.append(tmp_loss1 + tmp_loss2)
            if verbose and i % verbose == 0:
                sys.stdout.write('\r{} / {} : loss = {}'.format(
                    i, num_batches, np.mean(losses[-verbose:])
                ))
                sys.stdout.flush()
        loss = np.mean(losses)
        print("\nMean loss in this epoch is: {}".format(loss))
        return loss

    ##add weight
    def create_feed_dict(self, u, i, r=None, w=None, drop=None):
        return {self.user: u,
                self.item: i,
                self.rate: r,
                self.weight: w,
                self.drop: drop}
    

    def evaluate(self, sess, topK):
        precisions = np.zeros(4)
        recalls = np.zeros(4)
        ndcgs = np.zeros(4)
        mrrs = np.zeros(4)
        total_mrrs = 0

        sumPrecision = np.zeros(4)
        sumRecall = np.zeros(4)
        sumNDCG = np.zeros(4)
        sumMRR = np.zeros(4)
        sumTotalMRR = 0

        testUser = self.testNeg[0]
        testItem = self.testNeg[1]

        # evaluatedUsers = 0

        for i in range(len(testUser)):  # the number of users
            # print('Test user:' + str(i+1))
            # target = testItem[i][0]    # the item of testset for the corresponding user
            currentUser = testUser[i][0]
            # print('Current user: ' + str(currentUser))

            target_gt = self.groundTruth[currentUser]

            if len(target_gt) == 0:
                continue

            # evaluatedUsers += 1

            feed_dict = self.create_feed_dict(testUser[i], testItem[i])
            predict = sess.run(self.y_, feed_dict=feed_dict)
            item_score_dict = {}
            for j in range(len(testItem[i])):
                item = testItem[i][j]
                item_score_dict[item] = predict[j]

            full_ranklist = heapq.nlargest(self.shape[1], item_score_dict, key=item_score_dict.get)
            userTotalMRR = 0
            for k in range(self.shape[1]):
                if full_ranklist[k] in target_gt:
                    userTotalMRR = 1.0 / (k + 1.0)
                    break
            sumTotalMRR += userTotalMRR

            for topN_idx in range(4):
                ranklist = heapq.nlargest((topN_idx + 1) * 5, item_score_dict, key=item_score_dict.get)

                mrrFlag = True
                userHit = 0
                userMRR = 0
                dcg = 0
                idcg = 0
                idcgCount = len(target_gt)
                ndcg = 0

                for j in range((topN_idx + 1) * 5):
                    if ranklist[j] in target_gt:
                        dcg += 1.0 / math.log2(j + 2)
                        if mrrFlag:
                            userMRR = (1.0 / (j + 1))
                            mrrFlag = False
                        userHit += 1
                    if idcgCount > 0:
                        idcg += 1.0 / math.log2(j + 2)
                        idcgCount -= 1
                if idcg != 0:
                    ndcg += (dcg / idcg)

                sumPrecision[topN_idx] += (userHit / ((topN_idx + 1) * 5))
                sumRecall[topN_idx] += (userHit / len(target_gt))
                sumNDCG[topN_idx] += ndcg
                sumMRR[topN_idx] += userMRR

        total_mrrs = sumTotalMRR / len(testUser)
        for idx in range(4):
            precisions[idx] = sumPrecision[idx] / len(testUser)
            recalls[idx] = sumRecall[idx] / len(testUser)
            ndcgs[idx] = sumNDCG[idx] / len(testUser)
            mrrs[idx] = sumMRR[idx] / len(testUser)

        return precisions, recalls, ndcgs, mrrs, total_mrrs


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 21:26:28 2018

@author: Charlotte
"""

from __future__ import division
import numpy as np
import csv
import os
from analysis.StatisticalAnalysis import ImportFeatures,  ConfidenceInterval
import SimpleITK as sitk
import os,sys
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.model_selection import LeaveOneOut, GridSearchCV, KFold
from sklearn.metrics import roc_auc_score, roc_curve
#from sklearn.svm import SVR
from sklearn.svm import SVC

Modality = ('ADC', 'CESAG', 'CETRA', 'T2SAG', 'T2TRA',)
# addr_AType = '../Features_Extracted/ATYPE/'
# addr_BType = '../Features_Extracted/BTYPE/'

# patho = ('LVSI', 'Nerve', 'LYMN', 'ParaAorta', 'Stroma')#病理学

class SVC_AB(object):
    def __init__(self):
        super(SVC_AB, self).__init__()

    def __call__(self, *args, **kw):
        return self.evaluate(*args, **kw)

    def evaluate(self, addr_AType,addr_BType, result_folder):

        NumOfSubjects = 40
        cv = KFold(NumOfSubjects)#K折检验
        fpr = {}
        tpr = {}
        thresholds = {}
        auc = {}
        acc = {}
        sen = {}
        spe = {}

        for i_modality_index in np.arange(5):

            addr_feature_A = addr_AType +'/'+ Modality[i_modality_index] + '_Features/' #存储feature的文件夹  '../Features_Extracted/ATYPE/ADC_Features'
            addr_feature_B = addr_BType +'/'+ Modality[i_modality_index] + '_Features/'

            feature_arrayA0, patientA, feature_par = ImportFeatures(addr_feature_A)#Feature_arrayA0的shape为105*20，相当于把所有的A拼成了一个array,但不一定是0-20,但是是对应的
            feature_arrayB0, patientB, feature_par = ImportFeatures(addr_feature_B)

           # print(feature_arrayA0)
            # remove outlier 异常值
            feature_arrayA = np.zeros([np.size(feature_arrayA0, 0), np.size(feature_arrayA0, 1)]) + 1e-10
            UsePositionA = np.where(abs(feature_arrayA0) > 1e-10)
            feature_arrayA[UsePositionA] = feature_arrayA0[UsePositionA]  #创建全为0的array

            feature_arrayB = np.zeros([np.size(feature_arrayB0, 0), np.size(feature_arrayB0, 1)]) + 1e-10
            UsePositionB = np.where(abs(feature_arrayB0) > 1e-10)
            feature_arrayB[UsePositionB] = feature_arrayB0[UsePositionB]

            NumOfFeaturesA = np.size(feature_arrayA0[:, 0])#Feature的数量  105
            NumOfFeaturesB = np.size(feature_arrayB0[:, 0])

            NumOfSubjectsA = np.size(feature_arrayA0[0, :])#病人的数量，20
            NumOfSubjectsB = np.size(feature_arrayB0[0, :])

            feature_array = np.transpose(np.hstack((feature_arrayA, feature_arrayB)))#shape:(40,105)
            patient = np.hstack((patientA, patientB))#病人名（40，1）
            group_inf = np.hstack((np.ones(NumOfSubjectsA), np.zeros(NumOfSubjectsB)))#shape:(40,),A的全是1，B的全是0
            # GroupInfArr = GroupInfLoading(PatientName_Existed=patientA, Location_Of_GroupInformation='../GI.xlsx')
            # group_inf = GroupInfArr[:, i_patho]

            test_score_record = {}
            test_score = []
            for train, test in cv.split(feature_array, group_inf):   #40折，test为1，trian为39   array([0])
                train_feature_array = feature_array[train, :]#训练集 (39,105)
                train_label = group_inf[train]#对应的A（1）,B（0）类别
                test_feature_array = feature_array[test, :]#测试集（1，105）
                test_label = group_inf[test]#对应的类别

                #数据预处理，给数据标准化
                scaler = preprocessing.StandardScaler()#size(scaler.mean)=105;
                scaler.fit(train_feature_array)
                train_DataX = scaler.transform(train_feature_array)
                test_DataX = scaler.transform(test_feature_array)

                clf = SVC(kernel='rbf',probability=True)
                #y_rbf = svr_rbf.fit(X, y).predict(X)


                clf.fit(train_DataX, train_label)#拟合训练集
                a = clf.predict_proba(test_DataX)#测试集的预测结果  test_score_record[A1]=[[0.5503842 0.4496158]] 前者为0 后者为1
                #print(a)
                test_score_record[patient[test[0]]] = clf.predict_proba(test_DataX)[0][1]  #test[0]即为训练集的标号，本次为0，实时记录本次预测的值.[0][1]
                test_score.append(test_score_record[patient[test[0]]])#记录总的
                print ('patient: ', patient[test[0]], ', predict: ', test_score_record[patient[test[0]]])#patient[0]=A1,实时记录值

            bit_score = np.array(np.array(test_score)> 0.5).astype(int)  #大于0.5则标记为1   test_score大小为（40，）
            #fpr[ADC]横坐标 在该阈值下的假阳率  ,tpr[ADC]纵坐标 在该阈值下的真阳率,thresholds[ADC]= 按道理来说应当根据score从小到大一个个取
            #roc_curve(实际标签，是正样本的概率值，标签为1被认为是正样本)依次测试不同的threshold 得到假阳率和真阳率
            fpr[Modality[i_modality_index]], tpr[Modality[i_modality_index]], thresholds[Modality[i_modality_index]] = roc_curve(group_inf, np.array(test_score), pos_label=1)

            sen[Modality[i_modality_index]] = np.sum(bit_score*group_inf)/np.sum(group_inf)#计算用阈值为0.5得到的sensitivity和spcific   sen['ADC']=0.75
            spe[Modality[i_modality_index]] = np.sum((1-bit_score)*(1-group_inf))/np.sum(1-group_inf)

            auc[Modality[i_modality_index]] = roc_auc_score(group_inf, test_score)#计算预测得分曲线下的面积
            Score = np.sum(np.count_nonzero(bit_score == group_inf))#预测对的例子个数
            acc[Modality[i_modality_index]] = Score / NumOfSubjects#预测精度

            # confidence interval
            CI_arr = (bit_score==group_inf)  #由True，False组成的array
            up, down = ConfidenceInterval(CI_arr)#置信区间

            #和上面重复了吧orz
            sen[Modality[i_modality_index]] = np.sum(bit_score * group_inf) / np.sum(group_inf)
            spe[Modality[i_modality_index]] = np.sum((1 - bit_score) * (1 - group_inf)) / np.sum(1 - group_inf)
            # sen
            CI_sen_pos = np.where(group_inf == 1)#找到正样本的索引[0,1,2...19]
            sen_group_inf = group_inf[CI_sen_pos]#[1,1,1,]
            sen_rst = bit_score[CI_sen_pos]#正样本的预测值
            CI_sen = (sen_group_inf == sen_rst)#比较正样本是否预测正确
            # spe
            CI_spe_pos = np.where(group_inf == 0)
            spe_group_inf = group_inf[CI_spe_pos]
            spe_rst = bit_score[CI_spe_pos]
            CI_spe = (spe_group_inf == spe_rst)#比较负样本是否预测正确

        #获得置信区间
            sen_up, sen_down = ConfidenceInterval(CI_sen)
            spe_up, spe_down = ConfidenceInterval(CI_spe)


            if not os.path.exists(result_folder):
                os.makedirs(result_folder)

            # print and saving
            # print('nested_test_accuracy:', np.array(bit_score == group_inf).astype(int))
            # print('prob:', test_score)#预测得分
            # print('Modality:',Modality[i_modality_index])
            # print('Accuracy:', Score)#预测对的个数
            # print('Accuracy:', acc[Modality[i_modality_index]])#预测精度
            # print('AUC: ', auc[Modality[i_modality_index]])#曲线面积

            prob_filename = result_folder + '/TYPE_' + str(Modality[i_modality_index]) + '_prob.csv'
            with open(prob_filename, 'w+') as f:
                f.write('auc' + ',' + str(auc[Modality[i_modality_index]]) + '\n')

                f.write('Accuracy' + ',' + str(Score / NumOfSubjects) + '\n')
                f.write('sensitivity' + ',' + str(sen[Modality[i_modality_index]]) + '\n')
                f.write('specificity' + ',' + str(spe[Modality[i_modality_index]]) + '\n')

                f.write('CI_acc' + ',' + str(down) + ',' + str(up) + ',' + '\n')
                f.write('CI_sen' + ',' + str(sen_down) + ',' + str(sen_up) + ',' + '\n')
                f.write('CI_spe' + ',' + str(spe_down) + ',' + str(spe_up) + ',' + '\n')


                f.write('ACCNUM' + ',' + str(Score) + '\n')

                for i in np.arange(np.size(test_score)):
                    f.write(patient[i] + ',' + str(test_score[i]) + '\n')
            f.close()

            # plotting
        plt.figure(dpi=1200)
        lw = 2
        plt.plot(fpr['ADC'], tpr['ADC'], color='darkorange', lw=lw,
                 label='ADC (auc = %0.2f, acc = %0.2f, sen = %0.2f, spe = %0.2f)' % (auc['ADC'], acc[('ADC')], sen[('ADC')], spe['ADC']))
        plt.plot(fpr['CESAG'], tpr['CESAG'], color='green', lw=lw,
                 label='CESAG (auc = %0.2f, acc = %0.2f, sen = %0.2f, spe = %0.2f)' % (auc['CESAG'], acc['CESAG'], sen[('CESAG')], spe['CESAG']))
        plt.plot(fpr['CETRA'], tpr['CETRA'], color='blue', lw=lw,
                 label='CETRA (auc = %0.2f, acc = %0.2f, sen = %0.2f, spe = %0.2f)' % (auc['CETRA'], acc['CETRA'], sen[('CETRA')], spe['CETRA']))
        plt.plot(fpr['T2SAG'], tpr['T2SAG'], color='pink', lw=lw,
                 label='T2SAG (auc = %0.2f, acc = %0.2f, sen = %0.2f, spe = %0.2f)' % (auc['T2SAG'], acc['T2SAG'], sen[('T2SAG')], spe['T2SAG']))
        plt.plot(fpr['T2TRA'], tpr['T2TRA'], color='red', lw=lw,
                 label='T2TRA (auc = %0.2f, acc = %0.2f, sen = %0.2f, spe = %0.2f)' % (auc['T2TRA'], acc['T2TRA'], sen[('T2TRA')], spe['T2TRA']))
        plt.plot()

        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig(
            fname=result_folder + '/'+str('TYPE') + '.png')
        # plt.show(1)


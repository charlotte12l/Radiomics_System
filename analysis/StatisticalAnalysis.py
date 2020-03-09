import os
import pandas as pd
import csv
import numpy as np
from scipy import stats
import xlrd

def ImportFeatures(feature_addr):
    filenames = os.listdir(feature_addr)
    filenames.sort(key=str.lower)
    feature_array = np.array([])
    PatientName_Existed = []

    for i_filename in filenames:
        # get patient name
        i_patient_name = i_filename[:-4]#A1,A2,A3
        PatientName_Existed.append(i_patient_name)#记录所有的病人姓名
        # read feature from file
        addr = os.path.join(feature_addr, i_filename)#每个病人特征的路径
        feature = pd.read_csv(addr, header=None)#读取文件
        feature_par = feature.values[:, 0]#读取第一列  即特征名字

        if len(feature_array) != 0:
            feature_array = np.hstack((feature_array, feature.values[:, [1]]))#按列把数组给堆叠起来，feature.values[:, [1]]为第二列数据，但给每一个数据加上[]从而可堆叠
        else:
            feature_array = feature.values[:, [1]] #不知道啥意思。。。,创建array

    return feature_array, PatientName_Existed, feature_par


def TTestOfFeatures(array_A, array_B):

    NumOfFeatures = np.size(array_A[:, 0])
    t_TTest = np.zeros(NumOfFeatures)
    p_TTest = np.zeros(NumOfFeatures)
    for i_feature in np.arange(NumOfFeatures):
        t_TTest[i_feature], p_TTest[i_feature] = stats.ttest_ind(array_A[i_feature, :], array_B[i_feature, :], equal_var=False)
    FeatureIndex_TTest = np.where(p_TTest < 0.05)

    return t_TTest, p_TTest, FeatureIndex_TTest


def RankSumTestOfFeatures(arrayA, arrayB):

    NumOfFeatures = np.size(arrayA[:, 0])
    t_RankSum = np.zeros(NumOfFeatures)
    p_RankSum = np.zeros(NumOfFeatures)

    for i_feature in np.arange(NumOfFeatures):
        iA = arrayA[i_feature, :]
        iB = arrayB[i_feature, :]
        t_RankSum[i_feature], p_RankSum[i_feature] = stats.ranksums(iA, iB)
    FeatureIndex_RankSum= np.where(p_RankSum < 0.05)

    p_RankSum = p_RankSum

    return t_RankSum, p_RankSum, FeatureIndex_RankSum





def saving_stats_results(ResultName, feature_par, t, p):
    NumOfFeatures = len(t)
    with open(ResultName, 'w+') as f:
        for i in np.arange(NumOfFeatures):
            f.write(str(feature_par[i]) + ',' + str(t[i]) + ',' + str(p[i]) + '\n')

    return 0

def saving_selected_feature(ResultName, feature_par, FeatureIndex, p):

    with open(ResultName, 'w+') as f:
        f.write(str(len(FeatureIndex[0])) + '\n')
        for i_FeatureIndex in FeatureIndex[0]:
            f.write(str(i_FeatureIndex) + ',')
            f.write(str(feature_par[i_FeatureIndex]) + ',' + str(p[i_FeatureIndex]) + '\n')
    return 0



def GroupInfLoading(PatientName_Existed, Location_Of_GroupInformation):

    GroupInformation = xlrd.open_workbook(Location_Of_GroupInformation, encoding_override='utf-8')
    GroupInformationTable = GroupInformation.sheet_by_index(0)
    TableRows = GroupInformationTable.nrows
    TableCols = GroupInformationTable.ncols

    #### ASTRO #####
    #GroupFolder = ['astro', '0.2', '0.5']

    NumOfSubject = len(PatientName_Existed)
    GroupInformationArray0 = np.zeros([TableRows, TableCols - 2])
    GroupInformationArray = np.zeros([NumOfSubject, TableCols - 2])

    for i in np.arange(TableCols - 2):
        GroupInformationArray0[:, i] = GroupInformationTable.col_values(i + 2)
    PatientName_G = GroupInformationTable.col_values(0)
    if isinstance(PatientName_G[0], float):
        # PatientName_G = map(int, PatientName_G)
        PatientName_G = map(str, PatientName_G)

    GroupInfoDic = {}
    for groupway in np.arange(TableCols-2):
        for i in np.arange(TableRows):
            GroupInfoDic[PatientName_G[i].encode('utf-8')] = GroupInformationArray0[i, groupway]  # 0 astro 1 0.2 2 0.5
            # print  GroupInfoDic['SHEN_DONG_HAI']
        for i in np.arange(NumOfSubject):
            if PatientName_Existed[i] in PatientName_G:
                GroupInformationArray[i, groupway] = GroupInfoDic[PatientName_Existed[i]]

    return GroupInformationArray

def ConfidenceInterval(arr):
    num = len(arr)
    correct = np.sum(arr)
    down, up = stats.norm.interval(0.95, correct/float(num), np.std(arr)/np.sqrt(num))
    return down, up

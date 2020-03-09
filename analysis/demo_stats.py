# -*- coding: utf-8 -*-
import numpy as np
import csv
import os
from StatisticalAnalysis import ImportFeatures, TTestOfFeatures,  RankSumTestOfFeatures, saving_stats_results, saving_selected_feature
import SimpleITK as sitk

Modality = ('ADC', 'CESAG', 'CETRA', 'T2SAG', 'T2TRA',)
addr_AType = '../Features_Extracted/ATYPE/'
addr_BType = '../Features_Extracted/BTYPE/'
ResultAddress = '../StatsResults/TYPE/'

if __name__ == '__main__':
    for i_modality in Modality:

        addr_feature_A = addr_AType + i_modality + '_Features/'
        addr_feature_B = addr_BType + i_modality + '_Features/'
        result_addr = ResultAddress + i_modality + '_stats/'

        feature_arrayA0, patientA, feature_par = ImportFeatures(addr_feature_A)
        feature_arrayB0, patientB, feature_par = ImportFeatures(addr_feature_B)

        # remove outlier
        feature_arrayA = np.zeros([np.size(feature_arrayA0, 0), np.size(feature_arrayA0, 1)]) + 1e-10
        feature_arrayB = np.zeros([np.size(feature_arrayB0, 0), np.size(feature_arrayB0, 1)]) + 1e-10
        UsePositionA = np.where(abs(feature_arrayA0) > 1e-10)
        UsePositionB = np.where(abs(feature_arrayB0) > 1e-10)
        feature_arrayA[UsePositionA] = feature_arrayA0[UsePositionA]
        feature_arrayB[UsePositionB] = feature_arrayB0[UsePositionB]
        NumOfFeatures = np.size(feature_arrayA0[:, 0])
        '''
        # Welch TTest
        addr_of_ttest = result_addr + 'TTest/'
        if os.path.exists(addr_of_ttest) == 0:
            os.makedirs(addr_of_ttest)
    
        TTest_ResultName = os.path.join(addr_of_ttest,  i_modality + '.csv')
        TTest_SelName = os.path.join(addr_of_ttest, i_modality + '_selected.csv')
    
        t_TTest, p_TTest, FeatureIndex_TTest = TTestOfFeatures(feature_arrayA, feature_arrayB)
        saving_stats_results(TTest_ResultName, feature_par, t_TTest, p_TTest)
        saving_selected_feature(TTest_SelName, feature_par, FeatureIndex_TTest, p_TTest)
        '''
        # RankSum Test
        addr_of_Ranksum = result_addr + 'RanksumTest/'

        if os.path.exists(addr_of_Ranksum) == 0:
            os.makedirs(addr_of_Ranksum)

        Ranksum_ResultName = os.path.join(addr_of_Ranksum, i_modality + '.csv')
        Ranksum_SelName = os.path.join(addr_of_Ranksum, i_modality + '_selected.csv')

        t_Ranksum, p_Ranknum, FeatureIndex_Ranksum = RankSumTestOfFeatures(feature_arrayA, feature_arrayB)
        saving_stats_results(Ranksum_ResultName, feature_par, t_Ranksum, p_Ranknum)
        saving_selected_feature(Ranksum_SelName, feature_par, FeatureIndex_Ranksum, p_Ranknum)
















































        #with open(NormTest_ResultName, 'w+') as csv_file:
        #   writer = csv.writer(csv_file)
        #    writer.writerow([len(FeatureIndex_NormTest0[0]),  len(FeatureIndex_NormTest1[0])])
        #    writer.writerow([name[FeatureIndex_NormTest0[0]], name[FeatureIndex_NormTest1[0]]])
        #    for i in np.arange(NumOfFeatures):
        #        writer.writerow([name[i], t_NormTest[i, 0], p_NormTest[i, 0], t_NormTest[i, 1], p_NormTest[i, 1]])

#tp=np.hstack((t,p))

#R=DataFrame([name:tp])
#writer.writecols(tp)
#write_csv(tp,T_Result_Name)


#Various_Ratio=np.array(np.size(feature_array))
#for i in np.arange(np.size(patient_name)):
#    for j in np.arange(np.size(time_name)):
#        Various_Ratio=feature_array[:, j, i]/feature_array[:, 1, i]

#VR_Result_Name = '../Various_Ratio/' + label_type
## T_Result_Name=os.path.join('../TTest/',label_type)
#with open(VR_Result_Name, 'w+') as csv_file:
#    writer = csv.writer(csv_file)
#    for i in feature_number:
#        writer.writerow([name[i]])
#    for i in np.arange(np.size(t)):
#        writer.writerow([name[i],feature_array[i,:]])

#Feature_Mean = np.mean(feature_array, axis=2)
#Various_Mean_Ratio=np.zeros((np.size(Feature_Mean,0),np.size(Feature_Mean,1)))
#for j in np.arange(np.size(time_name)):
#    #a=np.array(Feature_Mean[:, i])
#    Various_Mean_Ratio[:,j]=Feature_Mean[:, j]/Feature_Mean[:, 0]

#print Feature_Mean[21,:]

#VMR_Result_Name='../Various_Ratio/' + label_type
#with open(VMR_Result_Name, 'w+') as csv_file:
#    writer = csv.writer(csv_file)
#    #for i in feature_number:
#    #    writer.writerow([name[i]])
#    for i in np.arange(np.size(name)):
#        a=Various_Mean_Ratio[i,:]
#        writer.writerow([name[i],a])
#        #writer.writerow([name[i], t[i], p[i]])



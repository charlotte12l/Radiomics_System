import numpy as np
import csv
import os
from analysis.StatisticalAnalysis import ImportFeatures, TTestOfFeatures, RankSumTestOfFeatures, saving_stats_results, \
    saving_selected_feature

Modality = ('ADC', 'CESAG', 'CETRA', 'T2SAG', 'T2TRA',)

class FeatureSel(object):
    def __init__(self):
        super(FeatureSel, self).__init__()

    def __call__(self, *args, **kw):
        return self.evaluate(*args, **kw)

    def evaluate(self, addr_AType,addr_BType, ResultAddress):
        for i_modality in Modality:

            addr_feature_A = addr_AType + '/'+ i_modality + '_Features/'
            addr_feature_B = addr_BType + '/'+ i_modality + '_Features/'
            result_addr = ResultAddress + '/'+ i_modality + '_stats/'

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

            # Welch TTest
            addr_of_ttest = result_addr + 'TTest/'
            if os.path.exists(addr_of_ttest) == 0:
                os.makedirs(addr_of_ttest)

            TTest_ResultName = os.path.join(addr_of_ttest,  i_modality + '.csv')
            TTest_SelName = os.path.join(addr_of_ttest, i_modality + '_selected.csv')

            t_TTest, p_TTest, FeatureIndex_TTest = TTestOfFeatures(feature_arrayA, feature_arrayB)
            saving_stats_results(TTest_ResultName, feature_par, t_TTest, p_TTest)
            saving_selected_feature(TTest_SelName, feature_par, FeatureIndex_TTest, p_TTest)

            # RankSum Test
            addr_of_Ranksum = result_addr + 'RanksumTest/'

            if os.path.exists(addr_of_Ranksum) == 0:
                os.makedirs(addr_of_Ranksum)

            Ranksum_ResultName = os.path.join(addr_of_Ranksum, i_modality + '.csv')
            Ranksum_SelName = os.path.join(addr_of_Ranksum, i_modality + '_selected.csv')

            t_Ranksum, p_Ranknum, FeatureIndex_Ranksum = RankSumTestOfFeatures(feature_arrayA, feature_arrayB)
            saving_stats_results(Ranksum_ResultName, feature_par, t_Ranksum, p_Ranknum)
            saving_selected_feature(Ranksum_SelName, feature_par, FeatureIndex_Ranksum, p_Ranknum)

        return True


from FeatureExtraction import FeatureExtraction

Param = '../Params.yaml'
addr_A = '../data/ATYPE/'
addr_B = '../data/BTYPE/'
Selector = ("ADC", "CESAG", "CETRA", "T2SAG", "T2TRA")
sav_folder_A = '../Features_Extracted/ATYPE/'
sav_folder_B = '../Features_Extracted/BTYPE/'

if __name__ == '__main__':

    for i_sel in Selector:
        FeatureExtraction(Param, addr_A, i_sel, sav_folder_A, 1)
        FeatureExtraction(Param, addr_B, i_sel, sav_folder_B, 2)

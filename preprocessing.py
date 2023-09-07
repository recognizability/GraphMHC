import pandas as pd
import numpy as np
seed = 42
np.random.seed(seed)

raw = pd.read_csv('bdata.20130222.mhci.txt', sep='\t')
iedb_hla_peptide = raw.loc[raw['species']=='human', ['mhc', 'sequence', 'meas']]
iedb_hla_peptide['binding'] = (iedb_hla_peptide['meas']<=500)*1
iedb_hla_peptide['hla_wo_asterisk'] = iedb_hla_peptide['mhc'].apply(lambda string: string.replace('*', ''))
hla_mhc = pd.read_csv('MHC_pseudo.dat', delim_whitespace=True, header=None)
hla_mhc.columns = ['hla', 'mhc_sequence']
mhc_peptide = pd.merge(iedb_hla_peptide, hla_mhc, left_on='hla_wo_asterisk', right_on='hla', how='left')
mhc_peptide_not_na = mhc_peptide[~(mhc_peptide['mhc_sequence'].isna())]

split = 0.8
indices = np.random.permutation(len(mhc_peptide_not_na)) #shuffle indices
trainset = mhc_peptide_not_na.iloc[indices[:int(len(mhc_peptide_not_na)*split)]]
testset = mhc_peptide_not_na.iloc[indices[int(len(mhc_peptide_not_na)*split):]]
trainset.to_csv('iedb_sequences/raw/iedb_trainset.csv', index=False)
testset.to_csv('iedb_sequences/raw/iedb_testset.csv', index=False)

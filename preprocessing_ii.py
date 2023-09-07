from glob import glob
import pandas as pd
import os

path = './iedb_ii/raw/'
if not os.path.isdir(path):
    os.makedirs(path)

for dataset in ['train', 'test']:
    hla_ii = pd.DataFrame()
    for file in glob(f'class_II_similarity_reduced_5cv_sep/HLA*{dataset}*'):
        hla_ii = pd.concat([hla_ii, pd.read_csv(file, delim_whitespace=True, header=None)[[1,4,6]]])
    hla_ii = hla_ii.reset_index(drop=True)
    hla_ii.columns = ['hla_iedb', 'peptide', 'ic50']
    hla_ii['hla'] = hla_ii['hla_iedb'].str.replace('*', '')
    hla_ii['hla'] = hla_ii['hla'].str.replace('/', '-')
    hla_ii['hla'] = hla_ii['hla'].str.replace('HLA-', '')

    pseudosequences = pd.read_csv('mhc_ii/methods/netmhciipan-4.0-executable/netmhciipan_4_0_executable/data/pseudosequence.2016.all.X.dat', delim_whitespace=True, header=None)
    pseudosequences.columns = ['hla_netmhc', 'mhc']
    pseudosequences['hla'] = pseudosequences['hla_netmhc'].str.replace('_', '')
    pseudosequences['hla'] = pseudosequences['hla'].str.replace('HLA-', '')

    joined = pd.merge(hla_ii, pseudosequences, how='left', on='hla')
    joined = joined.dropna() #Remove some unmatched HLAs
    joined = joined[~(joined['mhc'].str.contains('X'))] #Remove if contains X amino acid

    joined['binding'] = (joined['ic50']<=500)*1
    joined.to_csv(f'iedb_ii/raw/iedb_ii_{dataset}.csv', index=False)

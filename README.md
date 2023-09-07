# GraphMHC
Neoantigen prediction model applying the graph neural network to molecular structure from amino acid sequences of MHC proteins and peptides

## Requirements
* python=3.9
* pandas=1.4.4
* numpy=1.23.2
* rdkit=2022.03.2
* torch=1.11.0+cu113
* torch_geometric=2.1.0
* sklearn=1.1.2

## IEDB I
```
wget http://tools.iedb.org/static/main/binding_data_2013.zip
unzip binding_data_2013.zip
```
```
python preprocessing.py
```
```
python graphmhc.py
```

## IEDB II
```
wget http://tools.iedb.org/static/download/classII_binding_data_Nov_16_2009.tar.gz
tar xvzf classII_binding_data_Nov_16_2009.tar.gz
```
```
wget https://downloads.iedb.org/tools/mhcii/3.1.6/IEDB_MHC_II-3.1.6.tar.gz
```

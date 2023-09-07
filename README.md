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

## Apply to IEDB class I data
* Download data:
  ```
  wget http://tools.iedb.org/static/main/binding_data_2013.zip
  unzip binding_data_2013.zip
  wget https://downloads.iedb.org/tools/mhci/3.1.2/IEDB_MHC_I-3.1.2.tar.gz
  tar xvzf IEDB_MHC_I-3.1.2.tar.gz
  ```
* Preprocessing:
  ```
  python preprocessing_i.py
  ```
* Apply GraphMHC model:
  ```
  python graphmhc.py --root iedb_i --train iedb_trainset.csv --test iedb_testset.csv --mhc mhc_sequence --peptide sequence --binding binding
  ```

## Apply to IEDB class II data
* Download data:
  ```
  wget http://tools.iedb.org/static/download/classII_binding_data_Nov_16_2009.tar.gz
  tar xvzf classII_binding_data_Nov_16_2009.tar.gz
  wget https://downloads.iedb.org/tools/mhcii/3.1.6/IEDB_MHC_II-3.1.6.tar.gz
  tar xvzf IEDB_MHC_II-3.1.6.tar.gz
  ``` 
* Preprocessing:
  ```
  python preprocessing_ii.py
  ```
* Apply GraphMHC model:
  ```
  python graphmhc.py --root iedb_ii --train iedb_ii_train.csv --test iedb_ii_test.csv --mhc mhc --peptide peptide --binding binding
  ```
## Apply to the other binding dataset between amino acid sequences
```
python graphmhc.py --root <root directory> --train <trainset file> --test <testset file> --mhc <MHC field> --peptide <peptide field> --binding <binding affinity field>
```

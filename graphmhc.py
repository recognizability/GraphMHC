import os
import numpy as np 
import pandas as pd
from rdkit import Chem 
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from argparse import ArgumentParser
from datetime import datetime
from sklearn import metrics

parser = ArgumentParser()
parser.add_argument('--root', help='Root directory')
parser.add_argument('--train', help='File of traning dataset')
parser.add_argument('--test', help='File of test dataset')
parser.add_argument('--mhc', help='MHC field')
parser.add_argument('--peptide', help='Peptide field')
parser.add_argument('--binding', help='Binding affinity field')
args = parser.parse_args()

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if device == 'cuda:0':
    torch.cuda.manual_seed_all(seed)
    
class MoleculeDataset(Dataset): #Inherited from Dataset
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None):
        self.test = test
        self.filename = filename
        super(MoleculeDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return self.filename

    @property
    def processed_file_names(self): #Check if file exists
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        if self.test:
            return [f'testset_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'trainset_{i}.pt' for i in list(self.data.index)]

    def download(self):
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0])
        for index, sequences in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            mol = Chem.MolFromSmiles(
                '.'.join( #. is non-bond
                    Chem.MolToSmiles(
                        Chem.MolFromSequence(sequences[sequence])
                    ) for sequence in [args.mhc, args.peptide] #Corresponds to columns of MHC sequence and peptide sequence, respectively
                )
            )
            mol = Chem.AddHs(mol)
            x = self._get_node_features(mol)
            edge_attr, edge_index, edge_weight = self._get_edge_features(mol)
            y = self._get_labels(sequences[args.binding]) #Column corresponding to binding information coded as 0 or 1
            data = Data(x=x, edge_attr=edge_attr, edge_index=edge_index, edge_weight=edge_weight, y=y)
            
            if self.test:
                torch.save(data, os.path.join(self.processed_dir, f'testset_{index}.pt'))
            else:
                torch.save(data, os.path.join(self.processed_dir, f'trainset_{index}.pt'))

    def _get_node_features(self, mol):
        all_node_feats = []

        for atom in mol.GetAtoms(): #about atoms
            symbol = atom.GetSymbol() #atomic symbol
            valid_atoms = {'H': 0, 'C':1, 'N':2, 'O':3, 'S':4}
            atom_one_hot = [0] * len(valid_atoms)
            try:
                idx = valid_atoms[symbol]
                atom_one_hot[idx] = 1
            except:
                pass

            hybrid = atom.GetHybridization()
            hybrid_one_hot = [0] * 7
            if hybrid == Chem.HybridizationType.SP3:
                hybrid_one_hot[0] = 1
            elif hybrid == Chem.HybridizationType.SP2:
                hybrid_one_hot[1] = 1
            elif hybrid == Chem.HybridizationType.SP:
                hybrid_one_hot[2] = 1
            elif hybrid == Chem.HybridizationType.S:
                hybrid_one_hot[3] = 1
            elif hybrid == Chem.HybridizationType.SP3D:
                hybrid_one_hot[4] = 1
            elif hybrid == Chem.HybridizationType.SP3D2:
                hybrid_one_hot[5] = 1
            else:
                hybrid_one_hot[6] = 1

            degree = atom.GetDegree() #number of covalent bonds
            degree_one_hot = [0, 0, 0, 0, 0, 0]
            if degree >= 5: #Atoms with 5 or more covalent bonds
                degree_one_hot[5]=1
            else:
                degree_one_hot[degree]=1

            num_h = atom.GetTotalNumHs() #Number of bonded hydrogen atoms
            h_one_hot = [0, 0, 0, 0, 0]
            if num_h >= 4:
                h_one_hot[4] = 1
            else:
                h_one_hot[num_h] = 1

            chiral = atom.GetChiralTag() #chirality
            if chiral == Chem.rdchem.ChiralType.CHI_OTHER:
                chiral_one_hot = [1, 0, 0, 0]
            elif chiral == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW:
                chiral_one_hot = [0, 1, 0, 0]
            elif chiral == Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW:
                chiral_one_hot = [0, 0, 1, 0]
            elif chiral == Chem.rdchem.ChiralType.CHI_UNSPECIFIED:
                chiral_one_hot = [0, 0, 0, 1]

            aromatic = 1 if atom.GetIsAromatic() else 0 #aromaticity
            ring = 1 if atom.IsInRing() else 0 #Inclusion in the ring

            node_attr = atom_one_hot + hybrid_one_hot + degree_one_hot + h_one_hot + chiral_one_hot + [aromatic, ring, atom.GetFormalCharge(), atom.GetNumRadicalElectrons()]
            all_node_feats.append(node_attr)
            
        all_node_feats = np.asarray(all_node_feats)
        return torch.tensor(all_node_feats, dtype=torch.float)

    def _get_edge_features(self, mol):
        all_edge_feats = []
        edge_indices = []
        edge_weights = []
        for bond in mol.GetBonds(): #about covalent bonds
            bond_type = bond.GetBondType() #Types of Covalent Bonds
            if bond_type == Chem.rdchem.BondType.SINGLE:
                bond_one_hot = [1, 0, 0, 0]
                edge_weights.extend([1.0, 1.0])
            elif bond_type == Chem.rdchem.BondType.AROMATIC:
                bond_one_hot = [0, 1, 0, 0]
                edge_weights.extend([1.5, 1.5])
            elif bond_type == Chem.rdchem.BondType.DOUBLE:
                bond_one_hot = [0, 0, 1, 0]
                edge_weights.extend([2.0, 2.0])
            elif bond_type == Chem.rdchem.BondType.TRIPLE:
                bond_one_hot = [0, 0, 0, 1]   
                edge_weights.extend([3.0, 3.0])  

            stereo_type = bond.GetStereo()
            if stereo_type == Chem.rdchem.BondStereo.STEREOANY:
                stereo_one_hot = [1, 0, 0, 0, 0, 0]
            elif stereo_type == Chem.rdchem.BondStereo.STEREOCIS:
                stereo_one_hot = [0, 1, 0, 0, 0, 0]
            elif stereo_type == Chem.rdchem.BondStereo.STEREOE:
                stereo_one_hot = [0, 0, 1, 0, 0, 0]
            elif stereo_type == Chem.rdchem.BondStereo.STEREONONE:
                stereo_one_hot = [0, 0, 0, 1, 0, 0]
            elif stereo_type == Chem.rdchem.BondStereo.STEREOTRANS:
                stereo_one_hot = [0, 0, 0, 0, 1, 0]
            elif stereo_type == Chem.rdchem.BondStereo.STEREOZ:
                stereo_one_hot = [0, 0, 0, 0, 0, 1]

            ring_bond = 1 if bond.IsInRing() else 0 #In-ring or not
            conjugate_bond = 1 if bond.GetIsConjugated() else 0

            edge_attr = bond_one_hot + stereo_one_hot + [ring_bond, conjugate_bond]
            all_edge_feats.append(edge_attr) #Since it is an undirected graph, it is added twice.
            all_edge_feats.append(edge_attr)
            
            i = bond.GetBeginAtomIdx() #index of starting atom
            j = bond.GetEndAtomIdx() #index of endpoint atom
            edge_indices += [[i, j], [j, i]]

        all_edge_feats = np.asarray(all_edge_feats)
        all_edge_feats = torch.tensor(all_edge_feats, dtype=torch.float)
        
        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)
        
        edge_weights = torch.tensor(edge_weights, dtype = torch.float)
        
        return all_edge_feats, edge_indices, edge_weights

    def _get_labels(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, f'testset_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, f'trainset_{idx}.pt'))   
        return data

trainset = MoleculeDataset(root=args.root, filename=args.train) #The file must be under the raw directory under the root directory
testset = MoleculeDataset(root=args.root, filename=args.test, test=True)

batch_size = 64
#batch_size = 128
#batch_size = 256
#batch_size = 512
#batch_size = 1024
#batch_size = 2048
#batch_size = 4096

train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=False)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, drop_last=False)

import torch.nn.functional as F #To use an activation function or dropout
from torch.nn import MultiheadAttention, Conv1d, BatchNorm1d, MaxPool1d, Conv2d, BatchNorm2d, MaxPool2d, AvgPool1d
from torch.nn import Linear, ModuleList #The reason for defining the module list is to enable learning
from torch_geometric.nn import GATConv, TransformerConv, SAGPooling, TopKPooling, GraphNorm, BatchNorm
from torch_geometric.nn import global_mean_pool as gmp #Mean value of node feature vectors. Convert graph matrices to vectors (with permutation invariance). Used for readout. Readout is used only for graph classification and is the process of creating the hidden state of the graph by adding the hidden states of all nodes.
#from torch_geometric.nn import global_max_pool as gmp #Maximum of node feature vectors

class GraphMHC(torch.nn.Module): #Inherited from nn.Module
    def __init__(self, hyperparameters):
        super(GraphMHC, self).__init__()
        in_channels= hyperparameters["in_channels"]
        channels= hyperparameters["channels"]
        heads = hyperparameters["heads"]
        self.heads = heads
        dropout_rate = hyperparameters["dropout_rate"]
        self.dropout_rate = dropout_rate
        edge_dim = hyperparameters["edge_dim"]
        kernel_size = hyperparameters["kernel_size"]

        self.conv1 = GATConv(in_channels, channels, heads=heads, dropout=dropout_rate, edge_dim=edge_dim, concat=False) 
        self.norm1 = GraphNorm(channels)
        
        self.conv2 = GATConv(channels, channels, heads=heads, dropout=dropout_rate, edge_dim=edge_dim, concat=False) 
        self.norm2 = GraphNorm(channels)
        
        self.conv3 = GATConv(channels, channels, heads=heads, dropout=dropout_rate, edge_dim=edge_dim, concat=False) 
        self.norm3 = GraphNorm(channels)
        
        self.conv4 = GATConv(channels, channels, heads=heads, dropout=dropout_rate, edge_dim=edge_dim, concat=False) 
        self.norm4 = GraphNorm(channels)
        
        self.conv5 = Conv1d(heads, heads*2, kernel_size=kernel_size, padding='same')
        self.norm5 = BatchNorm1d(heads*2)
        self.pool5 = AvgPool1d(2)
        
        self.conv6 = Conv1d(heads*2, heads*4, kernel_size=kernel_size, padding='same')
        self.norm6 = BatchNorm1d(heads*4)
        self.pool6 = AvgPool1d(2)
        
        self.lin7 = Linear(channels, 1)
        
    def forward(self, x, edge_attr, edge_index, edge_weight, batch_index):
        x = self.conv1(x, edge_index, edge_attr)
        x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout_rate)
        
        x = self.conv2(x, edge_index, edge_attr)
        x = self.norm2(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout_rate)
        
        x = self.conv3(x, edge_index, edge_attr)
        x = self.norm3(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout_rate)
        
        x = self.conv4(x, edge_index, edge_attr)
        x = self.norm4(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout_rate)
        
        x = gmp(x, batch_index)

        skip4 = x
        x = torch.reshape(x, (x.shape[0], self.heads, x.shape[1]//self.heads)) #put in as many channels as the number of heads
        
        x = self.conv5(x)
        x = self.norm5(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout_rate)
        x = self.pool5(x)
        skip5 = x.flatten(start_dim=1, end_dim=2)
                
        x = self.conv6(x)
        x = self.norm6(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=self.dropout_rate)
        x = self.pool6(x)
        
        x = x.flatten(start_dim=1, end_dim=2)
        x += skip4 + skip5 #skip connection
        
        x = self.lin7(x)
        
        return x

def evaluate(labels, predictions, mode):
    roc_auc = metrics.roc_auc_score(labels, predictions)
    print(mode, 'ROC AUC:', roc_auc)
    fpr, tpr, thresholds = metrics.roc_curve(labels, predictions)
    optimal_threshold = thresholds[np.argmax(tpr-fpr)]
    casted = np.array([1 if element >= optimal_threshold else 0 for element in predictions])
    print(mode, 'F1-score:', metrics.f1_score(labels, casted))
    print(mode, 'Precision:', metrics.precision_score(labels, casted))
    print(mode, 'Recall (sensitivity):', metrics.recall_score(labels, casted))
    precision, recall, thresholds = metrics.precision_recall_curve(labels, predictions)
    pr_auc = metrics.auc(recall, precision)
    print(mode, 'PR AUC', pr_auc)
    print(mode, 'Accuracy:', metrics.accuracy_score(labels, casted))
    print(mode, 'Balanced accuracy:', metrics.balanced_accuracy_score(labels, casted))
    print(mode, 'Optimal threshold:', optimal_threshold)
    return f'{roc_auc:.3f}'

def train(epoch, model, train_loader, optimizer, criterion):
    model.train() #training mode
    labels = []
    predictions = []
    first = False
    for batch in tqdm(train_loader):
        batch.to(device)  
        labels.append(batch.y.float().cpu().detach().numpy())
        optimizer.zero_grad() #gradient initialization
        prediction = model(batch.x.float(), batch.edge_attr.float(), batch.edge_index, batch.edge_weight, batch.batch,) 
        loss = criterion(torch.squeeze(prediction), batch.y.float())
        loss.backward() #differentiation and backpropagation
        optimizer.step() #update weights
        predictions.append(torch.sigmoid(prediction).cpu().detach().numpy()) #Apply sigmoid
    labels = np.concatenate(labels).ravel()
    predictions = np.concatenate(predictions).ravel()
    evaluate(labels, predictions, 'Train,')

def test(epoch, model, test_loader, criterion):
    model.eval() #inference mode
    labels = []
    predictions = []
    for batch in test_loader:  
        batch.to(device) 
        labels.append(batch.y.float().cpu().detach().numpy())
        prediction = model(batch.x.float(), batch.edge_attr.float(), batch.edge_index, batch.edge_weight, batch.batch,)
        predictions.append(torch.sigmoid(prediction).cpu().detach().numpy()) #Apply sigmoid
    labels = np.concatenate(labels).ravel()
    predictions = np.concatenate(predictions).ravel()
    return evaluate(labels, predictions, 'Test,')

hyperparameters = {
    "in_channels": trainset[0].x.shape[1],
    "channels": 256,
    "heads": 8,
    "dropout_rate": 0.1,
    "edge_dim": trainset[0].edge_attr.shape[1],
    "kernel_size": 9,
}

model = GraphMHC(hyperparameters=hyperparameters)
model = model.to(device)

trainset_raw = pd.read_csv(args.root + 'raw' + args.train)
pos_weight = (trainset_raw[args.binding]==0).sum()/trainset_raw[args.binding].sum()
pos_weight = torch.tensor([pos_weight], dtype=torch.float).to(device)

optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight) #Binary cross entropy loss with logit applied

for epoch in range(100): 
    print('Epoch', epoch)
    train(epoch, model, train_loader, optimizer, criterion)
    with torch.no_grad():
        roc_auc = test(epoch, model, test_loader, criterion)
    now = datetime.now().strftime('%Y%m%dT%H%M%S')
    filename = f'{now}_model_{roc_auc}.pt'
    torch.save(model, filename)
    print()

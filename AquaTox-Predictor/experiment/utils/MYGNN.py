import datetime
import tensorflow as tf
from sklearn.metrics import roc_auc_score, mean_squared_error, precision_recall_curve, auc, r2_score, recall_score,accuracy_score
import torch
import torch.nn.functional as F
import dgl
import numpy as np
import random
from dgl.readout import sum_nodes
from dgl.nn.pytorch.conv import RelGraphConv
from torch import nn
import pandas as pd
from utils import weight_visualization
import pandas as pd
from rdkit import Chem , DataStructs
from rdkit.Chem import AllChem, MACCSkeys, rdFingerprintGenerator
from rdkit.Chem import Descriptors
from dgl.nn import GATConv
from dgl.nn.pytorch import GlobalAttentionPooling, SumPooling


class WeightAndSum(nn.Module):
    def __init__(self, in_feats, task_num=1, attention=True, return_weight=False):
        super(WeightAndSum, self).__init__()
        self.attention = attention
        self.in_feats = in_feats
        self.task_num = task_num
        self.return_weight=return_weight
        self.atom_weighting_specific = nn.ModuleList([self.atom_weight(self.in_feats) for _ in range(self.task_num)])
        self.shared_weighting = self.atom_weight(self.in_feats)
    def forward(self, bg, feats):
        feat_list = []
        atom_list = []

        for i in range(self.task_num):
            with bg.local_scope():
                bg.ndata['h'] = feats
                weight = self.atom_weighting_specific[i](feats)
                bg.ndata['w'] = weight
                specific_feats_sum = sum_nodes(bg, 'h', 'w')
                atom_list.append(bg.ndata['w'])
            feat_list.append(specific_feats_sum)


        with bg.local_scope():
            bg.ndata['h'] = feats
            bg.ndata['w'] = self.shared_weighting(feats)
            shared_feats_sum = sum_nodes(bg, 'h', 'w')

        if self.attention:
            if self.return_weight:
                return feat_list, atom_list
            else:
                return feat_list
        else:
            return shared_feats_sum

    def atom_weight(self, in_feats):
        return nn.Sequential(
            nn.Linear(in_feats, 1),
            nn.Sigmoid()
            )
    

class RGCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, num_rels=64*21, activation=F.relu, loop=False,
                 residual=True, batchnorm=True, rgcn_drop_out=0.5):
        super(RGCNLayer, self).__init__()

        self.activation = activation
        self.graph_conv_layer = RelGraphConv(in_feats, out_feats, num_rels=num_rels, regularizer='basis',
                                               num_bases=None, bias=True, activation=activation,
                                               self_loop=loop, dropout=rgcn_drop_out)
        self.residual = residual
        if residual:
            self.res_connection = nn.Linear(in_feats, out_feats)

        self.bn = batchnorm
        if batchnorm:
            self.bn_layer = nn.BatchNorm1d(out_feats)

    def forward(self, bg, node_feats, etype, norm=None):
        new_feats = self.graph_conv_layer(bg, node_feats, etype, norm)
        if self.residual:
            res_feats = self.activation(self.res_connection(node_feats))
            new_feats = new_feats + res_feats
        if self.bn:
            new_feats = self.bn_layer(new_feats)
        del res_feats
        torch.cuda.empty_cache()
        return new_feats

def generate_fingerprints(data, fp_type, device):
    fingerprints_list = []
    
    for smiles in data:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
        
        fingerprints = []
        
        if 'morgan' in fp_type:
            fpgen = AllChem.GetMorganGenerator(radius=2)
            morgan_fp = fpgen.GetFingerprint(mol)
            fingerprints.extend([int(bit) for bit in morgan_fp.ToBitString()])
        else:
            fingerprints.extend([0] * 2048) 
        if 'maccs' in fp_type:
            maccs_fp = MACCSkeys.GenMACCSKeys(mol)
            fingerprints.extend([int(bit) for bit in maccs_fp.ToBitString()])
        else:
            fingerprints.extend([0] * 167) 
 
        if 'rdkit' in fp_type:
            fpgen = AllChem.GetRDKitFPGenerator()
            rdkit_fp = fpgen.GetFingerprint(mol)
            fingerprints.extend([int(bit) for bit in rdkit_fp.ToBitString()])
        else:
            fingerprints.extend([0] * 2048) 
        
        combined_fingerprints = torch.tensor(fingerprints, dtype=torch.float32).to(device)
        fingerprints_list.append(combined_fingerprints)
        fp = torch.stack(fingerprints_list, dim=0)
    return fp


class FingerPrintEncoder(nn.Module):
    def __init__(self, emb_dim, drop_ratio, fp_type, device):
        super(FingerPrintEncoder, self).__init__()
        self.fp_type = fp_type
        self.device = device
        morgan_dim = 2048 if 'morgan' in fp_type else 0
        maccs_dim = 167 if 'maccs' in fp_type else 0
        rdit_dim = 2048 if 'rdkit' in fp_type else 0

        init_dim = morgan_dim + maccs_dim + rdit_dim

        self.fc1 = nn.Linear(init_dim, emb_dim).to(device)
        self.batch_norm = nn.BatchNorm1d(emb_dim).to(device)
        self.act_func = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_ratio)
        self.fc2 = nn.Sequential(
            nn.Linear(init_dim, 512).to(device),
            nn.Dropout(),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.init_emb()
        self.sigmoid = nn.Sigmoid()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, smiles, fp_type, device):
        data = generate_fingerprints(smiles, fp_type, device)

        fps_rep = self.act_func(self.batch_norm(self.fc2(data.float())))

        return fps_rep

class FingerPrintEncoderx(nn.Module):
    def __init__(self, emb_dim, drop_ratio, fp_type, device):
        super(FingerPrintEncoderx, self).__init__()
        self.fp_type = fp_type
        self.device = device
        morgan_dim = 2048 if 'morgan' in fp_type else 0
        maccs_dim = 167 if 'maccs' in fp_type else 0
        rdit_dim = 2048 if 'rdkit' in fp_type else 0

        init_dim = morgan_dim + maccs_dim + rdit_dim

        self.fc1 = nn.Linear(init_dim, emb_dim).to(device)
        self.batch_norm = nn.BatchNorm1d(emb_dim).to(device)
        self.act_func = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_ratio)
        self.fc2 = nn.Sequential(
            nn.Linear(init_dim, 512).to(device),
            nn.Dropout(),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        self.init_emb()
        self.sigmoid = nn.Sigmoid()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, smiles, fp_type, device):
        data = generate_fingerprints(smiles, fp_type, device)

        fps_rep = self.act_func(self.batch_norm(self.fc2(data.float())))

        return fps_rep

class BaseGNN(nn.Module):
    def __init__(self, gnn_out_feats, n_tasks, rgcn_drop_out=0.5, return_mol_embedding=False, return_weight=False,
                 classifier_hidden_feats=128, dropout=0.):
        super(BaseGNN, self).__init__()
        self.task_num = n_tasks
        self.gnn_layers = nn.ModuleList()
        self.return_weight = return_weight
        self.weighted_sum_readout = WeightAndSum(gnn_out_feats, self.task_num, return_weight=self.return_weight)
        self.fc_in_feats = gnn_out_feats
        self.return_mol_embedding=return_mol_embedding

        self.fc_layers1 = nn.ModuleList([self.fc_layer(dropout, self.fc_in_feats, classifier_hidden_feats) for _ in range(self.task_num)])
        self.fc_layers2 = nn.ModuleList(
            [self.fc_layer(dropout, classifier_hidden_feats, classifier_hidden_feats) for _ in range(self.task_num)])
        self.fc_layers3 = nn.ModuleList(
            [self.fc_layer(dropout, classifier_hidden_feats, classifier_hidden_feats) for _ in range(self.task_num)])

        self.output_layer1 = nn.ModuleList(
            [self.output_layer(classifier_hidden_feats, 1) for _ in range(self.task_num)])

    def forward(self, bg, node_feats, etype, norm=None):
        for gnn in self.gnn_layers:
            node_feats = gnn(bg, node_feats, etype, norm)

        if self.return_weight:
            feats_list, atom_weight_list = self.weighted_sum_readout(bg, node_feats)
        else:
            feats_list = self.weighted_sum_readout(bg, node_feats)

        for i in range(self.task_num):
            mol_feats = feats_list[i]
            h1 = self.fc_layers1[i](mol_feats)
            h2 = self.fc_layers2[i](h1)
            h3 = self.fc_layers3[i](h2)
            predict = self.output_layer1[i](h3)
            if i == 0:
                prediction_all = predict
            else:
                prediction_all = torch.cat([prediction_all, predict], dim=1)

        if self.return_mol_embedding:
            return feats_list[0]
        else:

            if self.return_weight:
                return prediction_all, atom_weight_list, node_feats

            return prediction_all

    def fc_layer(self, dropout, in_feats, hidden_feats):
        return nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_feats, hidden_feats),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_feats)
                )

    def output_layer(self, hidden_feats, out_feats):
        return nn.Sequential(
                nn.Linear(hidden_feats, out_feats)
                )
    
    
class GraphKnowledgeExtractor(BaseGNN):
    def __init__(self, in_feats, rgcn_hidden_feats, n_tasks, return_weight=False,
                 classifier_hidden_feats=128, loop=False, return_mol_embedding=False,
                 rgcn_drop_out=0.5, dropout=0.):
        super(GraphKnowledgeExtractor, self).__init__(gnn_out_feats=rgcn_hidden_feats[-1],
                                             n_tasks=n_tasks,
                                             classifier_hidden_feats=classifier_hidden_feats,
                                             return_mol_embedding=return_mol_embedding,
                                             return_weight=return_weight,
                                             rgcn_drop_out=rgcn_drop_out,
                                             dropout=dropout)
        
        for i in range(len(rgcn_hidden_feats)):
            out_feats = rgcn_hidden_feats[i]
            self.gnn_layers.append(RGCNLayer(in_feats, out_feats, loop=loop, rgcn_drop_out=rgcn_drop_out))
            in_feats = out_feats
        
        gate_nn_attention = torch.nn.Linear(256, 1)
        self.pooling_layer = GlobalAttentionPooling(gate_nn_attention)
    def forward(self, bg, node_feats, etype, norm=None):
        
        for gnn in self.gnn_layers:
            node_feats = gnn(bg, node_feats, etype, norm)
        node_feats, attention = self.pooling_layer(bg, node_feats, get_attention=True)

        return node_feats, attention

class MGA(BaseGNN):
    def __init__(self, in_feats, rgcn_hidden_feats, n_tasks, return_weight=False,
                 classifier_hidden_feats=128, loop=False, return_mol_embedding=False,
                 rgcn_drop_out=0.5, dropout=0.):
        super(MGA, self).__init__(gnn_out_feats=rgcn_hidden_feats[-1],
                                  n_tasks=n_tasks,
                                  classifier_hidden_feats=classifier_hidden_feats,
                                  return_mol_embedding=return_mol_embedding,
                                  return_weight=return_weight,
                                  rgcn_drop_out=rgcn_drop_out,
                                  dropout=dropout,
                                  )

        for i in range(len(rgcn_hidden_feats)):
            out_feats = rgcn_hidden_feats[i]
            self.gnn_layers.append(RGCNLayer(in_feats, out_feats, loop=loop, rgcn_drop_out=rgcn_drop_out))
            in_feats = out_feats


class AquaToxPredictor(nn.Module):
    def __init__(self, rgcn_in_feats, rgcn_hidden_feats, num_rels , num_experts, expert_hidden_dim,
                num_tasks, task_hidden_dim, fp_dim, drop_ratio, fp_type, device):
        super(AquaToxPredictor, self).__init__()
        

        self.rgcns = GraphKnowledgeExtractor(in_feats=rgcn_in_feats, rgcn_hidden_feats=rgcn_hidden_feats, n_tasks=num_tasks, return_weight=False,)

        self.mol_fp_encoder = FingerPrintEncoder(emb_dim=256, drop_ratio=drop_ratio, fp_type=fp_type, device=device)

        self.mmoe = MMOE(input_dim=rgcn_hidden_feats[-1] + fp_dim, num_experts=num_experts, expert_hidden_dim=expert_hidden_dim,
                        num_tasks=num_tasks, task_hidden_dim=task_hidden_dim)
        
    def forward(self, bg, atom_feats, etype, smiles, fp_type ,device, norm=None ):
        rgcn_output, attention_weights = self.rgcns(bg, atom_feats, etype, norm)
        fp_output = self.mol_fp_encoder(smiles,fp_type,device)
        combined_output = torch.cat((rgcn_output, fp_output), dim=1)
        mmoe_output,gate_weight = self.mmoe(combined_output)
        return mmoe_output, attention_weights, gate_weight 




class fpMMOE(nn.Module):
    def __init__(self, num_experts, expert_hidden_dim,
                 num_tasks, task_hidden_dim, drop_ratio, fp_type, device):
        super(fpMMOE, self).__init__()

        self.mol_fp_encoder = FingerPrintEncoderx(emb_dim=512, drop_ratio=drop_ratio, fp_type=fp_type, device=device)

        self.mmoe = MMOE(input_dim=512, num_experts=num_experts, expert_hidden_dim=expert_hidden_dim,
                         num_tasks=num_tasks, task_hidden_dim=task_hidden_dim)
        
    def forward(self, bg, atom_feats, etype, smiles, fp_type ,device, norm=None ):
        
        fp_output = self.mol_fp_encoder(smiles,fp_type,device)

        mmoe_output = self.mmoe(fp_output)
        return mmoe_output
    

class AquaToxPredictorNoMMOE(nn.Module):
    def __init__(self, rgcn_in_feats, rgcn_hidden_feats, shared_hidden_dim,
                 num_tasks, fp_dim, drop_ratio, fp_type, device):
        super(AquaToxPredictorNoMMOE, self).__init__()
        

        self.rgcns = GraphKnowledgeExtractor(in_feats=rgcn_in_feats, rgcn_hidden_feats=rgcn_hidden_feats, n_tasks=num_tasks, return_weight=False,)

        self.mol_fp_encoder = FingerPrintEncoder(emb_dim=256, drop_ratio=drop_ratio, fp_type=fp_type, device=device)

        combined_dim = rgcn_hidden_feats[-1] + fp_dim

        self.shared_layer = nn.Sequential(
        nn.Linear(combined_dim, shared_hidden_dim),
        nn.ReLU(),
        nn.Linear(shared_hidden_dim, shared_hidden_dim),
        nn.ReLU()
        )

        self.task_layers = nn.ModuleList([nn.Linear(shared_hidden_dim, 1) for _ in range(num_tasks)]) 
    def forward(self, bg, atom_feats, etype, smiles, fp_type ,device, norm=None):

        rgcn_output = self.rgcns(bg, atom_feats, etype, norm)
        
        fp_output = self.mol_fp_encoder(smiles, fp_type, device)

        combined_output = torch.cat((rgcn_output, fp_output), dim=1)
        

        shared_output = self.shared_layer(combined_output)


        final_outputs = [task_layer(shared_output) for task_layer in self.task_layers]

        final_outputs_tensor = torch.cat(final_outputs, dim=1) 

        return final_outputs_tensor 


class MMOE(nn.Module):
    def __init__(self, input_dim, num_experts, expert_hidden_dim, num_tasks, task_hidden_dim, num_shared_units=128):
        super(MMOE, self).__init__()

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_hidden_dim),
                nn.ReLU(),
                nn.Linear(expert_hidden_dim, expert_hidden_dim),
                nn.ReLU()
            ) for _ in range(num_experts)
        ])

        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, num_experts),
                nn.Softmax(dim=-1)
            ) for _ in range(num_tasks)
        ])


        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_hidden_dim, task_hidden_dim),
                nn.ReLU(),
                nn.Linear(task_hidden_dim, 1),
                nn.Sigmoid() 
            ) for _ in range(num_tasks)
        ])
    def forward(self, x, threshold=0.5): 

            expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
            task_outputs = []
            gate_weights_list = []
            for gate, tower in zip(self.gates, self.towers):
                gate_weights = gate(x) 
                task_expert_output = torch.einsum('be,beh->bh', gate_weights, expert_outputs) 
                task_output = tower(task_expert_output) 
                task_outputs.append(task_output)
                gate_weights_list.append(gate_weights)
            
            gate_weight = torch.stack(gate_weights_list, dim=1) 
     
            output = torch.cat(task_outputs, dim=1) 
            return  output , gate_weight 
 

def set_random_seed(seed=10):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def pos_weight(train_set, classification_num):
    smiles, graphs, labels, mask = map(list, zip(*train_set))
    labels = np.array(labels)
    task_pos_weight_list = []
    for task in range(classification_num):
        num_pos = 0
        num_impos = 0
        for i in labels[:, task]:
            if i == 1:
                num_pos = num_pos + 1
            if i == 0:
                num_impos = num_impos + 1
        weight = num_impos / (num_pos+0.00000001)
        task_pos_weight_list.append(weight)
    task_pos_weight = torch.tensor(task_pos_weight_list)
    return task_pos_weight


class Meter(object):
    """Track and summarize model performance on a dataset for
    (multi-label) binary classification."""
    def __init__(self):
        self.mask = []
        self.y_pred = []
        self.y_true = []

    def update(self, y_pred, y_true, mask):
        """Update for the result of an iteration
        Parameters
        ----------
        y_pred : float32 tensor
            Predicted molecule labels with shape (B, T),
            B for batch size and T for the number of tasks
        y_true : float32 tensor
            Ground truth molecule labels with shape (B, T)
        mask : float32 tensor
            Mask for indicating the existence of ground
            truth labels with shape (B, T)
        """
        self.y_pred.append(y_pred.detach().cpu())
        self.y_true.append(y_true.detach().cpu())
        self.mask.append(mask.detach().cpu())

    def roc_auc_score(self):
        """Compute roc-auc score for each task.
        Returns
        -------
        list of float
            roc-auc score for all tasks
        """       
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        y_pred = torch.sigmoid(y_pred)
        n_tasks = y_true.shape[1]
        scores = []
        if len(set(y_true)) < 2:
            print("警告：y_true中只有一个类别，无法计算ROC AUC。")
            return 0.5  
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(round(roc_auc_score(task_y_true, task_y_pred), 4))
        return scores

    def return_pred_true(self):
        """Compute roc-auc score for each task.
        Returns
        -------
        list of float
            roc-auc score for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        y_pred = torch.sigmoid(y_pred)
        n_tasks = y_true.shape[1]
        scores = []
        return y_pred, y_true

    def l1loss(self, reduction):
        """Compute l1 loss for each task.
        Returns
        -------
        list of float
            l1 loss for all tasks
        reduction : str
            * 'mean': average the metric over all labeled data points for each task
            * 'sum': sum the metric over all labeled data points for each task
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(F.l1_loss(task_y_true, task_y_pred, reduction=reduction).item())
        return scores

    def rmse(self):
        """Compute RMSE for each task.
        Returns
        -------
        list of float
            rmse for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(np.sqrt(F.mse_loss(task_y_pred, task_y_true).cpu().item()))
        return scores

    def mae(self):
        """Compute MAE for each task.
        Returns
        -------
        list of float
            mae for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(mean_squared_error(task_y_true, task_y_pred))
        return scores

    def r2(self):
        """Compute R2 for each task.
        Returns
        -------
        list of float
            r2 for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(round(r2_score(task_y_true, task_y_pred), 4))
        return scores
    

    def accuracy_score(self):
        """Compute Accuracy for each task.
        Returns
        -------
        list of float
            Accuracy values for all tasks
        """
        mask = torch.cat(self.mask, dim=0) 
        y_pred = torch.cat(self.y_pred, dim=0) 
        y_true = torch.cat(self.y_true, dim=0) 

        y_pred = torch.sigmoid(y_pred)  
        y_pred_binary = (y_pred > 0.5).float()  

        n_tasks = y_true.shape[1]  
        accuracy_scores = []

        for task in range(n_tasks):
            task_mask = mask[:, task]
            task_y_true = y_true[:, task][task_mask != 0].numpy()  
            task_y_pred = y_pred_binary[:, task][task_mask != 0].numpy()  
            accuracy = accuracy_score(task_y_true, task_y_pred)
            accuracy_scores.append(accuracy)

        return accuracy_scores

    def recall_score(self):
        """
        Compute Recall for each task.
        Returns
        -------
        list of float
        Recall values for all tasks
        """
        mask = torch.cat(self.mask, dim=0) 
        y_pred = torch.cat(self.y_pred, dim=0) 
        y_true = torch.cat(self.y_true, dim=0) 

        y_pred = torch.sigmoid(y_pred)  
        y_pred_binary = (y_pred > 0.5).float() 

        n_tasks = y_true.shape[1]  
        recall_scores = []

        for task in range(n_tasks):
            task_mask = mask[:, task] 
            task_y_true = y_true[:, task][task_mask != 0].numpy()  
            task_y_pred = y_pred_binary[:, task][task_mask != 0].numpy() 
            recall = recall_score(task_y_true, task_y_pred)
            recall_scores.append(recall)

        return recall_scores

    def roc_precision_score(self):
        """Compute AUC_PRC for each task.
        Returns
        -------
        list of float
            AUC_PRC for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        y_pred = torch.sigmoid(y_pred)
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            precision, recall, _thresholds = precision_recall_curve(task_y_true, task_y_pred)
            scores.append(auc(recall, precision))
        return scores

    def compute_metric(self, metric_name, reduction='mean'):
        """Compute metric for each task.
        Parameters
        ----------
        metric_name : str
            Name for the metric to compute.
        reduction : str
            Only comes into effect when the metric_name is l1_loss.
            * 'mean': average the metric over all labeled data points for each task
            * 'sum': sum the metric over all labeled data points for each task
        Returns
        -------
        list of float
            Metric value for each task
        """
        assert metric_name in ['roc_auc', 'l1', 'rmse', 'acc','mae', 'roc_prc', 'r2', 'return_pred_true','recall'], \
            'Expect metric name to be "roc_auc", "l1" or "rmse", "mae", "roc_prc", "r2", "return_pred_true", got {}'.format(metric_name)
        assert reduction in ['mean', 'sum']
        if metric_name == 'roc_auc':
            return self.roc_auc_score()
        if metric_name == 'l1':
            return self.l1loss(reduction)
        if metric_name == 'rmse':
            return self.rmse()
        if metric_name == 'mae':
            return self.mae()
        if metric_name == 'roc_prc':
            return self.roc_precision_score()
        if metric_name == 'recall':
            return self.recall_score()
        if metric_name == 'acc':
            return self.accuracy_score()
        if metric_name == 'r2':
            return self.r2()
        if metric_name == 'return_pred_true':
            return self.return_pred_true()


def collate_molgraphs(data):
    smiles, graphs, labels, mask = map(list, zip(*data))
    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = np.array(labels)
    labels = torch.tensor(labels)
    mask = np.array(mask)
    mask = torch.tensor(mask)

    return smiles, bg, labels,  mask


def run_a_train_epoch_heterogeneous_with_gate(args, epoch, model, data_loader, loss_criterion_c, loss_criterion_r, optimizer, alpha, task_weight=None):
    model.train()
    train_meter_c = Meter()
    train_meter_r = Meter()
    gate_weights_total = np.zeros((0, 7, 4))
    a = alpha
    if task_weight is not None:
        task_weight = task_weight.float().to(args['device'])
    for batch_id, batch_data in enumerate(data_loader):
        smiles, bg, labels, mask = batch_data
        fp_type = args['fingerprint_type']
        bg = bg.to(args['device'])
        mask = mask.float().to(args['device'])
        labels.float().to(args['device'])
        atom_feats = bg.ndata.pop(args['atom_data_field']).float().to(args['device'])
        bond_feats = bg.edata.pop(args['bond_data_field']).long().to(args['device'])
        logits, attentionweights, gate_weights = model(bg, atom_feats, bond_feats, smiles , fp_type,  device=args['device'],norm=None)
        print(gate_weights.shape)
        gate_weights = gate_weights.clone().detach()
        gate_weights_avg = torch.mean(gate_weights, dim=0, keepdim=True)
        print("gate_weights_avg shape:", gate_weights_avg.shape)
        gate_weights_avg = gate_weights_avg.detach().cpu().numpy()
        gate_weights_total = np.concatenate((gate_weights_total, gate_weights_avg), axis=0) 
        labels = labels.type_as(logits).to(args['device'])
        loss = (loss_criterion_c(logits, labels)*(mask != 0).float()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_meter_c.update(logits, labels, mask)
        del bg, mask, labels, atom_feats, bond_feats, loss,  logits
        torch.cuda.empty_cache()
        metric_roc_auc_results = train_meter_c.compute_metric('acc')
        avg_roc_auc = np.mean(metric_roc_auc_results) 
        loss = 1 - avg_roc_auc
        print('epoch {:d}/{:d}, training roc_auc {:.4f}, loss {:.4f}'.format(
            epoch + 1, args['num_epochs'], avg_roc_auc, loss))
    return gate_weights_total

def run_a_train_epoch_heterogeneous(args, epoch, model, data_loader, loss_criterion_c, optimizer):
    model.train()
    train_meter_c = Meter()
    for batch_id, batch_data in enumerate(data_loader):
        smiles, bg, labels, mask = batch_data
        fp_type = args['fingerprint_type']
        bg = bg.to(args['device'])
        mask = mask.float().to(args['device'])
        labels.float().to(args['device'])
        atom_feats = bg.ndata.pop(args['atom_data_field']).float().to(args['device'])
        bond_feats = bg.edata.pop(args['bond_data_field']).long().to(args['device'])
        logits, attentionweights, gate_weights = model(bg, atom_feats, bond_feats, smiles , fp_type,  device=args['device'],norm=None)
        labels = labels.type_as(logits).to(args['device'])
        loss = (loss_criterion_c(logits, labels)*(mask != 0).float()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_meter_c.update(logits, labels, mask)
        del bg, mask, labels, atom_feats, bond_feats, loss,  logits
        torch.cuda.empty_cache()
        train_score = np.mean(train_meter_c.compute_metric(args['classification_metric_name']))
        print('epoch {:d}/{:d}, training {} {:.4f}'.format(
            epoch + 1, args['num_epochs'], args['classification_metric_name'], train_score))

def run_an_eval_epoch_heterogeneous(args, model, data_loader,fp_type):
    model.eval()
    eval_meter_c = Meter()
    eval_meter_r = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, mask = batch_data
            labels = labels.float().to(args['device'])
            bg = bg.to(args['device']) 
            fp_type = args['fingerprint_type']
            mask = mask.float().to(args['device'])
            atom_feats = bg.ndata.pop(args['atom_data_field']).float().to(args['device'])
            bond_feats = bg.edata.pop(args['bond_data_field']).long().to(args['device'])
            logits,attentionweights, gate_weights = model(bg, atom_feats, bond_feats,  smiles, fp_type, device=args['device'], norm=None)
            labels = labels.type_as(logits).to(args['device'])
            eval_meter_c.update(logits, labels, mask)
            del smiles, bg,  mask, labels, atom_feats, bond_feats, logits
            torch.cuda.empty_cache()
        metric_results = {}
        for metric_name in args['classification_metric_name']:
            metric_results[metric_name] = eval_meter_c.compute_metric(metric_name)
        return metric_results

def run_an_eval_epoch_heterogeneous_generate_weight(args, model, data_loader):
    model.eval()
    atom_list_all = []
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            print("batch: {}/{}".format(batch_id+1, len(data_loader)))
            smiles, bg, labels, mask = batch_data
            labels = labels.float().to(args['device'])
            atom_feats = bg.ndata.pop(args['atom_data_field']).float().to(args['device'])
            bond_feats = bg.edata.pop(args['bond_data_field']).long().to(args['device'])
            logits,attentionweights, gate_weights = model(bg, atom_feats, bond_feats, norm=None)
            for atom_weight in attentionweights:
                atom_list_all.append(atom_weight[args['select_task_index']])
    task_name = args['select_task_list'][0]
    atom_weight_list = pd.DataFrame(atom_list_all, columns=['atom_weight'])
    atom_weight_list.to_csv(task_name+"_atom_weight.csv", index=None)



def generate_mol_feats(args, model, data_loader, dataset_output_path):
    model.eval()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, mask = batch_data
            atom_feats = bg.ndata.pop(args['atom_data_field']).float().to(args['device'])
            bond_feats = bg.edata.pop(args['bond_data_field']).long().to(args['device'])
            feats = model(bg, atom_feats, bond_feats, norm=None).numpy().tolist()
            feats_name = ['graph-feature' + str(i+1) for i in range(64)]
            data = pd.DataFrame(feats, columns=feats_name)
            data['smiles'] = smiles
            data['labels'] = labels.squeeze().numpy().tolist()
    data.to_csv(dataset_output_path, index=None)

class EarlyStopping(object):
    def __init__(self, pretrained_model='Null_early_stop.pth', mode='higher', patience=10, filename=None, task_name="None"):
        if filename is None:
            task_name = task_name
            filename ='/home/jwzhang/fuxianwenxian/MGA-main/model/{}_early_stop.pth'.format(task_name)

        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.counter = 0
        self.filename = filename
        self.best_score = None
        self.early_stop = False
        self.pretrained_model = pretrained_model

    def _check_higher(self, score, prev_best_score):
        return (score > prev_best_score)

    def _check_lower(self, score, prev_best_score):
        return (score < prev_best_score)

    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(
                'EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def nosave_step(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._check(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            print(
                'EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when the metric on the validation set gets improved.'''
        torch.save({'model_state_dict': model.state_dict()}, self.filename)
      
    def load_checkpoint(self, model):
        '''Load model saved with early stopping.'''
        model.load_state_dict(torch.load(self.filename, map_location=torch.device('cpu'))['model_state_dict'])

    def load_pretrained_model(self, model):
        pretrained_parameters = ['gnn_layers.0.graph_conv_layer.weight',
                                 'gnn_layers.0.graph_conv_layer.h_bias',
                                 'gnn_layers.0.graph_conv_layer.loop_weight',
                                 'gnn_layers.0.res_connection.weight',
                                 'gnn_layers.0.res_connection.bias',
                                 'gnn_layers.0.bn_layer.weight',
                                 'gnn_layers.0.bn_layer.bias',
                                 'gnn_layers.0.bn_layer.running_mean',
                                 'gnn_layers.0.bn_layer.running_var',
                                 'gnn_layers.0.bn_layer.num_batches_tracked',
                                 'gnn_layers.1.graph_conv_layer.weight',
                                 'gnn_layers.1.graph_conv_layer.h_bias',
                                 'gnn_layers.1.graph_conv_layer.loop_weight',
                                 'gnn_layers.1.res_connection.weight',
                                 'gnn_layers.1.res_connection.bias',
                                 'gnn_layers.1.bn_layer.weight',
                                 'gnn_layers.1.bn_layer.bias',
                                 'gnn_layers.1.bn_layer.running_mean',
                                 'gnn_layers.1.bn_layer.running_var',
                                 'gnn_layers.1.bn_layer.num_batches_tracked']
        if torch.cuda.is_available():
            pretrained_model = torch.load('/home/jwzhang/fuxianwenxian/MGA-main/model/'+self.pretrained_model)
        else:
            pretrained_model = torch.load('/home/jwzhang/fuxianwenxian/MGA-main/model/'+self.pretrained_model, map_location=torch.device('cpu'))
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_model['model_state_dict'].items() if k in pretrained_parameters}
        model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)

    def load_model_attention(self, model):
        pretrained_parameters = ['gnn_layers.0.graph_conv_layer.weight',
                                 'gnn_layers.0.graph_conv_layer.h_bias',
                                 'gnn_layers.0.graph_conv_layer.loop_weight',
                                 'gnn_layers.0.res_connection.weight',
                                 'gnn_layers.0.res_connection.bias',
                                 'gnn_layers.0.bn_layer.weight',
                                 'gnn_layers.0.bn_layer.bias',
                                 'gnn_layers.0.bn_layer.running_mean',
                                 'gnn_layers.0.bn_layer.running_var',
                                 'gnn_layers.0.bn_layer.num_batches_tracked',
                                 'gnn_layers.1.graph_conv_layer.weight',
                                 'gnn_layers.1.graph_conv_layer.h_bias',
                                 'gnn_layers.1.graph_conv_layer.loop_weight',
                                 'gnn_layers.1.res_connection.weight',
                                 'gnn_layers.1.res_connection.bias',
                                 'gnn_layers.1.bn_layer.weight',
                                 'gnn_layers.1.bn_layer.bias',
                                 'gnn_layers.1.bn_layer.running_mean',
                                 'gnn_layers.1.bn_layer.running_var',
                                 'gnn_layers.1.bn_layer.num_batches_tracked',
                                 'weighted_sum_readout.atom_weighting_specific.0.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.0.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.1.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.1.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.2.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.2.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.3.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.3.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.4.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.4.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.5.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.5.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.6.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.6.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.7.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.7.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.8.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.8.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.9.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.9.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.10.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.10.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.11.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.11.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.12.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.12.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.13.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.13.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.14.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.14.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.15.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.15.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.16.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.16.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.17.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.17.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.18.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.18.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.19.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.19.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.20.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.20.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.21.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.21.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.22.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.22.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.23.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.23.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.24.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.24.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.25.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.25.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.26.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.26.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.27.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.27.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.28.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.28.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.29.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.29.0.bias',
                                 'weighted_sum_readout.atom_weighting_specific.30.0.weight',
                                 'weighted_sum_readout.atom_weighting_specific.30.0.bias',
                                 'weighted_sum_readout.shared_weighting.0.weight',
                                 'weighted_sum_readout.shared_weighting.0.bias',
                                 ]
        if torch.cuda.is_available():
            pretrained_model = torch.load('../model/' + self.pretrained_model)
        else:
            pretrained_model = torch.load('../model/' + self.pretrained_model, map_location=torch.device('cpu'))
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_model['model_state_dict'].items() if k in pretrained_parameters}
        model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)



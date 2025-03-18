from utils import build_dataset
import torch
from torch.utils.data import DataLoader
import os
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
import torch
from matplotlib.colors import LinearSegmentedColormap
from utils import build_dataset
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from utils.MYGNN import collate_molgraphs, EarlyStopping, run_a_train_epoch_heterogeneous, run_an_eval_epoch_heterogeneous, set_random_seed, pos_weight, AquaToxPredictor
import pandas as pd
args = {}
args['device'] = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
device = args['device']
args['num_epochs'] = 500
args['patience'] = 50
args['batch_size'] = 128
args['rgcn_hidden_feats'] = [256, 256]
args['lr'] = 3
args['weight_decay'] = 5
args['num_experts']= 4
args['task_hidden_dim']=128
args['model_name'] ='AquaTox-Predictor'
args['expert_hidden_dim'] = 128  # change
args['task_hidden_dim'] = 256  # change
args['num_experts'] = 4 
args['num_rels'] = 64*21  # change
args['task_name'] = 'aquatic_toxicity'  # change
args['fingerprint_type'] = ['morgan','rdkit','maccs']  # change
args['times'] = 10
args['select_task_list'] = ['DM', 'FHM', 'LP', 'PS', 'RT', 'SHM', 'TP']  # change
args['num_tasks'] = 7
args['bin_path'] = '../data/' + args['task_name'] + '.bin'
args['group_path'] = '../data/' + args['task_name'] + '_group.csv'
num_experts=args['num_experts']
expert_hidden_dim=args['expert_hidden_dim']
task_hidden_dim=args['task_hidden_dim']
set_random_seed(2024)
print('***************************************************************************************************')
train_set, val_set, test_set, task_number = build_dataset.load_graph_from_csv_bin_for_splited(
    bin_path=args['bin_path'],
    group_path=args['group_path'],
    select_task_index=args['select_task_index']
)
print("Molecule graph generation is complete !")
train_loader = DataLoader(dataset=train_set,
                            batch_size=args['batch_size'],
                            shuffle=True,
                            collate_fn=collate_molgraphs
                            )
model_path = '../model/Model.pth'
model=AquaToxPredictor(rgcn_in_feats=40,rgcn_hidden_feats=args['rgcn_hidden_feats'],
            num_experts=args['num_experts'],
            expert_hidden_dim=args['expert_hidden_dim'],
            task_hidden_dim=args['task_hidden_dim'],num_rels=args['num_rels'],
            num_tasks=args['num_tasks'],fp_dim=256, drop_ratio=0.2, fp_type=args['fp_type'],
            device=args['device']) 
model.load_state_dict(torch.load(model_path)) 
model.eval() 
save_path = '../attention/'
for batch_id, batch_data in enumerate(train_loader):
    smiles, bg, labels, mask = batch_data
    labels = labels.float().to(args['device'])
    bg = bg.to(args['device'])  
    fp_type = args['fingerprint_type']
    mask = mask.float().to(args['device'])
    atom_feats = bg.ndata.pop(args['atom_data_field']).float().to(args['device'])
    bond_feats = bg.edata.pop(args['bond_data_field']).long().to(args['device'])
    logits, attention_weights, gate_weights = model(bg, atom_feats, bond_feats,  smiles, fp_type, device=args['device'], norm=None)
    torch.sigmoid(logits)
    logits = (logits >= 0.5).float()
    logits = logits.int()
    attention_scores = attention_weights.squeeze()
    attention_scores = attention_scores.cpu().detach().numpy()
    results = []
    atoms_per_molecule = [Chem.MolFromSmiles(smiles).GetNumAtoms() for smiles in smiles]
    colors = [
    "#6666FF", "#0000FF", "#0040FF", "#0080FF", "#00CCFF", 
    "#00FFFF", "#9FFFFF", "#FFFFFF", "#FFDEDE", "#FEA5A5", 
    "#FF6F6F", "#FF3737", "#FF0000", "#BE0000", "#FF3399"
]
    custom_cmap = LinearSegmentedColormap.from_list("custom_gradient", colors)
    start_idx = 0
    for idx, num_atoms in enumerate(atoms_per_molecule):
        end_idx = start_idx + num_atoms
        molecule_attention_scores = attention_scores[start_idx:end_idx]
        results.append({
            "SMILES": smiles[idx],
            'DM': logits[idx][0].item(),
            'FHM': logits[idx][1].item(),  
            'LP': logits[idx][2].item(),  
            'PS': logits[idx][3].item(),  
            'RT': logits[idx][4].item(),  
            'SHM': logits[idx][5].item(), 
            'TP': logits[idx][6].item(), 
        })
        mol = Chem.MolFromSmiles(smiles[idx])
        norm = plt.Normalize(vmin=molecule_attention_scores.min(), vmax=molecule_attention_scores.max())
        atom_colors = {i: custom_cmap(norm(score))[:3] for i, score in enumerate(molecule_attention_scores)} 
        atom_radii = {i: float(0.3 + 0.4 * abs(norm(score))) for i, score in enumerate(molecule_attention_scores)} 
        drawer = Draw.MolDraw2DCairo(400, 400)  
        options = drawer.drawOptions()
        options.continuousHighlight = False 
        options.setHighlightBondWidthMultiplier = 0
        drawer.DrawMolecule(
            mol,
            highlightAtoms=range(mol.GetNumAtoms()),  
            highlightAtomColors=atom_colors,
            highlightBonds=[],                       
            highlightAtomRadii=atom_radii
        )
        drawer.FinishDrawing()
        img = drawer.GetDrawingText()
        mol_img = plt.imread(Draw.BytesIO(img))
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(mol_img)  
        ax.axis('off')      
        sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, orientation='vertical', label='Attention Score')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{idx+1}attention.png"))
        plt.show()
        print(f"分子 {idx+1} 绘制完成，图像保存为{idx+1}attention.png")
        start_idx = end_idx
df_results = pd.DataFrame(results)
df_results.to_csv('../predictresult/prediction.csv', index=False)





import numpy as np
from utils import build_dataset
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from utils.MYGNN import collate_molgraphs, EarlyStopping, run_a_train_epoch_heterogeneous, run_an_eval_epoch_heterogeneous, set_random_seed, pos_weight, AquaToxPredictor
import time
import pandas as pd
start = time.time()
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

result_pd = pd.DataFrame(columns=args['select_task_list']+['group'])
all_times_train_result = []
all_times_val_result = []
all_times_test_result = []
for time_id in range(args['times']):
    set_random_seed(2020+time_id)
    one_time_train_result = []
    one_time_val_result = []
    one_time_test_result = []
    print('***************************************************************************************************')
    print('{}, {}/{} time'.format(args['task_name'], time_id+1, args['times']))
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

    val_loader = DataLoader(dataset=val_set,
                            batch_size=args['batch_size'],
                            shuffle=True,
                            collate_fn=collate_molgraphs
                            )

    test_loader = DataLoader(dataset=test_set,
                             batch_size=args['batch_size'],
                             collate_fn=collate_molgraphs
                             )
    
    pos_weight_np = pos_weight(train_set, classification_num=7)
    loss_criterion_c = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight_np.to(args['device']))
    model=AquaToxPredictor(rgcn_in_feats=40,rgcn_hidden_feats=args['rgcn_hidden_feats'],
        num_experts=num_experts,
        expert_hidden_dim=expert_hidden_dim,
        task_hidden_dim=task_hidden_dim,num_rels=args['num_rels'],
        num_tasks=args['num_tasks'],fp_dim=256, drop_ratio=0.2, fingerprint_type=args['fingerprint_type'],
        device=args['device'])
    optimizer = Adam(model.parameters(), lr=10**-args['lr'], weight_decay=10**-args['weight_decay'])
    stopper = EarlyStopping(patience=args['patience'], task_name=args['task_name'], mode=args['mode'])
    model.to(args['device'])
    for epoch in range(args['num_epochs']):
        run_a_train_epoch_heterogeneous(args, epoch, model, train_loader, loss_criterion_c, optimizer)
        validation_result = run_an_eval_epoch_heterogeneous(args, model, val_loader,fingerprint_type=args['fingerprint_type'])
        val_scores = {key: np.mean(value) for key, value in validation_result.items() if key == 'roc_auc'}
        val_score = val_scores['roc_auc']
        early_stop = stopper.step(val_score, model)
        print('epoch {:d}/{:d}, validation {:.4f}, best validation {:.4f}'.format(
            epoch + 1, args['num_epochs'],
        val_score,  stopper.best_score))
        if early_stop:
            break
    stopper.load_checkpoint(model)
    filename = f"../model/num_experts{num_experts}_expert_hidden_dim{expert_hidden_dim}_task_hidden_dim{task_hidden_dim}_time{time_id}.pth"
    torch.save(model.state_dict(), filename)
    train_result = run_an_eval_epoch_heterogeneous(args, model, train_loader,fingerprint_type=args['fingerprint_type'])
    val_result = run_an_eval_epoch_heterogeneous(args, model, val_loader,fingerprint_type=args['fingerprint_type'])
    test_result = run_an_eval_epoch_heterogeneous(args, model, test_loader,fingerprint_type=args['fingerprint_type'])
    # deal result
    result = []
    for key in train_result.keys():  # 假设三个字典有相同的keys
    # 添加训练结果
        result.append(round(train_result[key], 4) if isinstance(train_result[key], (float, np.float64)) else train_result[key])
        result.append(f'train_{key}')
    # 添加验证结果
        result.append(round(val_result[key], 4) if isinstance(val_result[key], (float, np.float64)) else val_result[key])
        result.append(f'valid_{key}')
    # 添加测试结果
        result.append(round(test_result[key], 4) if isinstance(test_result[key], (float, np.float64)) else test_result[key])
        result.append(f'test_{key}')
    num_columns = 8
    rows = [result[i:i+num_columns] for i in range(0, len(result), num_columns)]
    for row in rows:
        result_pd.loc[len(result_pd)] = row
    print('********************************{}, {}_times_result*******************************'.format(args['task_name'], time_id+1))
    print("training_result:", train_result)
    print("val_result:", val_result)
    print("test_result:", test_result)
    result_pd.to_csv('../result/'+args['task_name']+args['model_name']+'_result.csv', index=None)


elapsed = (time.time() - start)
m, s = divmod(elapsed, 60)
h, m = divmod(m, 60)
print("Time used:", "{:d}:{:d}:{:d}".format(int(h), int(m), int(s)))













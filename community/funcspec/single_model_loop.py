from types import new_class
import wandb 
from community.funcspec.masks import train_and_get_mask_metric
from community.funcspec.bottleneck import readout_retrain
from community.funcspec.correlation import get_pearson_metrics
from community.common.training import train_community
from community.common.init import init_community, init_optimizers
from community.utils.wandb_utils import get_training_dict
import numpy as np
import copy


def init_and_train(config, loaders, device) : 

    agents_params_dict = config['model_params']['agents_params']
    connections_params_dict = config['model_params']['connections_params']

    p_con, n_hid = connections_params_dict['sparsity'], agents_params_dict['n_hid']

    deepR_params_dict = config['optimization']['connections']
    params_dict = config['optimization']['agents']

    wandb.log({'p_connection' : p_con})
    wandb.log({'q_measure' : (1 - p_con)/(2 * (1 + p_con)) })
    wandb.log({'n_hidden' : n_hid})

    # ------  Train ------

    train_outs = {}
    trained_coms = {}
    for use_bottleneck in [False] : 
            
        agents_params_dict['use_bottleneck'] = use_bottleneck
        community = init_community(agents_params_dict, connections_params_dict, device)

        optimizers, schedulers = init_optimizers(community, params_dict, deepR_params_dict)

        if config['do_training'] : 
                
            training_dict = get_training_dict(config)
            train_out = train_community(community, *loaders, optimizers, schedulers,                          
                                        config=training_dict, device=device, use_tqdm=1)

            test_accs = train_out['test_accs']
            if len(test_accs.shape) == 1 : 
                best_test_acc = np.max(test_accs)
            else : 
                best_test_acc = np.max(test_accs, 0).min()

            mean_d_ags = train_out['deciding_agents'].mean()

            metric_names = ['Best Test Acc' + '_bottleneck'*use_bottleneck, 'Mean Decision' + '_bottleneck'*use_bottleneck]
            for metric, metric_name in zip([best_test_acc, mean_d_ags], metric_names) : 
                
                try : 
                    for m, sub_metric in enumerate(metric) : 
                        wandb.define_metric(metric_name + f'_{m}', step_metric='p_connection')
                        wandb.define_metric(metric_name + f'_{m}', step_metric='n_hidden')
                        wandb.log({
                            metric_name + f'_{m}' : sub_metric
                            })
                except : 
                    wandb.define_metric(metric_name, step_metric='p_connection')
                    wandb.define_metric(metric_name, step_metric='n_hidden')
                    wandb.log({
                        metric_name : metric
                        })

            community.best_acc = best_test_acc
            train_outs[f'With{(1-use_bottleneck)*"out"} Bottleneck'] = train_out
            trained_coms[f'With{(1-use_bottleneck)*"out"} Bottleneck'] = copy.deepcopy(community)

    return trained_coms, train_outs


def compute_all_metrics(trained_coms, loaders, config, device) : 

    symbols = config['datasets']['data_type'] == 'symbols'
    deepR_params_dict = config['optimization']['connections']
    n_classes = config['datasets']['n_classes']

    for n in range(2) : 
        wandb.define_metric(f'Correlation Diff {n}', step_metric='p_connection')
        wandb.define_metric(f'Masks Diff {n}', step_metric='p_connection')
        wandb.define_metric(f'Bottleneck Diff {n}', step_metric='p_connection')

        for t in range(2) : 
            wandb.define_metric(f'Correlation Agent {n}, Task {t}', step_metric='p_connection')
            wandb.define_metric(f'Masks Agent {n}, Task {t}', step_metric='p_connection')
            wandb.define_metric(f'Bottleneck Agent {n}, Task {t}', step_metric='p_connection')
           

    community = trained_coms['Without Bottleneck']
    #print('Correlations')
    correlations_results = get_pearson_metrics(community, loaders, device=device, use_tqdm=1, symbols=symbols)
    correlations_metric = correlations_results.mean(-1).mean(-1)

    #print('Weight Masks')
    masks_metric = {}
    #masks_results = train_and_get_mask_metric(community, 0.1, loaders, device=device, n_tests=1, n_epochs=1, use_tqdm=1, use_optimal_sparsity=True, symbols=symbols)
    #masks_props, masks_accs, _, masks_states, masks_spars = list(masks_results.values())
    #masks_metric, masks_accs, masks_spars = masks_props.mean(0), masks_accs.mean(0).max(-1), masks_spars.mean(0)

    community = trained_coms['Without Bottleneck']
    #print('Bottlenecks Retrain')
    bottleneck_results = readout_retrain(community, loaders, n_classes, deepR_params_dict=deepR_params_dict, n_tests=1, n_epochs=3, device=device, use_tqdm=1, symbols=symbols)
    bottleneck_metric = bottleneck_results['accs'].mean(0)

    diff_metric = lambda metric, ag : (metric[ag, ag] - metric[1-ag, ag]) / (metric[ag, ag] + metric[1-ag, ag])

    # ------ Log ------
    #metrics = [correlations_metric, masks_metric, bottleneck_metric]
    #metric_names = ['Correlation', 'Masks', 'Bottleneck']
    #all_results = [correlations_results, masks_results, bottleneck_results]

    metrics = [correlations_metric, bottleneck_metric]
    metric_names = ['Correlation', 'Bottleneck']
    all_results = [correlations_results, bottleneck_results]

    metric_results = {metric_name : metric for metric, metric_name in zip(metrics, metric_names)}
    all_metric_results = {metric_name : metric for metric, metric_name in zip(all_results, metric_names)}

    for metric, metric_name in zip(metrics, metric_names) : 
        
        metric_log = {}
        for ag in range(2) : 
            metric_log[f'{metric_name} Diff {ag}'] = diff_metric(metric, ag)
            for task in range(2) : 
                metric_log[f'{metric_name} Agent {ag}, Task {task}'] = metric[task, ag]
        
        wandb.log(metric_log)

    return metric_results, all_metric_results



def train_and_compute_metrics(config, loaders, device) : 

    # ------ Train ------ 

    trained_coms, train_outs = init_and_train(config, loaders, device)

    # ------ Metrics ------
    
    metric_results, all_metric_results = compute_all_metrics(trained_coms, loaders, config, device)

    return metric_results, train_outs, all_metric_results
import wandb 
from community.funcspec.masks import train_and_get_mask_metric
from community.funcspec.bottleneck import readout_retrain
from community.funcspec.correlation import get_pearson_metrics
from community.common.training import train_community
from community.common.init import init_community, init_optimizers
from community.common.wandb_utils import get_training_dict
import numpy as np

def train_and_compute_metrics(p_con, config, loaders, device) : 

    agents_params_dict = config['model_params']['agents_params']
    deepR_params_dict = config['optimization']['connections']
    params_dict = config['optimization']['agents']

    wandb.define_metric('p_connection')
    wandb.log({'p_connection' : p_con})
    wandb.log({'q_measure' : (1 - p_con)/(2 * (1 + p_con)) })

    # ------  Train ------

    community = init_community(agents_params_dict,
                               p_con,
                               use_deepR=config['model_params']['use_deepR'],
                               device=device)
    optimizers, schedulers = init_optimizers(community, params_dict, deepR_params_dict)

    if config['do_training'] : 
            
        training_dict = get_training_dict(config)
        train_out = train_community(community, *loaders, optimizers, schedulers,                          
                                    config=training_dict, device=device, use_tqdm=1)

        best_test_acc = np.max(train_out['test_accs'])
        mean_d_ags = train_out['deciding_agents'].mean()

        for metric, metric_name in zip([best_test_acc, mean_d_ags], ['Best Test Acc', 'Mean Decision']) : 

            wandb.define_metric(metric_name, step_metric='p_connection')
            wandb.log({
                metric_name : metric
                })

    # ------ Metrics ------

    for n in range(2) : 
        wandb.define_metric(f'Correlation Diff {n}', step_metric='p_connection')
        wandb.define_metric(f'Masks Diff {n}', step_metric='p_connection')
        wandb.define_metric(f'Bottleneck Diff {n}', step_metric='p_connection')

        """
        for t in range(2) : 
            wandb.define_metric(f'Correlation Agent {n}, Task {t}', step_metric='p_connection')
            wandb.define_metric(f'Masks Agent {n}, Task {t}', step_metric='p_connection')
            wandb.define_metric(f'Bottleneck Agent {n}, Task {t}', step_metric='p_connection')
        """   

    #print('Correlations')
    correlations = get_pearson_metrics(community, loaders, n_tests=64, device=device, use_tqdm=1)
    correlations_metric = correlations.mean(-1).mean(-1)

    #print('Weight Masks')
    masks_props, masks_accs, _, masks_states = train_and_get_mask_metric(community, 0.1, loaders, device=device, n_tests=3, n_epochs=1, use_tqdm=1)
    masks_metric, masks_accs = masks_props.mean(0), masks_accs.mean(0)[..., -1]

    #print('Bottlenecks Retrain')
    bottleneck = readout_retrain(community, loaders, deepR_params_dict=deepR_params_dict, n_tests=3, n_epochs=1, device=device, use_tqdm=1)
    bottleneck_metric = bottleneck['accs'].mean(0).max(-1)

    ag_metric = lambda metric, ag : (metric[ag, ag], metric[ag, 1-ag])
    diff_metric = lambda metric, ag : ((ag_metric(metric, ag)[0]-ag_metric(metric, ag)[1])/(ag_metric(metric, ag)[0]+ag_metric(metric, ag)[1]))

    # ------ Log ------

    for metric, metric_name in zip([correlations_metric, masks_metric, bottleneck_metric], ['Correlation', 'Masks', 'Bottleneck']) : 
        
        metric_log  = {
            f'{metric_name} Diff {ag}' : diff_metric(metric, ag)
            for ag in range(2)
        }

        wandb.log(metric_log)
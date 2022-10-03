import torch
import wandb
from pathlib import Path
from .configs import get_new_config

#------ WandB Utils ------

def get_wandb_runs(run_id=None, config=None, entity='gbena', project='funcspec', process_config=False) :

    api = wandb.Api()    
    if process_config and config is not None: 
        config = get_new_config(config, 'config')
    elif config is not None : 
        config['state'] = 'finished'

    runs = api.runs(f'{entity}/{project}', filters=config) # Filtered

    if run_id is not None : 
        runs = [r for r in runs if r.id == run_id]

    assert len(runs) > 0, f'No runs found for current filters'

    print(f'Found {len(runs)} runs, returning...')

    return runs


def get_wandb_artifact(config=None, project='Spec_vs_Sparsity', name='correlations', type='metric', process_config=False, run_id=None, ensure_latest=False) : 
    entity = 'gbena'
    
    runs = get_wandb_runs(run_id, config, entity, project, process_config)

    if len(runs) != 1 : 
        print(f'Warning : {len(runs)} runs found for current filters')#, taking last one by default as no run id is specified')
    artifacts = [r.logged_artifacts() for r in runs]
    
    wanted_artifacts =[[art for art in artifact if name in art.name] for artifact in artifacts]
    wanted_artifacts = [art for art in wanted_artifacts if len(art) > 0]
    assert len(wanted_artifacts) > 0, f'No artifacts found for name {name} or type {type} for {len(runs)} currently filtered runs'
    
    if len(wanted_artifacts) != 1 : 
        print(f'Warning : {len(wanted_artifacts)} runs containing wanted artifact, taking last one by default as no run id or precise name is specified')  
    wanted_artifact = wanted_artifacts[0]

    if len(wanted_artifact) != 1 : 
        print(f'Warning : {len(wanted_artifacts)} artifacts found for single run, taking last one by default')
    wanted_artifact = wanted_artifact[0]

    if ensure_latest : 
        assert 'latest' in wanted_artifact.aliases, 'Artifact found is not the latest, disable ensure_latest to return anyway'

    wanted_artifact.download()
    try : 
        wandb.use_artifact(wanted_artifact.name)
    except wandb.Error : 
        pass
    return torch.load(wanted_artifact.file()), wanted_artifacts, runs


def mkdir_or_save_torch(to_save, save_name, save_path) : 
    try : 
        torch.save(to_save, save_path + save_name)
    except FileNotFoundError : 
        path = Path(save_path)
        path.mkdir(parents=True)
        torch.save(to_save, save_path + save_name)


def update_dict(config, new_config) : 
    
    for k, v in new_config.items() : 
        if type(v) is dict : 
            config[k] = update_dict(config[k], v)
        else : 
            config[k] = v
            
    return config



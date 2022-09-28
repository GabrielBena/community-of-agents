
def get_training_dict(config)  : 

    training_dict = {
        'n_epochs' : config['training']['n_epochs'], 
        'task' : config['task'],
        'global_rewire' : config['model_params']['connections_params']['global_rewire'], 
        'check_gradients' : False, 
        'reg_factor' : 0.,
        'train_connections' : True,
        'decision_params' : config['training']['decision_params'],
        'stopping_acc' : config['training']['stopping_acc'] ,
        'early_stop' : config['training']['early_stop'] ,
        'deepR_params_dict' : config['optimization']['connections'],
        'data_type' : config['datasets']['data_type'] , 
        'force_connections' : config['training']['force_connections']
    }

    return training_dict

def find_and_change(config, v_param, param_to_change=None) : 
    if param_to_change is None : 
        param_to_change = config['varying_param']

    for k, v in config.items() : 
        if type(v) is dict : 
            find_and_change(v, v_param, param_to_change)
        else : 
            if k == param_to_change : 
                config[param_to_change] = v_param

def find_varying_param(config, param_to_change=None) : 

    if param_to_change is None : 
        param_to_change = config['varying_param']

    for k, v in config.items() : 
        if type(v) is dict : 
            return find_varying_param(v, param_to_change)
        else : 
            if k == param_to_change : 
                return config[param_to_change]

def get_new_config(config, key_prefix='config') : 
    new_config = {}
    for k1, v1 in config.items() : 
        if type(v1) is dict : 
            sub_config = get_new_config(v1, k1)
            new_config.update({key_prefix + '.' + k : v for k,v in sub_config.items()})
        else : 
            new_config[key_prefix + '.' + k1] = v1
    return new_config
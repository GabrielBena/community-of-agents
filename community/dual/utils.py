import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

#------ Task Inference ------

get_entropy = lambda logits, n_classes: -(mask_detach(logits, n_classes).softmax(dim=1) * mask_detach(logits, n_classes).log_softmax(dim=1)).sum(1).mean()
mask = lambda logits, n_classes :  torch.tensor([int(y<n_classes) for y in range(logits.shape[1])]).to(logits.device)
masked = lambda logits, n_classes : mask(logits, n_classes)*logits
n_masked = lambda logits, n_classes : (1-mask(logits, n_classes))*logits
mask_detach = lambda logits, n_classes : n_masked(logits, n_classes).detach() + masked(logits, n_classes)
get_entropy_2 = lambda logits, n_classes : -torch.logsumexp(mask_detach(logits, n_classes), 1).mean()

def oneshot_task_inference(model, batch, shots=1, num_examples=20, eps=2e-3, e_function=0):
        # Initialize alphas to uniform
        alphas = torch.ones(model.num_tasks) / model.num_tasks
        alphas.requires_grad_(True)
        model.alphas = alphas


        e_functions = [get_entropy, get_entropy_2]

        data = batch[:, :num_examples, ...]
        for _ in range(shots) : 
                
            logits, _ = model.forward(data, '-1')
            logits = logits[-1].sum(0)
            # Entropy of logits
            entropy = e_functions[e_function](logits, model.n_out[0])

            # Gradient wrt alphas
            g, = autograd.grad(entropy, model.alphas)
            inferred_task = (-g).squeeze().argmax()

            model.alphas = model.alphas - 10e-2*g
            model.alphas = F.softmax(model.alphas, 0)            

        confidence = F.softmax(-g.squeeze(), 0)

        if model.num_tasks*confidence.max() < 1 + eps and model.train: 
            print('Adding Task')
            return str(model.num_tasks), confidence
        else : 
            inferred_task = confidence.argmax()
            return model.task_ids[inferred_task.item()] , confidence, logits, entropy, g

def binary_task_inference(model, batch, num_examples=20, eps=2e-3, e_function=0):
        # Initialize alphas to uniform
        alphas = torch.ones(model.num_tasks) / model.num_tasks
        alphas.requires_grad_(True)
        model.alphas = alphas
        
        mask_alphas = lambda alphas, mask : (~mask)*alphas.detach() + mask*alphas

        e_functions = [get_entropy, get_entropy_2]
        zero_idx = torch.tensor([False for gi in alphas])
        grads = []

        data = batch[:, :num_examples, ...]
        while model.alphas.norm(0) > 1  : 
                
            logits, _ = model.forward(data, '-1')
            logits = logits[-1].sum(0)
            # Entropy of logits
            entropy = e_functions[e_function](logits, model.n_out[0])
            # Gradient wrt alphas
            g, = autograd.grad(entropy, model.alphas)
            g = -g.squeeze()
            grads.append(g)

            zero_idx = torch.tensor([gi <= g[g.nonzero().squeeze()].median() for gi in g])
            model.alphas = (model.alphas*~zero_idx)
            model.alphas =  model.alphas/model.alphas.norm(1)        
            #model.alphas = mask_alphas(model.alphas, ~zero_idx)

        return model.task_ids[model.alphas.argmax()], grads

        if model.num_tasks*confidence.max() < 1 + eps and model.train: 
            print('Adding Task')
            return str(model.num_tasks), confidence
        else : 
            inferred_task = confidence.argmax()
            return model.task_ids[inferred_task.item()] , confidence, logits, entropy, g

#------ Loading Function ------
         
def sup_sup_load_dict(community, state_dict, data) : 
    try :
        state_dict = torch.load(state_dict)
    except AttributeError : 
        pass
    task_ids = []
    for n in state_dict.keys() : 
        if 'output_scores' in n : 
            task_ids.append(n[14:])
            
    for task in task_ids : 
        community(data, task)
    community.load_state_dict(state_dict)

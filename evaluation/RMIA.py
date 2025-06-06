import numpy as np
import torch
import torch.nn.functional as F


def get_x_y_from_data_dict(data, device):
    x, y = data.values()
    if isinstance(x, list):
        x, y = x[0].to(device), y[0].to(device)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def entropy(p, dim=-1, keepdim=False):
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)


def m_entropy(p, labels, dim=-1, keepdim=False):
    log_prob = torch.where(p > 0, p.log(), torch.tensor(1e-30).to(p.device).log())
    reverse_prob = 1 - p
    log_reverse_prob = torch.where(
        p > 0, p.log(), torch.tensor(1e-30).to(p.device).log()
    )
    modified_probs = p.clone()
    modified_probs[:, labels] = reverse_prob[:, labels]
    modified_log_probs = log_reverse_prob.clone()
    modified_log_probs[:, labels] = log_prob[:, labels]
    return -torch.sum(modified_probs * modified_log_probs, dim=dim, keepdim=keepdim)


def collect_prob(data_loader, model, device, one_hot, logits_out):
    prob = []
    all_targets = []

    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.float().to(device), targets.to(device)
            outputs = model(inputs)

            if logits_out:
                prob.append(outputs)
            else:
                prob.append(F.softmax(outputs, dim=-1))

            all_targets.append(targets.reshape(-1))
        
        prob = torch.cat(prob, axis=0)
        print(prob.shape)
        all_targets = torch.cat(all_targets, axis=0)
        print(all_targets.shape)
        _, predicted = prob.max(1)
        correct = predicted.eq(all_targets)
        accuracy = torch.mean(correct.float())

        if not one_hot:
            prob = prob[torch.arange(all_targets.size(0)), all_targets]
        print('prob shape: ', prob.shape)

    return prob, accuracy, all_targets


def RMIA(model, remain_loader, forget_loader, test_loader, device, one_hot=True, logits_out=True):
    """
    train_loader is the data involved in the training of the reference models.
    """

    remain_probs, remain_acc, remain_targets = collect_prob(remain_loader, model, device, one_hot, logits_out)
    forget_probs, forget_acc, forget_targets = collect_prob(forget_loader, model, device, one_hot, logits_out)
    test_probs, test_acc, test_targets  = collect_prob(test_loader, model, device, one_hot, logits_out)

    m = {
        "remain_likelihood": remain_probs,
        "forget_likelihood": forget_probs,
        "test_likelihood": test_probs,
        "remain_targets": remain_targets,
        "forget_targets": forget_targets,
        "test_targets": test_targets,
        "remain_acc": remain_acc,
        "forget_acc": forget_acc,
        "test_acc": test_acc,

    }
    
    return m

import torch
import torch.optim as optim
import torch.nn.functional as F
import copy
from api import FedAlgorithm
from utils import weighted_sum_functions
from collections import namedtuple
from typing import List
import ray
from torch.optim.optimizer import Optimizer



FEDPD_server_state = namedtuple("FEDPD_server_state", ['global_round', 'model'])
FEDPD_client_state = namedtuple("FEDPD_client_state", ['global_round', 'model', 'lambda_var'])


class FEDPD(FedAlgorithm):
    def __init__(self, init_model,
                 client_dataloaders,
                 loss,
                 loggers,
                 config,
                 device
                 ):
        super(FEDPD, self).__init__(init_model, client_dataloaders, loss, loggers, config, device)
        self.eta = config.eta
        self.n_workers = config.n_workers
        self.n_workers_per_round = config.n_workers_per_round
        if self.config.use_ray:
            ray.init()

    def server_init(self, init_model):
        return FEDPD_server_state(global_round=0, model=init_model)

    def client_init(self, server_state: FEDPD_server_state, client_dataloader):
        return FEDPD_client_state(global_round=server_state.global_round, model=server_state.model, lambda_var=None)

    def clients_step(self, clients_state, active_ids):
        active_clients = zip([clients_state[i] for i in active_ids], [self.client_dataloaders[i] for i in active_ids])
        if not self.config.use_ray:
            new_clients_state = [
                _client_step(self.config, self.loss, self.device, client_state, client_dataloader, self.eta)
                for client_state, client_dataloader in active_clients]
        else:
            new_clients_state = ray.get(
                [client_step.remote(self.config, self.loss, self.device, client_state, client_dataloader, self.eta)
                 for client_state, client_dataloader in active_clients])
        for i, new_client_state in zip(active_ids, new_clients_state):
            clients_state[i] = new_client_state
        return clients_state

    def server_step(self, server_state: FEDPD_server_state, client_states: FEDPD_client_state, weights, active_ids):
        # todo: implement the weighted version
        active_clients = [client_states[i] for i in active_ids]

        new_server_state = FEDPD_server_state(
            global_round=server_state.global_round + 1,
            model=weighted_sum_functions(
                [client_state.model for client_state in active_clients],
                [(1. / self.n_workers_per_round)] * len(active_clients)
            )
        )
        return new_server_state

    def clients_update(self, server_state: FEDPD_server_state, clients_state: List[FEDPD_client_state], active_ids):
        return [FEDPD_client_state(global_round=server_state.global_round, model=server_state.model, lambda_var=client.lambda_var)
                for client in clients_state]


@ray.remote(num_gpus=.3, num_cpus=4)
def client_step(config, loss_fn, device, client_state: FEDPD_client_state, client_dataloader, eta):
    f_local = copy.deepcopy(client_state.model)
    f_initial = client_state.model
    f_local.requires_grad_(True)

    lr_decay = 1.
    optimizer = MYOPT(f_local.parameters(), lr=lr_decay * config.local_lr)

    for epoch in range(config.local_epoch):
        for data, label in client_dataloader:
            optimizer.zero_grad()
            data = data.to(device)
            label = label.to(device)
            loss = loss_fn(f_local(data), label)

            # Now compute the quadratic part
            quad_penalty = 0.0
            for theta, theta_init in zip(f_local.parameters(), f_initial.parameters()):
                quad_penalty += F.mse_loss(theta, theta_init, reduction='sum')

            loss += quad_penalty / 2. / eta

            # Now take loss
            loss.backward()

            optimizer.step(client_state.lambda_var)

    # Update the dual variable

    print(loss.item())
    with torch.autograd.no_grad():
        lambda_delta = None
        for param_1, param_2 in zip(f_local.parameters(), f_initial.parameters()):
            if not isinstance(lambda_delta, torch.Tensor):
                lambda_delta = (param_1 - param_2).view(-1) / eta
            else:
                lambda_delta = torch.cat((lambda_delta, (param_1 - param_2).view(-1) / eta), dim=0)

        lambda_var = lambda_delta if client_state.lambda_var is None else client_state.lambda_var + lambda_delta

        # update f_local
        sd = f_local.state_dict()
        for key, param in zip(sd, lambda_var):
            sd[key] = sd[key] + eta * param
        f_local.load_state_dict(sd)

    return FEDPD_client_state(global_round=client_state.global_round, model=f_local, lambda_var=lambda_var)


def _client_step(config, loss_fn, device, client_state: FEDPD_client_state, client_dataloader, eta):
    f_local = copy.deepcopy(client_state.model)
    f_initial = client_state.model
    f_local.requires_grad_(True)

    lr_decay = 1.
    optimizer = MYOPT(f_local.parameters(), lr=lr_decay * config.local_lr)

    for epoch in range(config.local_epoch):
        for data, label in client_dataloader:
            optimizer.zero_grad()
            data = data.to(device)
            label = label.to(device)
            loss = loss_fn(f_local(data), label)

            # Now compute the quadratic part
            quad_penalty = 0.0
            for theta, theta_init in zip(f_local.parameters(), f_initial.parameters()):
                quad_penalty += F.mse_loss(theta, theta_init, reduction='sum')

            loss += quad_penalty / 2. / eta

            # Now take loss
            loss.backward()

            optimizer.step(client_state.lambda_var)

    # Update the dual variable
    print(loss.item())
    with torch.autograd.no_grad():

        lambda_delta = None
        for param_1, param_2 in zip(f_local.parameters(), f_initial.parameters()):
            if not isinstance(lambda_delta, torch.Tensor):
                lambda_delta = (param_1 - param_2).view(-1) / eta
            else:
                lambda_delta = torch.cat((lambda_delta, (param_1 - param_2).view(-1) / eta), dim=0)

        lambda_var = lambda_delta if client_state.lambda_var is None else client_state.lambda_var + lambda_delta

        # update f_local
        sd = f_local.state_dict()
        for key, param in zip(sd, lambda_var):
            sd[key] = sd[key] + eta * param
        f_local.load_state_dict(sd)

    return FEDPD_client_state(global_round=client_state.global_round, model=f_local, lambda_var=lambda_var)


class MYOPT(Optimizer):
    def __init__(self, params, lr=1e-3):
        defaults = dict(lr=lr)
        super(MYOPT, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, lambda_var):
        if lambda_var is not None:
            for p_group in self.param_groups:
                for p, l in zip(p_group['params'], lambda_var):
                    if p.grad is None:
                        continue
                    d_p = p.grad

                    p.add_(d_p + l, alpha=-p_group['lr'])
        else:
            for p_group in self.param_groups:
                for p in p_group['params']:
                    if p.grad is None:
                        continue
                    d_p = p.grad

                    p.add_(d_p, alpha=-p_group['lr'])

        return True
import torch
from torch.optim.optimizer import Optimizer
import math
import torch.distributed as dist
from swift.tuners.module_mapping import MODEL_KEYS_MAPPING


class AdamMini(Optimizer):

    def __init__(
            self,
            model_type,
            model=None,
            weight_decay=0.1,
            lr=1,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            model_sharding=False,
            n_embd=2048,
            n_head=32,
            n_query_groups=None,
            world_size=1,
    ):
        '''
        model: the model you are training.

        model_sharding: set to True if you are using model parallelism with more than 1 GPU, including FSDP and zero_1,2,3 in Deepspeed. Set to False if otherwise.

        n_embd: number of embedding dimensions. Could be unspecified if you are training non-transformer models.

        n_head: number of attention heads. Could be unspecified if you are training non-transformer models.

        n_query_groups: number of query groups in Group query Attention. If not specified, it will be equal to n_head. Could be unspecified if you are training non-transformer models.
        '''

        self.n_embd = n_embd
        self.n_head = n_head
        if n_query_groups is not None:
            self.n_query_groups = n_query_groups
            assert self.n_head % self.n_query_groups == 0
        else:
            self.n_query_groups = self.n_head

        self.model = model
        self.world_size = world_size
        self.model_sharding = model_sharding
        assert model_type in MODEL_KEYS_MAPPING, f'AdamMini does not support {model_type}'
        model_keys = MODEL_KEYS_MAPPING[model_type]
        assert model_keys.q_proj is not None or model_keys.qkv_proj is not None, 'AdamMini needs a q_proj or a qkv_proj'
        optim_groups = []
        q_proj = k_proj = qkv_proj = ''
        if model_keys.q_proj is not None:
            q_proj = model_keys.q_proj.split('{}.')[1]
            k_proj = model_keys.k_proj.split('{}.')[1]
        else:
            qkv_proj = model_keys.qkv_proj.split('{}.')[1]
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.qkv_proj = qkv_proj
        self.embed_token = model_keys.embedding
        self.output = model_keys.output
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param_group = {"name": name, "params": param}
                if "norm" in name or "ln_f" in name:
                    param_group["weight_decay"] = 0
                else:
                    param_group["weight_decay"] = weight_decay

                if q_proj and (q_proj in name or k_proj in name):
                    param_group["parameter_per_head"] = self.n_embd * self.n_embd // self.n_head

                if qkv_proj and qkv_proj in name:
                    param_group["n_head"] = self.n_head
                    param_group["q_per_kv"] = self.n_head // self.n_query_groups

                optim_groups.append(param_group)

        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon)
        super(AdamMini, self).__init__(optim_groups, defaults)

    def step(self, closure):
        with torch.no_grad():
            for group in self.param_groups:
                beta1 = group["beta1"]
                beta2 = group["beta2"]
                lr = group["lr"]
                name = group["name"]
                epsilon = group["epsilon"]

                for p in group["params"]:
                    state = self.state[p]
                    if self.embed_token in name or self.output in name:
                        if p.grad is None:
                            continue
                        if len(state) == 0:
                            state["m"] = torch.zeros_like(p.data).to(torch.float32)
                            state["iteration"] = 0
                            state["v"] = torch.zeros_like(p.data).to(torch.float32)

                        grad = p.grad.data.to(torch.float32)
                        state["v"].mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
                        state["iteration"] += 1
                        if group["weight_decay"] != 0:
                            p.data.mul_(1 - lr * group["weight_decay"])

                        state["m"].lerp_(grad, 1 - beta1)

                        bias_correction_1 = 1 - beta1 ** state["iteration"]
                        bias_correction_2 = 1 - beta2 ** state["iteration"]
                        bias_correction_2_sqrt = math.sqrt(bias_correction_2)

                        h = (state["v"].sqrt() / bias_correction_2_sqrt).add_(epsilon)
                        stepsize = lr / bias_correction_1
                        p.addcdiv_(state["m"], h, value=-stepsize)

                    elif self.q_proj and (self.q_proj in name or self.k_proj in name) and 'lora' not in name and 'bias' not in name:
                        if p.grad is None:
                            continue
                        dim = group["parameter_per_head"]
                        if len(state) == 0:
                            state["m"] = torch.zeros_like(p.data).to(torch.float32)
                            state["m"] = state["m"].view(-1, dim)
                            state['head'] = state['m'].shape[0]
                            state["iteration"] = 0
                            state["vmean"] = torch.zeros(state['head']).to(p.grad.device)

                        grad = p.grad.data.to(torch.float32)
                        head = state['head']
                        grad = grad.view(head, dim)

                        tmp_lr = torch.mean(grad * grad, dim=1).to(p.grad.device)
                        state["vmean"].mul_(beta2).add_(tmp_lr, alpha=1 - beta2)
                        v = state["vmean"]

                        state["iteration"] += 1
                        if group["weight_decay"] != 0:
                            p.data.mul_(1 - lr * group["weight_decay"])

                        state["m"].lerp_(grad, 1 - beta1)

                        bias_correction_1 = 1 - beta1 ** state["iteration"]
                        bias_correction_2 = 1 - beta2 ** state["iteration"]
                        bias_correction_2_sqrt = math.sqrt(bias_correction_2)

                        h = (v.sqrt() / bias_correction_2_sqrt).add_(epsilon)
                        stepsize = ((1 / bias_correction_1) / h).view(head, 1)

                        update = state["m"] * (stepsize.to(state['m'].device))

                        if p.dim() > 1:
                            d0, d1 = p.size()
                            update = update.view(d0, d1)
                        else:
                            update = update.view(-1)

                        update.mul_(lr)
                        p.add_(-update)

                    elif self.qkv_proj and self.qkv_proj in name:
                        if p.grad is None:
                            continue
                        if len(state) == 0:
                            state["m"] = torch.zeros_like(p.data).to(torch.float32)
                            state["m"] = state["m"].view(group["n_head"], group["q_per_kv"] + 2, -1)
                            state["iteration"] = 0
                            state["vmean"] = torch.zeros(group["n_head"], group["q_per_kv"] + 2).to(p.grad.device)

                        grad = p.grad.data.to(torch.float32)
                        grad = grad.view(group["n_head"], group["q_per_kv"] + 2, -1)

                        tmp_lr = torch.mean(grad * grad, dim=2).to(p.grad.device)
                        state["vmean"].mul_(beta2).add_(tmp_lr, alpha=1 - beta2)
                        v = state["vmean"]

                        print(f'name = {name} tmp_lr = {tmp_lr.size()}, vmean = {v.size()}')

                        state["iteration"] += 1
                        if group["weight_decay"] != 0:
                            p.data.mul_(1 - lr * group["weight_decay"])

                        state["m"].lerp_(grad, 1 - beta1)

                        bias_correction_1 = 1 - beta1 ** state["iteration"]
                        bias_correction_2 = 1 - beta2 ** state["iteration"]
                        bias_correction_2_sqrt = math.sqrt(bias_correction_2)

                        h = (v.sqrt() / bias_correction_2_sqrt).add_(epsilon)
                        stepsize = ((1 / bias_correction_1) / h).view(group["n_head"], group["q_per_kv"] + 2, 1)

                        update = state["m"] * (stepsize.to(state['m'].device))

                        if p.dim() > 1:
                            d0, d1 = p.size()
                            update = update.view(d0, d1)
                        else:
                            update = update.view(-1)

                        update.mul_(lr)
                        p.add_(-update)
                    else:
                        if len(state) == 0:
                            dimension = torch.tensor(p.data.numel()).to(p.grad.device).to(torch.float32)
                            reduced = False
                            if (self.world_size > 1) and (self.model_sharding is True):
                                tensor_list = [torch.zeros_like(dimension) for _ in range(self.world_size)]
                                dist.all_gather(tensor_list, dimension)
                                s = 0
                                dimension = 0
                                for d in tensor_list:
                                    if d > 0:
                                        s = s + 1
                                    dimension = dimension + d
                                if s >= 2:
                                    reduced = True

                            state["m"] = torch.zeros_like(p.data).to(torch.float32)
                            state["iteration"] = 0
                            state["reduced"] = reduced
                            state["vmean"] = torch.tensor(0.0).to(p.grad.device)
                            state["dimension"] = dimension.item()
                        if p.grad is None:
                            tmp_lr = torch.tensor(0.0).to(0)
                        else:
                            grad = p.grad.data.to(torch.float32)
                            tmp_lr = torch.sum(grad * grad).to(p.grad.device)
                        if state["reduced"]:
                            dist.all_reduce(tmp_lr, op=dist.ReduceOp.SUM)
                        if p.grad is None:
                            continue
                        tmp_lr = tmp_lr / (state["dimension"])
                        tmp_lr = tmp_lr.to(grad.device)

                        if group["weight_decay"] != 0:
                            p.data.mul_(1 - lr * group["weight_decay"])
                        state["iteration"] += 1
                        state["m"].lerp_(grad, 1 - beta1)

                        bias_correction_1 = 1 - beta1 ** state["iteration"]
                        bias_correction_2 = 1 - beta2 ** state["iteration"]
                        bias_correction_2_sqrt = math.sqrt(bias_correction_2)
                        state["vmean"] = (1 - beta2) * tmp_lr + beta2 * state["vmean"]
                        h = (state["vmean"].sqrt() / bias_correction_2_sqrt).add_(epsilon)

                        stepsize = (1 / bias_correction_1) / h
                        update = state["m"] * (stepsize.to(state['m'].device))
                        update.mul_(lr)
                        p.add_(-update)


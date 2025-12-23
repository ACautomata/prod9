from einops import rearrange
import torch
import random
import math

class MaskGiTSampler:
    def __init__(
        self,
        steps,
        mask_value,
        scheduler_type='log'
    ):
        super().__init__()
        self.mask_value = mask_value
        self.steps = steps
        self.scheduler = MaskGiTScheduler(
            steps=steps, 
            mask_value=mask_value
        )
        self.f = self.schedule_fatctory(scheduler_type)
    
    @torch.no_grad() 
    def step(self, step, transformer, vae, x, cond, last_indices):
        bs, s, d = x.shape
        
        logits = transformer(x, cond) - transformer(x, None)   # [B,S,V]
        conf = logits.softmax(-1).amax(-1)                     # [B,S]
        token_id = logits.argmax(-1)                           # [B,S] 预测 token id

        # mask 出候选位置：last_mask [B,S]
        last_mask = torch.zeros_like(conf, dtype=torch.bool)
        last_mask.scatter_(1, last_indices, True)

        # 只在候选里排序
        selected_conf = conf.masked_fill(~last_mask, -1)
        sorted_pos = selected_conf.argsort(dim=1, descending=True)  # [B,S]

        # 根据 schedule 选出要更新的位置
        pos = sorted_pos[:, :self.schedule(step, s)]  # [B,K]

        # 对应的 token id
        tid = token_id.gather(1, pos)          # [B,K]

        # embed 成向量并写回
        vec = vae.embed(tid)                   # [B,K,d]
        x.scatter_(1, pos.unsqueeze(-1).expand(-1, -1, d), vec)
        
        new_last_indices = []
        for b in range(last_indices.size(0)):
            diff = last_indices[b][~torch.isin(last_indices[b], pos[b])]
            new_last_indices.append(diff)

        last_indices = torch.stack(new_last_indices)
        return x, last_indices
    
    @torch.no_grad()
    def sample(self, transformer, vae, shape, cond=None):
        bs, c, h, w, d = shape
        if transformer.device != vae.device:
            raise Exception(f'{transformer.device} != {vae.device}')
        z = torch.full((bs, h * w * d, c), self.mask_value, device=transformer.device)
        last_indices = torch.arange(end=h * w * d, device=transformer.device)[None, :].repeat(bs, 1)
        for step in range(self.steps):
            z, last_indices = self.step(step, transformer, vae, z, cond, last_indices)
        z = rearrange(z, 'bs (h w d) c -> bs c h w d', h=h, w=w, d=d)
        return vae.decode(z)
        
    @torch.no_grad()
    def schedule(self, step, seq_len):
        count = int(self.f(step / self.steps) * seq_len) - int(self.f((step + 1) / self.steps) * seq_len)
        if count <= 0:
            raise ValueError(
                f"Schedule truncation: step={step}, seq_len={seq_len}, steps={self.steps}. "
                f"Calculated count={count}, which would cause no tokens to be updated. "
                f"Consider increasing seq_len or using fewer steps."
            )
        return count
    
    def schedule_fatctory(self, schedule_type):
        match schedule_type:
            case "log":
                return lambda x: math.log2(2 - x)
            case "linear":
                return lambda x: 1 - x
            case "sqrt":
                return lambda x: math.sqrt(1 - x)
            case _:
                raise Exception(f'unknown scheduler {schedule_type}')
            
        
class MaskGiTScheduler:
    def __init__(self, steps, mask_value):
        self.mask_value = mask_value
        self.steps = steps
    
    @torch.no_grad() 
    def select_indices(self, z, step):
        bs, s, d = z.shape
        ratio = self.mask_ratio(step)
        indices = torch.stack([torch.randperm(s, device=z.device) for _ in range(bs)])
        indices = indices[:, :math.ceil(ratio * s)]
        return indices
    
    @torch.no_grad()
    def generate_pair(self, z, indices):
        # mask out the selected indices
        indices = indices[:, :, None].repeat(1, 1, z.shape[2])
        z_masked = torch.scatter(
            input=z,
            dim=1,
            index=indices,
            src=torch.full(z.shape, float(self.mask_value), device=z.device)
        )
        # generate the label
        label = torch.gather(z, dim=1, index=indices)

        return z_masked, label
    
    def mask_ratio(self, step):
        # NOTICE: step \belongs [1, self.steps]
        return random.random()
        
        
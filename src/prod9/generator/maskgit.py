from einops import rearrange
import torch
import random
import math

class MaskGiTSampler:
    def __init__(
        self,
        steps,
        mask_value,
        scheduler_type='log',
        guidance_scale=0.1
    ):
        super().__init__()
        self.mask_value = mask_value
        self.steps = steps
        self.scheduler = MaskGiTScheduler(
            steps=steps,
            mask_value=mask_value
        )
        self.f = self.schedule_fatctory(scheduler_type)
        self.guidance_scale = guidance_scale

    @torch.no_grad()
    def step(self, step, transformer, vae, x, cond, last_indices, guidance_scale=None):
        """
        Single step of MaskGiT sampling with Classifier-Free Guidance.

        Uses CFG formula:
            logits = (1 + w) * f(x, cond) - w * f(x, 0)

        where:
            - w (guidance_scale): controls conditioning strength
            - f(x, cond): conditional prediction
            - f(x, 0): unconditional prediction (zero conditioning)

        Reference: Ho and Salimans, "Classifier-Free Diffusion Guidance", 2022

        Args:
            cond: Conditioning tensor. Use zero tensor for unconditional generation.
            guidance_scale: Optional override for self.guidance_scale
                           Use 0.0 for unconditional, 1.0+ for stronger guidance
        """
        s, d = x.shape[1], x.shape[2]

        # Use provided guidance_scale or fall back to default
        w = guidance_scale if guidance_scale is not None else self.guidance_scale

        # Classifier-Free Guidance formula
        logits = (1 + w) * transformer(x, cond) - w * transformer(x, torch.zeros_like(cond))   # [B,S,V]
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
    def sample(self, transformer, vae, shape, cond):
        """
        Full sampling pipeline with Classifier-Free Guidance.

        Args:
            transformer: Transformer model for token prediction
            vae: VAE model for decoding
            shape: Target shape (bs, c, h, w, d)
            cond: Conditioning tensor. Use zero tensor for unconditional generation.

        Returns:
            Generated image tensor
        """
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
        
        
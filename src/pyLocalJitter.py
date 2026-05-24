import logging
import torch

logger = logging.getLogger("DataAugmentation")

class LocalIterativeNeighborJitter:
    def __init__(self, num_iterations=3, blend_rate=0.3, gaussian_std=0.1, patch_size=14, background_thresh=0.05):
        """
        num_iterations: How many times to cascade the neighbor-averaging passes.
        blend_rate: The baseline mixing weight between paired pixels (0.1 to 0.5).
        gaussian_std: The standard deviation of the Gaussian distribution modifying the blend.
        patch_size: Desired window dimension.
        background_thresh: Pixels below this are completely excluded from all math.
        """
        self.num_iterations = num_iterations
        self.blend_rate = blend_rate
        self.gaussian_std = gaussian_std
        self.desired_patch_size = patch_size
        self.background_thresh = background_thresh
        
        self.cached_patch_size = None
        self.feather_mask = None
        self.has_warned = False

    def _get_feather_mask(self, current_patch_size, device):
        if self.cached_patch_size == current_patch_size and self.feather_mask is not None:
            return self.feather_mask.to(device)
            
        coords = torch.linspace(-1.5, 1.5, current_patch_size)
        x, y = torch.meshgrid(coords, coords, indexing="ij")
        mask = torch.exp(-(x**2 + y**2))
        normalized_mask = (mask - mask.min()) / (mask.max() - mask.min())
        
        self.cached_patch_size = current_patch_size
        self.feather_mask = normalized_mask
        return self.feather_mask.to(device)

    def __call__(self, batch_tensor):
        b, c, h, w = batch_tensor.shape
        output_batch = batch_tensor.clone().detach()
        
        max_possible_size = min(h, w)
        active_patch_size = min(self.desired_patch_size, max_possible_size)
        
        if self.desired_patch_size > max_possible_size and not self.has_warned:
            logger.warning(f"[⚠️ Pipeline Adjustment] Clamping active patch_size down to {active_patch_size}.")
            self.has_warned = True
            
        feather_2d = self._get_feather_mask(active_patch_size, batch_tensor.device)
        feather = feather_2d.unsqueeze(0) 
        
        for i in range(b):
            y = torch.randint(0, h - active_patch_size + 1, (1,)).item()
            x = torch.randint(0, w - active_patch_size + 1, (1,)).item()
            
            original_patch = batch_tensor[i, :, y:y+active_patch_size, x:x+active_patch_size].clone().detach()
            fg_mask = (original_patch > self.background_thresh).any(dim=0, keepdim=True)
            
            if not fg_mask.any():
                continue
                
            current_patch = original_patch.clone()
            
            # --- ITERATIVE GAUSSIAN-VARIATION NEIGHBOR BLENDING ---
            for _ in range(self.num_iterations):
                axis = torch.randint(0, 2, (1,)).item()
                shift_direction = 1 if torch.rand((1,)).item() > 0.5 else -1
                
                neighbor_patch = current_patch.clone()
                if axis == 0:  
                    if shift_direction == 1:
                        neighbor_patch[:, 1:, :] = current_patch[:, :-1, :]
                    else:
                        neighbor_patch[:, :-1, :] = current_patch[:, 1:, :]
                else:          
                    if shift_direction == 1:
                        neighbor_patch[:, :, 1:] = current_patch[:, :, :-1]
                    else:
                        neighbor_patch[:, :, :-1] = current_patch[:, :, 1:]
                
                neighbor_fg_mask = (neighbor_patch > self.background_thresh).any(dim=0, keepdim=True)
                valid_blend_idx = fg_mask & neighbor_fg_mask
                
                # 1. DRAW FROM GAUSSIAN DISTRIBUTION
                # We generate a unique Gaussian modifier for every single pixel coordinate
                gaussian_modifier = torch.randn_like(current_patch) * self.gaussian_std
                
                # 2. DYNAMICALLY FLUCTUATE THE BLEND RATE
                # The effective blend weight varies normally around your baseline blend_rate
                dynamic_weight = self.blend_rate + gaussian_modifier
                
                # Clamp the weight strictly between 0.0 and 1.0 so it stays a valid interpolator
                dynamic_weight = torch.clamp(dynamic_weight, 0.0, 1.0)
                
                # 3. CONVERT TO GAUSSIAN NEIGHBOR INTERPOLATION
                # This draws variations directly from a normal distribution about the local mean
                iterated_step = (current_patch * (1.0 - dynamic_weight)) + (neighbor_patch * dynamic_weight)
                
                current_patch[valid_blend_idx.expand_as(current_patch)] = iterated_step[valid_blend_idx.expand_as(current_patch)]
            
            # Apply Gaussian feathering mask exclusively to the foreground object
            soft_blend = (current_patch * feather) + (original_patch * (1.0 - feather))
            
            final_mask = fg_mask.expand_as(original_patch)
            output_batch[i, :, y:y+active_patch_size, x:x+active_patch_size][final_mask] = \
                torch.clamp(soft_blend[final_mask], 0.0, 1.0)
            
        return output_batch

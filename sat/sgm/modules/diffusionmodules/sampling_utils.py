import math
import torch
from scipy import integrate

from ...util import append_dims


class NoDynamicThresholding:
    def __call__(self, uncond, cond, scale, **kwargs):
        return uncond + scale * (cond - uncond)

class RescaleThresholding:
    def __init__(self, phi=0.7):
        self.phi = phi

    def __call__(self, uncond, cond, scale, **kwargs):
        denoised_cfg = uncond + scale * (cond - uncond)
        sigma_pos, sigma_cfg = cond.std(), denoised_cfg.std()
        factor = self.phi * sigma_pos / sigma_cfg + (1 - self.phi)
        denoised_final = denoised_cfg * factor
        return denoised_final
    
class DynamicThresholding:
    Modes = ["Constant", "Linear Up", "Linear Down", "Half Cosine Up", "Half Cosine Down", "Power Up", "Power Down", "Cosine Down","Cosine Up"]
    def __init__(self, interpret_mode, 
                 scale_min = 3,
                 mimic_interpret_mode = 'Constant',
                 mimic_scale = 3, 
                 mimic_scale_min = 3, 
                 threshold_percentile = 1.0,
                 phi = 1.0,
                 separate_feature_channels = True,
                 measure = 'AD',
                 scaling_startpoint = 'ZERO',
                 ):
        assert interpret_mode in self.Modes
        assert mimic_interpret_mode in self.Modes
        assert measure in ['AD', 'STD']
        assert scaling_startpoint in ['ZERO', 'MEAN']
        self.mode = interpret_mode
        self.mimic_mode = mimic_interpret_mode
        self.scale_min = scale_min
        self.mimic_scale = mimic_scale
        self.mimic_scale_min = mimic_scale_min
        self.threshold_percentile = threshold_percentile
        self.phi = phi
        self.separate_feature_channels = separate_feature_channels
        self.measure = measure
        self.scaling_startpoint = scaling_startpoint
    
    def interpret_scale(self, mode, scale, scale_min, step, num_steps):
        """
        num_steps = 50
        step from 0 to 50.
        """
        scale -= scale_min
        frac = step / num_steps
        if mode == 'Constant':
            pass
        elif mode == "Linear Up":
            scale *= frac
        elif mode == "Linear Down":
            scale *= 1.0 - frac
        elif mode == "Half Cosine Up":
            scale *= 1.0 - math.cos(frac)
        elif mode == "Half Cosine Down":
            scale *= math.cos(frac)
        elif mode == "Cosine Down":
            scale *= math.cos(frac * 1.5707)
        elif mode == "Cosine Up":
            scale *= 1.0 - math.cos(frac * 1.5707)
        elif mode == "Power Up":
            scale *= math.pow(frac, 2.0)
        elif mode == "Power Down":
            scale *= 1.0 - math.pow(frac, 2.0)
        scale += scale_min
        return scale
    
    def __call__(self, uncond, cond, scale, step, num_steps):
        cfg_scale = self.interpret_scale(self.mode, scale, self.scale_min, step, num_steps)
        mimic_cfg_scale = self.interpret_scale(self.mimic_mode, self.mimic_scale, self.mimic_scale_min, step, num_steps)

        x = uncond + cfg_scale*(cond - uncond)
        mimic_x = uncond + mimic_cfg_scale*(cond - uncond)  

        x_flattened = x.flatten(2)
        mimic_x_flattened = mimic_x.flatten(2)
        
        if self.scaling_startpoint == 'MEAN':
            x_means = x_flattened.mean(dim=2, keepdim = True)
            mimic_x_means = mimic_x_flattened.mean(dim=2, keepdim = True)
            x_centered = x_flattened - x_means
            mimic_x_centered = mimic_x_flattened - mimic_x_means
        else:
            x_centered = x_flattened
            mimic_x_centered = mimic_x_flattened
            
        if self.separate_feature_channels:
            if self.measure == 'AD':
                x_scaleref = torch.quantile(x_centered.abs(), self.threshold_percentile, dim=2, keepdim = True)
                mimic_x_scaleref = mimic_x_centered.abs().max(dim=2, keepdim = True).values
            elif self.measure == 'STD':
                x_scaleref = x_centered.std(dim=2, keepdim = True)
                mimic_x_scaleref = mimic_x_centered.std(dim=2, keepdim = True)
        else:
            if self.measure == 'AD':
                x_scaleref = torch.quantile(x_centered.abs(), self.threshold_percentile)
                mimic_x_scaleref = mimic_x_centered.abs().max()
            elif self.measure == 'STD':
                x_scaleref = x_centered.std()
                mimic_x_scaleref = mimic_x_centered.std()
        
        if self.measure == 'AD':
            max_scaleref = torch.maximum(x_scaleref, mimic_x_scaleref)
            x_clamped = x_centered.clamp(-max_scaleref, max_scaleref)
            x_renormed = x_clamped * (mimic_x_scaleref / max_scaleref)
        elif self.measure == 'STD':
            x_renormed = x_centered * (mimic_x_scaleref / x_scaleref)
        
        if self.scaling_startpoint == 'MEAN':
            x_dyn = x_means + x_renormed
        else:
            x_dyn = x_renormed
        x_dyn = x_dyn.unflatten(2, x.shape[2:])
        return self.phi*x_dyn + (1-self.phi)*x
        

def linear_multistep_coeff(order, t, i, j, epsrel=1e-4):
    if order - 1 > i:
        raise ValueError(f"Order {order} too high for step {i}")

    def fn(tau):
        prod = 1.0
        for k in range(order):
            if j == k:
                continue
            prod *= (tau - t[i - k]) / (t[i - j] - t[i - k])
        return prod

    return integrate.quad(fn, t[i], t[i + 1], epsrel=epsrel)[0]


def get_ancestral_step(sigma_from, sigma_to, eta=1.0):
    if not eta:
        return sigma_to, 0.0
    sigma_up = torch.minimum(
        sigma_to,
        eta
        * (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5,
    )
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    return sigma_down, sigma_up


def to_d(x, sigma, denoised):
    return (x - denoised) / append_dims(sigma, x.ndim)


def to_neg_log_sigma(sigma):
    return sigma.log().neg()


def to_sigma(neg_log_sigma):
    return neg_log_sigma.neg().exp()

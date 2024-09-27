import torch

from ...util import default, instantiate_from_config


class EDMSampling:
    def __init__(self, p_mean=-1.2, p_std=1.2):
        self.p_mean = p_mean
        self.p_std = p_std

    def __call__(self, n_samples, rand=None):
        log_sigma = self.p_mean + self.p_std * default(rand, torch.randn((n_samples,)))
        return log_sigma.exp()


class DiscreteSampling:
    def __init__(self, discretization_config, num_idx, do_append_zero=False, flip=True, low_bound=0, up_bound=1):
        self.num_idx = num_idx
        self.sigmas = instantiate_from_config(discretization_config)(
            num_idx, do_append_zero=do_append_zero, flip=flip
        )
        self.low_bound = int(low_bound * num_idx)
        self.up_bound = int(up_bound * num_idx)
        print(f'sigma sampling from {self.low_bound} to {self.up_bound}')

    def idx_to_sigma(self, idx):
        return self.sigmas[idx]

    def __call__(self, n_samples, rand=None, return_idx=False):
        idx = default(
            rand,
            torch.randint(self.low_bound, self.up_bound, (n_samples,)),
        )
        if return_idx:
            return self.idx_to_sigma(idx), idx
        else:
            return self.idx_to_sigma(idx)

class PartialDiscreteSampling:
    def __init__(self, discretization_config, total_num_idx, partial_num_idx, do_append_zero=False, flip=True):
        self.total_num_idx = total_num_idx
        self.partial_num_idx = partial_num_idx
        self.sigmas = instantiate_from_config(discretization_config)(
            total_num_idx, do_append_zero=do_append_zero, flip=flip
        )

    def idx_to_sigma(self, idx):
        return self.sigmas[idx]

    def __call__(self, n_samples, rand=None):
        idx = default(
            rand,
            # torch.randint(self.total_num_idx-self.partial_num_idx, self.total_num_idx, (n_samples,)),
            torch.randint(0, self.partial_num_idx, (n_samples,)),
        )
        return self.idx_to_sigma(idx)
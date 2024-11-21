import numpy as np
import torch as th

from .blurring import dct_2d, idct_2d


def expand_t_like_x(t, x):
    """Function to reshape time t to broadcastable dimension of x
    Args:
      t: [batch_dim,], time vector
      x: [batch_dim,...], data point
    """
    dims = [1] * (len(x.size()) - 1)
    t = t.view(t.size(0), *dims)
    return t


#################### Coupling Plans ####################


class ICPlan:
    """Linear Coupling Plan"""

    def __init__(self, sigma=0.0, diffusion_form="none", use_blurring=False, blur_sigma_max=3, blur_upscale=4):
        self.sigma = sigma
        self.diffusion_form = diffusion_form
        self.use_blurring = use_blurring
        self.blur_sigma_max = blur_sigma_max
        self.blur_upscale = blur_upscale

    def compute_alpha_t(self, t):
        """Compute the data coefficient along the path"""
        return t, 1

    def compute_sigma_t(self, t):
        """Compute the noise coefficient along the path"""
        return 1 - t, -1

    def compute_d_alpha_alpha_ratio_t(self, t):
        """Compute the ratio between d_alpha and alpha"""
        return 1 / t

    def compute_drift(self, x, t):
        """We always output sde according to score parametrization;"""
        t = expand_t_like_x(t, x)
        alpha_ratio = self.compute_d_alpha_alpha_ratio_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        drift = alpha_ratio * x
        diffusion = alpha_ratio * (sigma_t**2) - sigma_t * d_sigma_t

        return -drift, diffusion

    def compute_diffusion(self, x, t, form="constant", norm=1.0):
        """Compute the diffusion term of the SDE
        Args:
          x: [batch_dim, ...], data point
          t: [batch_dim,], time vector
          form: str, form of the diffusion term
          norm: float, norm of the diffusion term
        """
        t = expand_t_like_x(t, x)
        choices = {
            "none": th.zeros((1,), device=t.device),
            "constant": th.full((1,), norm, device=t.device),
            "SBDM": norm * 2.0 * self.compute_drift(x, t)[1],
            "sigma": norm * self.compute_sigma_t(t)[0],
            "linear": norm * (1 - t),
            "decreasing": 0.25 * (norm * th.cos(np.pi * t) + 1) ** 2,
            "increasing-decreasing": norm * th.sin(np.pi * t) ** 2,
            "log": norm * th.log(t - t**2 + 1),
        }

        try:
            diffusion = choices[form]
        except KeyError:
            raise NotImplementedError(f"Diffusion form {form} not implemented")

        return diffusion

    def compute_d_diffusion(self, x, t, form="constant", norm=1.0):
        """Compute the derivative of diffusion term of the SDE
        Args:
          x: [batch_dim, ...], data point
          t: [batch_dim,], time vector
          form: str, form of the diffusion term
          norm: float, norm of the diffusion term
        """
        t = expand_t_like_x(t, x)
        choices = {
            "none": th.zeros((1,), device=t.device),
            "constant": th.zeros((1,), device=t.device),
            # "SBDM": norm * 2. * self.compute_drift(x, t)[1],
            # "sigma": norm * self.compute_sigma_t(t)[0],
            "linear": th.full((1,), -norm),
            "decreasing": -0.5
            * np.pi
            * norm
            * th.sin(np.pi * t)
            * (norm * th.cos(np.pi * t) + 1),  # 0.25 * (norm * th.cos(np.pi * t) + 1) ** 2,
            "increasing-decreasing": 2
            * norm
            * np.pi
            * th.sin(np.pi * t)
            * th.cos(np.pi * t),  # norm * th.sin(np.pi * t) ** 2,
            "log": norm * (1 - 2 * t) / (t - t**2 + 1),
        }

        try:
            diffusion = choices[form]
        except KeyError:
            raise NotImplementedError(f"Diffusion form {form} not implemented")

        return diffusion

    def get_score_from_velocity(self, velocity, x, t):
        """Wrapper function: transfrom velocity prediction model to score
        Args:
            velocity: [batch_dim, ...] shaped tensor; velocity model output
            x: [batch_dim, ...] shaped tensor; x_t data point
            t: [batch_dim,] time tensor
        """
        t = expand_t_like_x(t, x)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        mean = x
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
        score = (reverse_alpha_ratio * velocity - mean) / var
        return score

    def get_noise_from_velocity(self, velocity, x, t):
        """Wrapper function: transfrom velocity prediction model to denoiser
        Args:
            velocity: [batch_dim, ...] shaped tensor; velocity model output
            x: [batch_dim, ...] shaped tensor; x_t data point
            t: [batch_dim,] time tensor
        """
        t = expand_t_like_x(t, x)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        mean = x
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = reverse_alpha_ratio * d_sigma_t - sigma_t
        noise = (reverse_alpha_ratio * velocity - mean) / var
        return noise

    def get_velocity_from_score(self, score, x, t):
        """Wrapper function: transfrom score prediction model to velocity
        Args:
            score: [batch_dim, ...] shaped tensor; score model output
            x: [batch_dim, ...] shaped tensor; x_t data point
            t: [batch_dim,] time tensor
        """
        t = expand_t_like_x(t, x)
        drift, var = self.compute_drift(x, t)
        velocity = var * score - drift
        return velocity

    def compute_mu_t(self, t, x0, x1):
        """Compute the mean of time-dependent density p_t"""
        t = expand_t_like_x(t, x1)
        alpha_t, _ = self.compute_alpha_t(t)
        sigma_t, _ = self.compute_sigma_t(t)
        if self.use_blurring:
            blur_sigmas = (
                self.blur_sigma_max * th.sin(sigma_t * th.pi / 2) ** 2
            )  # importance: need to reverse t as we go from 0->1 (our data) and larger t induces greater blur levels
            x1 = DCTBlur(x1, self.blur_upscale, blur_sigmas, 1e-3, x1.device)
        return alpha_t * x1 + sigma_t * x0

    def compute_xt(self, t, x0, x1):
        """Sample xt from time-dependent density p_t; rng is required"""
        xt = self.compute_mu_t(t, x0, x1)
        return xt

    def compute_ut(self, t, x0, x1, xt):
        """Compute the vector field corresponding to p_t"""
        t = expand_t_like_x(t, x1)
        _, d_alpha_t = self.compute_alpha_t(t)
        _, d_sigma_t = self.compute_sigma_t(t)
        return d_alpha_t * x1 + d_sigma_t * x0

    def plan(self, t, x0, x1):
        xt = self.compute_xt(t, x0, x1)
        ut = self.compute_ut(
            t, x0, x1, xt
        )
        return t, xt, ut


class VPCPlan(ICPlan):
    """class for VP path flow matching"""

    def __init__(self, sigma_min=0.1, sigma_max=20.0, **kwargs):
        super().__init__(**kwargs)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.log_mean_coeff = (
            lambda t: -0.25 * ((1 - t) ** 2) * (self.sigma_max - self.sigma_min) - 0.5 * (1 - t) * self.sigma_min
        )
        self.d_log_mean_coeff = lambda t: 0.5 * (1 - t) * (self.sigma_max - self.sigma_min) + 0.5 * self.sigma_min

    def compute_alpha_t(self, t):
        """Compute coefficient of x1"""
        alpha_t = self.log_mean_coeff(t)
        alpha_t = th.exp(alpha_t)
        d_alpha_t = alpha_t * self.d_log_mean_coeff(t)
        return alpha_t, d_alpha_t

    def compute_sigma_t(self, t):
        """Compute coefficient of x0"""
        p_sigma_t = 2 * self.log_mean_coeff(t)
        sigma_t = th.sqrt(1 - th.exp(p_sigma_t))
        d_sigma_t = th.exp(p_sigma_t) * (2 * self.d_log_mean_coeff(t)) / (-2 * sigma_t)
        return sigma_t, d_sigma_t

    def compute_d_alpha_alpha_ratio_t(self, t):
        """Special purposed function for computing numerical stabled d_alpha_t / alpha_t"""
        return self.d_log_mean_coeff(t)

    def compute_drift(self, x, t):
        """Compute the drift term of the SDE"""
        t = expand_t_like_x(t, x)
        beta_t = self.sigma_min + (1 - t) * (self.sigma_max - self.sigma_min)
        return -0.5 * beta_t * x, beta_t / 2


class GVPCPlan(ICPlan):
    def __init__(self, sigma=0.0, **kwargs):
        super().__init__(sigma, **kwargs)

    def compute_alpha_t(self, t):
        """Compute coefficient of x1"""
        alpha_t = th.sin(t * np.pi / 2)
        d_alpha_t = np.pi / 2 * th.cos(t * np.pi / 2)
        return alpha_t, d_alpha_t

    def compute_sigma_t(self, t):
        """Compute coefficient of x0"""
        sigma_t = th.cos(t * np.pi / 2)
        d_sigma_t = -np.pi / 2 * th.sin(t * np.pi / 2)
        return sigma_t, d_sigma_t

    def compute_d_alpha_alpha_ratio_t(self, t):
        """Special purposed function for computing numerical stabled d_alpha_t / alpha_t"""
        return np.pi / (2 * th.tan(t * np.pi / 2))


def DCTBlur(x, patch_size, blur_sigmas, min_scale, device):
    blur_sigmas = th.as_tensor(blur_sigmas).to(device)
    freqs = th.pi * th.linspace(0, patch_size - 1, patch_size).to(device) / patch_size
    frequencies_squared = freqs[:, None] ** 2 + freqs[None, :] ** 2

    t = blur_sigmas**2 / 2

    dct_coefs = dct_2d(x, patch_size, norm="ortho")
    scale = x.shape[-1] // patch_size
    dct_coefs = dct_coefs * (th.exp(-frequencies_squared.repeat(scale, scale) * t) * (1 - min_scale) + min_scale)
    return idct_2d(dct_coefs, patch_size, norm="ortho")

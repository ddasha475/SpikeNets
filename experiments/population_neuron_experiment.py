"""
Population Neuron Experiment: Binary Classification via Competitive First-Spike Decoding (TTFS)
Model: 2-neuron ExpNeuron population vs LIF baseline.
Task: Classify whether lambda >= Lambda.
Neuron 0 preferred center C_0 < Lambda; Neuron 1 preferred center C_1 > Lambda.
First-Spike Decoding: whichever neuron fires first wins.
Only synaptic weights w = [w_0, w_1] are trained.
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from SNN.ExpNeuron import ExpNeuron

# Precomputed Gaver-Stehfest coefficients (order N=14)
GS_COEFFS = torch.tensor([
     0.002777777777777778, -3.201388888888889,   308.01666666666665,
    -8649.481944444445,   108064.22222222222,   -733057.7277777778,
     3012513.111111111,   -7993114.130555555,    14177508.838888889,
    -17013718.808333334,   13666133.366666666,   -7049346.791666667,
     2113760.3666666665,    -280396.7833333333], dtype=torch.float64)

def make_fourier_grid(x: float, n_xi: int = 256):
    xi_min = 1e-6
    xi_max = 100.0 / max(x, 1e-4)
    xi   = torch.logspace(np.log10(xi_min), np.log10(xi_max), n_xi, dtype=torch.float64)
    dlog = torch.log(xi[1]) - torch.log(xi[0])
    dxi  = xi * dlog
    dxi[0]  *= 0.5
    dxi[-1] *= 0.5
    return xi, dxi

def find_phi_newton(q: float, beta: float, ws: torch.Tensor, lambdas: torch.Tensor, n_iter: int = 8):
    q_t = torch.tensor(q, dtype=torch.float64)
    with torch.no_grad():
        theta = (q_t + lambdas.sum()) / beta
    for _ in range(n_iter):
        exp_wt   = torch.exp(-theta * ws)
        psi      = beta * theta + (lambdas * (exp_wt - 1.0)).sum()
        dpsi     = beta          - (lambdas * ws * exp_wt).sum()
        theta    = theta - (psi - q_t) / (dpsi + 1e-12)
    return theta

def psi_Y_complex(xi: torch.Tensor, beta: float, w: torch.Tensor, lambd: torch.Tensor):
    xi_c   = xi.to(torch.complex128)
    w_c    = w.to(torch.complex128)
    lam_c  = lambd.to(torch.complex128)
    phase  = torch.outer(xi_c, w_c)
    return -1j * beta * xi_c + (lam_c * (1.0 - torch.exp(-1j * phase))).sum(dim=-1)

def laplace_transform(q: float, x: float, beta: float, ws: torch.Tensor, lambdas: torch.Tensor, xi: torch.Tensor, dxi: torch.Tensor):
    q_t = torch.tensor(q, dtype=torch.float64)
    phi_q = find_phi_newton(q, beta, ws, lambdas)
    p0 = q_t / (beta * phi_q)
    Psi = psi_Y_complex(xi, beta, ws, lambdas)
    xi_c      = xi.to(torch.complex128)
    phi_q_c   = phi_q.to(torch.complex128)
    phi_minus = (q_t / phi_q_c) * (phi_q_c - 1j * xi_c) / (q_t + Psi)
    tilde_phi = phi_minus - p0.to(torch.complex128)
    integrand = torch.imag(torch.exp(1j * xi_c * x) * tilde_phi) / xi
    integral = (integrand * dxi).sum()
    L = (1.0 - p0) / 2.0 - integral / np.pi
    return L.clamp(0.0, 1.0)

def crossing_barrier_prob(U0: float, U_th: float, T: float, beta: float, lambdas: torch.Tensor, ws: torch.Tensor, n_xi: int = 256):
    x       = float(np.log(U_th / U0))
    xi, dxi = make_fourier_grid(x, n_xi=n_xi)
    ln2_T   = np.log(2.0) / T
    L_vals = torch.stack([
        laplace_transform(k * ln2_T, x, beta, ws, lambdas, xi, dxi)
        for k in range(1, 15)])
    return (GS_COEFFS * L_vals).sum().clamp(0.0, 1.0)

class PopulationEncoder(nn.Module):
    """
    Gaussian tuning curve encoder: maps scalar stimulus lambda to vector [lambda_0, lambda_1].
    lambda_i(x) = lambda_scale * exp(-(x - mu_i)^2 / (2 * sigma^2))
    """
    def __init__(self, n_neurons=2, mu=None, sigma=0.17, lambda_scale=1.0):
        super().__init__()
        if mu is None:
            self.mu = torch.tensor([0.34, 0.76], dtype=torch.float64)
        else:
            self.mu = torch.as_tensor(mu, dtype=torch.float64)
        self.sigma = sigma
        self.lambda_scale = lambda_scale

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float64)
        if x.dim() == 0:
            x = x.unsqueeze(0)
        # x: (B,) or (B, 1), mu: (N,)
        diff = x.unsqueeze(-1) - self.mu.unsqueeze(0)
        return self.lambda_scale * torch.exp(-diff**2 / (2 * self.sigma**2))

class PopulationNet(nn.Module):
    """
    2-neuron ExpNeuron population model.
    Forward accepts Poisson spike train (batch, steps, 2) and returns output spikes.
    """
    def __init__(self, weights, steps, beta=0.01, threshold=1.0, membrane_zero=0.85):
        super().__init__()
        self.weights = weights
        self.steps = steps
        self.neuron = ExpNeuron(
            beta=beta,
            threshold=threshold,
            membrane_zero=membrane_zero,
            learn_beta=False,
            learn_threshold=False,
            learn_membrane_zero=False
        )

    def forward(self, spk_input):
        # spk_input: (batch, steps, N_neurons)
        batch_size, steps, n_neurons = spk_input.shape
        self.neuron.init_neuron()

        spk_out = []
        mem_out = []
        w = self.weights.detach() if isinstance(self.weights, torch.Tensor) else torch.tensor(self.weights)

        for t in range(steps):
            inp = spk_input[:, t, :] * w
            spk, mem = self.neuron(inp)
            spk_out.append(spk)
            mem_out.append(mem)

        spk_out = torch.stack(spk_out, dim=1) # (batch, steps, n_neurons)
        mem_out = torch.stack(mem_out, dim=1)
        return spk_out, mem_out

def decode_first_spike_ttfs(spk_TN):
    """
    Time-To-First-Spike (TTFS) decoding for 2 neurons:
    spk_TN: (steps, 2)
    Returns:
      predicted_class: 0 or 1, or None if neither fired.
      (t0, t1): time of first spike for neuron 0 and neuron 1.
    """
    steps, n_neurons = spk_TN.shape
    has_spikes = (spk_TN > 0)
    counts = has_spikes.sum(dim=0)

    if counts.sum() == 0:
        return None, (steps + 1, steps + 1)

    # Find first spike times
    t0 = (has_spikes[:, 0].nonzero()[0].item() + 1) if counts[0] > 0 else (steps + 1)
    t1 = (has_spikes[:, 1].nonzero()[0].item() + 1) if counts[1] > 0 else (steps + 1)

    if counts[0] > 0 and counts[1] == 0:
        return 0, (t0, t1)
    elif counts[1] > 0 and counts[0] == 0:
        return 1, (t0, t1)
    else:
        # Both fired: winner is the one with smaller latency (TTFS)
        if t1 < t0:
            return 1, (t0, t1)
        elif t0 < t1:
            return 0, (t0, t1)
        else:
            # Exact tie at first step: fall back to total count or higher intensity class
            return 1 if counts[1] >= counts[0] else 0, (t0, t1)

def evaluate_population_ttfs(net, test_trains, test_labels):
    """
    Evaluate PopulationNet on test spike trains using TTFS.
    """
    correct = 0
    none_count = 0
    total = len(test_labels)

    spk_output, _ = net(test_trains)

    for i in range(total):
        pred, (t0, t1) = decode_first_spike_ttfs(spk_output[i])
        lbl = int(test_labels[i].item())
        if pred is None:
            none_count += 1
        elif pred == lbl:
            correct += 1

    accuracy = correct / total
    none_rate = none_count / total
    return accuracy, none_rate

def train_population_analytical(encoder, weights, train_lambdas, train_labels,
                                epochs=15, lr=5e-4, beta=0.01, threshold=1.0,
                                membrane_zero=0.85, steps=200):
    """
    Train weights w = [w_0, w_1] using analytical crossing barrier probabilities.
    Target for lambda < Lambda: [1, 0] (neuron 0 should fire, neuron 1 should not).
    Target for lambda >= Lambda: [0, 1] (neuron 1 should fire, neuron 0 should not).
    """
    optimizer = torch.optim.Adam([weights], lr=lr)
    targets_c0 = torch.tensor([1.0, 0.0], dtype=torch.float64)
    targets_c1 = torch.tensor([0.0, 1.0], dtype=torch.float64)

    losses = []
    w_history = [weights.detach().clone().numpy()]

    for epoch in range(epochs):
        sample_losses = []
        perm = torch.randperm(len(train_lambdas))
        for idx in perm:
            lam = train_lambdas[idx]
            lbl = train_labels[idx]

            optimizer.zero_grad()
            lam_i = encoder(lam).squeeze(0) # (2,)

            p0 = crossing_barrier_prob(membrane_zero, threshold, steps, beta,
                                       lambdas=lam_i[0:1], ws=weights[0:1])
            p1 = crossing_barrier_prob(membrane_zero, threshold, steps, beta,
                                       lambdas=lam_i[1:2], ws=weights[1:2])
            probs = torch.stack([p0, p1])

            target = targets_c1 if lbl.item() == 1.0 else targets_c0
            loss = ((probs - target) ** 2).mean()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                weights.clamp_(min=0.01, max=5.0)

            sample_losses.append(loss.item())

        mean_loss = float(np.mean(sample_losses))
        losses.append(mean_loss)
        w_history.append(weights.detach().clone().numpy())

    return weights, losses, np.array(w_history)

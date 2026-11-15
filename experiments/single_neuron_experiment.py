"""
Single Neuron Experiment: Binary Classification of Poisson Spike Train Intensity
Model: ExpNeuron with First-Spike Latency Decoding.
Task: Classify whether lambda >= Lambda.
Only synaptic weight w is trained.
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Ensure SNN package can be imported
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
    """
    Theoretical first-passage probability P(tau <= T) via Wiener-Hopf + Gaver-Stehfest.
    """
    x       = float(np.log(U_th / U0))
    xi, dxi = make_fourier_grid(x, n_xi=n_xi)
    ln2_T   = np.log(2.0) / T
    L_vals = torch.stack([
        laplace_transform(k * ln2_T, x, beta, ws, lambdas, xi, dxi)
        for k in range(1, 15)])
    return (GS_COEFFS * L_vals).sum().clamp(0.0, 1.0)

def simulate_single_neuron(w, lambdas_arr, steps, beta, threshold, membrane_zero, seed=42):
    """
    Simulate single ExpNeuron on Poisson spike trains for a batch of lambdas.
    Returns:
      spikes: (N_samples, steps)
      first_spike_times: (N_samples,) with T+1 if no spike
      membrane_traces: (N_samples, steps)
    """
    torch.manual_seed(seed)
    N_samples = len(lambdas_arr)
    # Generate Poisson spike trains: shape (N_samples, steps, 1)
    spk_trains = torch.poisson(lambdas_arr.view(N_samples, 1, 1).expand(-1, steps, 1))

    neuron = ExpNeuron(
        beta=beta,
        threshold=threshold,
        membrane_zero=membrane_zero,
        learn_beta=False,
        learn_threshold=False,
        learn_membrane_zero=False
    )
    neuron.init_neuron()

    w_val = w.detach().item() if isinstance(w, torch.Tensor) else float(w)

    spk_out = []
    mem_out = []
    for t in range(steps):
        inp = spk_trains[:, t, :] * w_val
        spk, mem = neuron(inp)
        spk_out.append(spk.squeeze(-1))
        mem_out.append(mem.squeeze(-1))

    spk_out = torch.stack(spk_out, dim=1) # (N_samples, steps)
    mem_out = torch.stack(mem_out, dim=1)

    # First spike time: index of first 1, or steps+1 if no spike
    first_spikes = []
    for i in range(N_samples):
        idx = (spk_out[i] > 0).nonzero()
        if len(idx) > 0:
            first_spikes.append(float(idx[0].item() + 1))
        else:
            first_spikes.append(float(steps + 1))
    first_spikes = torch.tensor(first_spikes, dtype=torch.float64)

    return spk_out, first_spikes, mem_out

def train_single_neuron_analytical(train_lambdas, train_labels, epochs=25, lr=1e-3,
                                    beta=0.01, threshold=1.0, membrane_zero=0.85, steps=200):
    """
    Train single weight w via exact analytical first-passage probability loss.
    """
    w = nn.Parameter(torch.tensor([0.25], dtype=torch.float64))
    optimizer = torch.optim.Adam([w], lr=lr)

    losses = []
    w_history = [w.item()]

    for epoch in range(epochs):
        perm = torch.randperm(len(train_lambdas))
        epoch_loss = 0.0
        for idx in perm:
            lam = train_lambdas[idx:idx+1]
            lbl = train_labels[idx:idx+1]

            optimizer.zero_grad()
            p_fire = crossing_barrier_prob(
                membrane_zero, threshold, steps, beta,
                lambdas=lam, ws=w
            )
            # MSE loss against binary label
            loss = (p_fire - lbl) ** 2
            loss.backward()
            optimizer.step()

            # Weight must remain positive (excitatory)
            with torch.no_grad():
                w.clamp_(min=0.01, max=5.0)

            epoch_loss += loss.item()

        epoch_loss /= len(train_lambdas)
        losses.append(epoch_loss)
        w_history.append(w.item())

    return w, losses, w_history

def calibrate_tau_cutoff(w, train_lambdas, train_labels, steps, beta, threshold, membrane_zero, seed=42):
    """
    Calibrate the optimal latency threshold tau* on the training set.
    Decision rule: predict 1 if first_spike <= tau*, else 0.
    """
    _, first_spikes, _ = simulate_single_neuron(
        w, train_lambdas, steps, beta, threshold, membrane_zero, seed=seed
    )
    best_acc = 0.0
    best_tau = steps
    for tau in range(1, steps + 1):
        preds = (first_spikes <= tau).float()
        acc = (preds == train_labels).sum().item() / len(train_labels)
        if acc > best_acc:
            best_acc = acc
            best_tau = tau
    return best_tau, best_acc

def evaluate_first_spike_decoding(w, test_lambdas, test_labels, steps, beta, threshold, membrane_zero,
                                  tau_cutoff=None, seed=42):
    """
    Evaluate binary classification using First-Spike Decoding.
    Decision: if first_spike_time <= tau_cutoff -> class 1 (lambda >= Lambda)
              else -> class 0 (lambda < Lambda)
    If tau_cutoff is None, tau_cutoff = steps (any spike within window T).
    """
    if tau_cutoff is None:
        tau_cutoff = steps

    _, first_spikes, _ = simulate_single_neuron(
        w, test_lambdas, steps, beta, threshold, membrane_zero, seed=seed
    )

    preds = (first_spikes <= tau_cutoff).float()
    correct = (preds == test_labels).sum().item()
    acc = correct / len(test_labels)

    # Calculate confusion matrix components
    tp = ((preds == 1) & (test_labels == 1)).sum().item()
    fp = ((preds == 1) & (test_labels == 0)).sum().item()
    tn = ((preds == 0) & (test_labels == 0)).sum().item()
    fn = ((preds == 0) & (test_labels == 1)).sum().item()

    return acc, first_spikes, {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}

"""
Comprehensive Experiment Suite for SpikeNets SNN Binary Classification
with First-Spike Decoding.

Generates data, runs parameter sweeps, saves quantitative metrics and plots.
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Matplotlib configuration for clean scientific plots
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.0

from single_neuron_experiment import (
    simulate_single_neuron,
    train_single_neuron_analytical,
    calibrate_tau_cutoff,
    evaluate_first_spike_decoding,
    crossing_barrier_prob
)
from population_neuron_experiment import (
    PopulationEncoder,
    PopulationNet,
    evaluate_population_ttfs,
    train_population_analytical,
    decode_first_spike_ttfs
)

PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

def generate_dataset(n_samples, lambda_min=0.0, lambda_max=1.0, Lambda=0.55, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    lambdas = torch.empty(n_samples, dtype=torch.float64).uniform_(lambda_min, lambda_max)
    labels = (lambdas >= Lambda).float()
    return lambdas, labels

def run_experiment_suite():
    results = {}
    print("=" * 70)
    print("STARTING COMPREHENSIVE EXPERIMENT SUITE: SNN BINARY CLASSIFICATION")
    print("=" * 70)

    # Base configuration
    LAMBDA_THRESH = 0.55
    STEPS = 200
    BETA_EXP = 0.01
    MEM_ZERO = 0.85
    THRESHOLD = 1.0
    N_TRAIN = 50
    N_TEST = 200
    SEED = 42

    train_lambdas, train_labels = generate_dataset(N_TRAIN, Lambda=LAMBDA_THRESH, seed=SEED)
    test_lambdas, test_labels = generate_dataset(N_TEST, Lambda=LAMBDA_THRESH, seed=SEED + 999)

    # -------------------------------------------------------------------------
    # PART 1: SINGLE NEURON EXPERIMENTS
    # -------------------------------------------------------------------------
    print("\n>>> Part 1: Training Single ExpNeuron...")
    t0 = time.time()
    w_single, losses_single, w_hist_single = train_single_neuron_analytical(
        train_lambdas, train_labels,
        epochs=20, lr=1e-3,
        beta=BETA_EXP, threshold=THRESHOLD, membrane_zero=MEM_ZERO, steps=STEPS
    )
    t_train_single = time.time() - t0
    w_opt_single = w_single.item()
    print(f"    Trained weight w*: {w_opt_single:.4f} (took {t_train_single:.2f}s)")

    # Calibrate optimal latency cutoff tau* on train set
    tau_star, train_acc_cal = calibrate_tau_cutoff(
        w_single, train_lambdas, train_labels,
        steps=STEPS, beta=BETA_EXP, threshold=THRESHOLD, membrane_zero=MEM_ZERO, seed=SEED
    )
    print(f"    Calibrated TTFS latency threshold tau*: {tau_star} steps (Train Acc: {train_acc_cal*100:.1f}%)")

    # Evaluate with calibrated TTFS cutoff
    acc_single_cal, first_spikes, conf_single = evaluate_first_spike_decoding(
        w_single, test_lambdas, test_labels,
        steps=STEPS, beta=BETA_EXP, threshold=THRESHOLD, membrane_zero=MEM_ZERO,
        tau_cutoff=tau_star, seed=SEED+1
    )
    # Also evaluate raw presence (tau <= T)
    acc_single_raw, _, _ = evaluate_first_spike_decoding(
        w_single, test_lambdas, test_labels,
        steps=STEPS, beta=BETA_EXP, threshold=THRESHOLD, membrane_zero=MEM_ZERO,
        tau_cutoff=STEPS, seed=SEED+1
    )
    print(f"    Test Accuracy (Calibrated TTFS tau<={tau_star}): {acc_single_cal * 100:.2f}%")
    print(f"    Test Accuracy (Spike presence within T={STEPS}): {acc_single_raw * 100:.2f}%")
    print(f"    Confusion Matrix: TP={conf_single['tp']}, FP={conf_single['fp']}, TN={conf_single['tn']}, FN={conf_single['fn']}")

    results['single_neuron_baseline'] = {
        'w_opt': w_opt_single,
        'tau_star': tau_star,
        'accuracy_calibrated_ttfs': acc_single_cal,
        'accuracy_raw_presence': acc_single_raw,
        'confusion': conf_single,
        'train_time': t_train_single
    }

    # 1.2: Latency vs Intensity curve tau(lambda)
    print("\n>>> Part 1.2: Measuring First-Spike Latency tau(lambda)...")
    eval_lambdas = torch.linspace(0.05, 0.95, 30, dtype=torch.float64)
    n_mc_runs = 15
    tau_matrix = np.zeros((len(eval_lambdas), n_mc_runs))
    prob_analytical = []

    for i, lam in enumerate(eval_lambdas):
        p = crossing_barrier_prob(
            MEM_ZERO, THRESHOLD, STEPS, BETA_EXP,
            lambdas=lam.unsqueeze(0), ws=w_single
        ).item()
        prob_analytical.append(p)
        for r in range(n_mc_runs):
            _, fs, _ = simulate_single_neuron(
                w_single, lam.unsqueeze(0), STEPS, BETA_EXP, THRESHOLD, MEM_ZERO, seed=SEED + 100*r + i
            )
            tau_matrix[i, r] = fs.item()

    tau_mean = np.mean(tau_matrix, axis=1)
    tau_std = np.std(tau_matrix, axis=1)

    # Plot 1: Learning curve + First-Spike Latency & Probability Curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(losses_single, 'o-', color='#1E40AF', lw=2, label='MSE Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('MSE Loss', fontsize=12)
    ax1.set_title(f'Analytical Training Dynamics (w* = {w_opt_single:.3f})', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2_twin = ax2.twinx()
    l1 = ax2.plot(eval_lambdas.numpy(), tau_mean, 's-', color='#DC2626', lw=2, label='First-Spike Latency tau')
    ax2.fill_between(eval_lambdas.numpy(), tau_mean - tau_std, tau_mean + tau_std, color='#DC2626', alpha=0.15)
    ax2.axhline(y=tau_star, color='#9333EA', linestyle='-.', lw=1.5, label=f'Decision Cutoff tau*={tau_star}')
    l2 = ax2_twin.plot(eval_lambdas.numpy(), prob_analytical, '^--', color='#059669', lw=2, label='Analytical P(tau <= T)')
    ax2.axvline(x=LAMBDA_THRESH, color='#4B5563', linestyle=':', lw=2, label=f'Barrier Lambda = {LAMBDA_THRESH}')

    ax2.set_xlabel('Input Poisson Intensity lambda', fontsize=12)
    ax2.set_ylabel('Time-To-First-Spike tau (steps)', color='#DC2626', fontsize=12)
    ax2_twin.set_ylabel('Probability P(tau <= T)', color='#059669', fontsize=12)
    ax2.set_title('TTFS Latency & Spiking Probability vs Intensity', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    lines = l1 + [ax2.get_lines()[-2]] + l2 + [ax2.get_lines()[-1]]
    labels_leg = [l.get_label() for l in lines]
    ax2.legend(lines, labels_leg, loc='center right', fontsize=9)

    plt.subplots_adjust(wspace=0.3)
    plot1_path = os.path.join(PLOTS_DIR, 'plot1_single_learning_and_latency.png')
    plt.savefig(plot1_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved Plot 1: {plot1_path}")

    # 1.3: Hyperparameter Sweep: Beta (leak rate)
    print("\n>>> Part 1.3: Sweeping Membrane Decay Beta...")
    beta_list = [0.002, 0.005, 0.01, 0.02, 0.05, 0.08, 0.12]
    beta_results = []

    for b in beta_list:
        w_b, _, _ = train_single_neuron_analytical(
            train_lambdas, train_labels, epochs=15, lr=1e-3,
            beta=b, threshold=THRESHOLD, membrane_zero=MEM_ZERO, steps=STEPS
        )
        t_star_b, _ = calibrate_tau_cutoff(
            w_b, train_lambdas, train_labels,
            steps=STEPS, beta=b, threshold=THRESHOLD, membrane_zero=MEM_ZERO, seed=SEED
        )
        acc_b, _, conf_b = evaluate_first_spike_decoding(
            w_b, test_lambdas, test_labels,
            steps=STEPS, beta=b, threshold=THRESHOLD, membrane_zero=MEM_ZERO,
            tau_cutoff=t_star_b, seed=SEED
        )
        fpr = conf_b['fp'] / max(1, (conf_b['fp'] + conf_b['tn']))
        fnr = conf_b['fn'] / max(1, (conf_b['fn'] + conf_b['tp']))
        beta_results.append({
            'beta': b,
            'w_opt': w_b.item(),
            'tau_star': t_star_b,
            'accuracy': acc_b,
            'fpr': fpr,
            'fnr': fnr
        })
        print(f"    Beta={b:<6}: w*={w_b.item():.4f}, tau*={t_star_b:<3}, Acc={acc_b*100:.1f}%, FPR={fpr*100:.1f}%, FNR={fnr*100:.1f}%")

    results['beta_sweep'] = beta_results

    # Plot 2: Beta Sweep
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    betas = [r['beta'] for r in beta_results]
    accs = [r['accuracy'] * 100 for r in beta_results]
    w_opts = [r['w_opt'] for r in beta_results]
    fprs = [r['fpr'] * 100 for r in beta_results]
    fnrs = [r['fnr'] * 100 for r in beta_results]

    ax1.plot(betas, accs, 'o-', color='#1D4ED8', lw=2.5, label='Test Accuracy (%)')
    ax1.set_xscale('log')
    ax1.set_xlabel('Membrane Decay Rate beta (log scale)', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', color='#1D4ED8', fontsize=12)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(betas, w_opts, 's--', color='#D97706', lw=2, label='Trained Weight w*')
    ax1_twin.set_ylabel('Optimal Synaptic Weight w*', color='#D97706', fontsize=12)
    ax1.set_title('Accuracy & Weight Adaptation vs Decay beta', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    ax2.plot(betas, fprs, 'o-', color='#DC2626', lw=2, label='False Positive Rate (FPR)')
    ax2.plot(betas, fnrs, 's-', color='#9333EA', lw=2, label='False Negative Rate (FNR)')
    ax2.set_xscale('log')
    ax2.set_xlabel('Membrane Decay Rate beta (log scale)', fontsize=12)
    ax2.set_ylabel('Error Rate (%)', fontsize=12)
    ax2.set_title('FPR (Over-integration) vs FNR (Leak loss)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.subplots_adjust(wspace=0.3)
    plot2_path = os.path.join(PLOTS_DIR, 'plot2_single_beta_sweep.png')
    plt.savefig(plot2_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved Plot 2: {plot2_path}")

    # 1.4: Hyperparameter Sweep: Time Horizon T
    print("\n>>> Part 1.4: Sweeping Observation Horizon T (steps)...")
    t_list = [20, 40, 75, 120, 200, 350, 500]
    t_results = []

    for t_step in t_list:
        w_t, _, _ = train_single_neuron_analytical(
            train_lambdas, train_labels, epochs=15, lr=1e-3,
            beta=BETA_EXP, threshold=THRESHOLD, membrane_zero=MEM_ZERO, steps=t_step
        )
        t_star_t, _ = calibrate_tau_cutoff(
            w_t, train_lambdas, train_labels,
            steps=t_step, beta=BETA_EXP, threshold=THRESHOLD, membrane_zero=MEM_ZERO, seed=SEED
        )
        acc_t, _, _ = evaluate_first_spike_decoding(
            w_t, test_lambdas, test_labels,
            steps=t_step, beta=BETA_EXP, threshold=THRESHOLD, membrane_zero=MEM_ZERO,
            tau_cutoff=t_star_t, seed=SEED
        )
        t_results.append({
            'T': t_step,
            'w_opt': w_t.item(),
            'tau_star': t_star_t,
            'accuracy': acc_t
        })
        print(f"    T={t_step:<4}: w*={w_t.item():.4f}, tau*={t_star_t:<3}, Acc={acc_t*100:.1f}%")

    results['horizon_sweep'] = t_results

    # Plot 3: T Horizon Sweep & Theoretical Poisson Variance
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ts = [r['T'] for r in t_results]
    t_accs = [r['accuracy'] * 100 for r in t_results]

    ax1.plot(ts, t_accs, 'o-', color='#059669', lw=2.5, label='Empirical Accuracy (%)')
    ax1.set_xlabel('Time Horizon T (discrete steps)', fontsize=12)
    ax1.set_ylabel('Classification Accuracy (%)', fontsize=12)
    ax1.set_title('Classification Accuracy vs Time Horizon T', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    ax1_twin = ax1.twinx()
    poisson_stds = [np.sqrt(LAMBDA_THRESH / t) for t in ts]
    ax1_twin.plot(ts, poisson_stds, '^--', color='#6B7280', lw=2, label='Poisson Jitter sqrt(Lambda / T)')
    ax1_twin.set_ylabel('Poisson Sampling Uncertainty sqrt(Lambda / T)', color='#6B7280', fontsize=12)

    lines = ax1.get_legend_handles_labels()[0] + ax1_twin.get_legend_handles_labels()[0]
    labels_leg = [l.get_label() for l in lines]
    ax1.legend(lines, labels_leg, loc='center right')

    plot3_path = os.path.join(PLOTS_DIR, 'plot3_single_horizon_T_sweep.png')
    plt.savefig(plot3_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved Plot 3: {plot3_path}")

    # 1.5 & 1.6: Sweeping Barrier Height x = ln(U_th / U_0) and Boundary Lambda
    print("\n>>> Part 1.5 & 1.6: Sweeping Barrier x and Boundary Lambda...")
    u0_list = [0.60, 0.75, 0.85, 0.95]
    u0_results = []
    for u0 in u0_list:
        w_u, _, _ = train_single_neuron_analytical(
            train_lambdas, train_labels, epochs=15, lr=1e-3,
            beta=BETA_EXP, threshold=THRESHOLD, membrane_zero=u0, steps=STEPS
        )
        t_star_u, _ = calibrate_tau_cutoff(
            w_u, train_lambdas, train_labels,
            steps=STEPS, beta=BETA_EXP, threshold=THRESHOLD, membrane_zero=u0, seed=SEED
        )
        acc_u, _, _ = evaluate_first_spike_decoding(
            w_u, test_lambdas, test_labels,
            steps=STEPS, beta=BETA_EXP, threshold=THRESHOLD, membrane_zero=u0,
            tau_cutoff=t_star_u, seed=SEED
        )
        x_val = np.log(THRESHOLD / u0)
        u0_results.append({
            'U0': u0,
            'x_barrier': x_val,
            'w_opt': w_u.item(),
            'tau_star': t_star_u,
            'accuracy': acc_u
        })
        print(f"    U0={u0:.2f} (x={x_val:.3f}): w*={w_u.item():.4f}, tau*={t_star_u:<3}, Acc={acc_u*100:.1f}%")

    results['barrier_sweep'] = u0_results

    lambda_targets = [0.35, 0.55, 0.75]
    lambda_results = []
    for lt in lambda_targets:
        tr_lam, tr_lbl = generate_dataset(N_TRAIN, Lambda=lt, seed=SEED)
        te_lam, te_lbl = generate_dataset(N_TEST, Lambda=lt, seed=SEED + 500)
        w_l, _, _ = train_single_neuron_analytical(
            tr_lam, tr_lbl, epochs=15, lr=1e-3,
            beta=BETA_EXP, threshold=THRESHOLD, membrane_zero=MEM_ZERO, steps=STEPS
        )
        t_star_l, _ = calibrate_tau_cutoff(
            w_l, tr_lam, tr_lbl,
            steps=STEPS, beta=BETA_EXP, threshold=THRESHOLD, membrane_zero=MEM_ZERO, seed=SEED
        )
        acc_l, _, conf_l = evaluate_first_spike_decoding(
            w_l, te_lam, te_lbl,
            steps=STEPS, beta=BETA_EXP, threshold=THRESHOLD, membrane_zero=MEM_ZERO,
            tau_cutoff=t_star_l, seed=SEED
        )
        lambda_results.append({
            'Lambda': lt,
            'w_opt': w_l.item(),
            'tau_star': t_star_l,
            'accuracy': acc_l,
            'confusion': conf_l
        })
        print(f"    Lambda={lt:.2f}: w*={w_l.item():.4f}, tau*={t_star_l:<3}, Acc={acc_l*100:.1f}%")

    results['lambda_sweep'] = lambda_results

    # Plot 4: Barrier x & Lambda Sweeps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x_barriers = [r['x_barrier'] for r in u0_results]
    x_w_opts = [r['w_opt'] for r in u0_results]
    x_accs = [r['accuracy'] * 100 for r in u0_results]

    ax1.plot(x_barriers, x_w_opts, 'o-', color='#7C3AED', lw=2.5, label='Required w*')
    ax1.set_xlabel('Log-Barrier Height x = ln(U_th / U_0)', fontsize=12)
    ax1.set_ylabel('Optimal Synaptic Weight w*', color='#7C3AED', fontsize=12)
    ax1_twin = ax1.twinx()
    ax1_twin.plot(x_barriers, x_accs, 's--', color='#059669', lw=2, label='Accuracy (%)')
    ax1_twin.set_ylabel('Accuracy (%)', color='#059669', fontsize=12)
    ax1.set_title('Effect of Barrier Height on Weight and Accuracy', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    l_vals = [r['Lambda'] for r in lambda_results]
    l_w_opts = [r['w_opt'] for r in lambda_results]
    l_accs = [r['accuracy'] * 100 for r in lambda_results]

    ax2.bar([str(lv) for lv in l_vals], l_accs, width=0.4, color='#3B82F6', alpha=0.8, label='Accuracy (%)')
    ax2.set_xlabel('Classification Barrier Lambda', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', color='#3B82F6', fontsize=12)
    ax2.set_ylim([70, 105])
    ax2_twin = ax2.twinx()
    ax2_twin.plot([str(lv) for lv in l_vals], l_w_opts, 'D-', color='#EF4444', lw=2.5, label='Weight w*')
    ax2_twin.set_ylabel('Optimal Weight w*', color='#EF4444', fontsize=12)
    ax2.set_title('Classification Across Different Lambda Values', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.subplots_adjust(wspace=0.3)
    plot4_path = os.path.join(PLOTS_DIR, 'plot4_single_barrier_and_lambda_sweep.png')
    plt.savefig(plot4_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved Plot 4: {plot4_path}")

    # -------------------------------------------------------------------------
    # PART 2: POPULATION 2-NEURON TTFS RACE EXPERIMENTS
    # -------------------------------------------------------------------------
    print("\n>>> Part 2: Training 2-Neuron Population Network (TTFS Race)...")
    DELTA = 0.21
    C_0 = LAMBDA_THRESH - DELTA # 0.34
    C_1 = LAMBDA_THRESH + DELTA # 0.76
    SIGMA = 0.17
    encoder = PopulationEncoder(n_neurons=2, mu=[C_0, C_1], sigma=SIGMA)

    pop_weights = nn.Parameter(torch.ones(2, dtype=torch.float64) * 0.1)
    pop_weights, pop_losses, pop_w_hist = train_population_analytical(
        encoder, pop_weights, train_lambdas, train_labels,
        epochs=15, lr=5e-4, beta=BETA_EXP, threshold=THRESHOLD, membrane_zero=MEM_ZERO, steps=STEPS
    )
    print(f"    Trained Population Weights: w0={pop_weights[0].item():.4f}, w1={pop_weights[1].item():.4f}")

    # Generate test spike trains for 2-neuron population
    test_trains = torch.stack([
        torch.poisson(encoder(l).squeeze(0).unsqueeze(0).expand(STEPS, -1))
        for l in test_lambdas
    ]) # (N_test, steps, 2)

    net_pop = PopulationNet(
        weights=pop_weights, steps=STEPS, beta=BETA_EXP,
        threshold=THRESHOLD, membrane_zero=MEM_ZERO
    )
    pop_acc, pop_none = evaluate_population_ttfs(net_pop, test_trains, test_labels)
    print(f"    Population TTFS Accuracy: {pop_acc * 100:.2f}% (Silent samples: {pop_none * 100:.2f}%)")

    results['population_baseline'] = {
        'w0': pop_weights[0].item(),
        'w1': pop_weights[1].item(),
        'accuracy': pop_acc,
        'none_rate': pop_none
    }

    # 2.2: Competitive TTFS Latency Curves tau_0(lambda) and tau_1(lambda)
    print("\n>>> Part 2.2: Measuring Competitive TTFS Race Dynamics...")
    tau0_list, tau1_list = [], []
    for lam in eval_lambdas:
        lam_enc = encoder(lam).squeeze(0)
        # Poisson spike train
        spk_in = torch.poisson(lam_enc.unsqueeze(0).expand(STEPS, -1)).unsqueeze(0)
        spk_out, _ = net_pop(spk_in)
        _, (t0, t1) = decode_first_spike_ttfs(spk_out.squeeze(0))
        tau0_list.append(t0)
        tau1_list.append(t1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(eval_lambdas.numpy(), tau0_list, 'o-', color='#2563EB', lw=2.5, label='Neuron 0 Latency tau_0 (C_0=0.34)')
    ax.plot(eval_lambdas.numpy(), tau1_list, 's-', color='#DC2626', lw=2.5, label='Neuron 1 Latency tau_1 (C_1=0.76)')
    ax.axvline(x=LAMBDA_THRESH, color='#111827', linestyle='--', lw=2, label=f'Decision Barrier Lambda = {LAMBDA_THRESH}')

    # Highlight winning regions
    ax.axvspan(0.05, LAMBDA_THRESH, alpha=0.1, color='#2563EB', label='Region y_pred=0 (tau_0 < tau_1)')
    ax.axvspan(LAMBDA_THRESH, 0.95, alpha=0.1, color='#DC2626', label='Region y_pred=1 (tau_1 < tau_0)')

    ax.set_xlabel('Input Poisson Intensity lambda', fontsize=12)
    ax.set_ylabel('Time-To-First-Spike (steps)', fontsize=12)
    ax.set_title('Competitive TTFS Latency Race: Crossover at Decision Boundary', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', framealpha=0.9)

    plot5_path = os.path.join(PLOTS_DIR, 'plot5_population_ttfs_dynamics.png')
    plt.savefig(plot5_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved Plot 5: {plot5_path}")

    # 2.3: Hyperparameter Sweep: Sigma (Tuning curve width)
    print("\n>>> Part 2.3: Sweeping Tuning Curve Width Sigma...")
    sigma_list = [0.08, 0.12, 0.17, 0.25, 0.35]
    sigma_results = []

    for s in sigma_list:
        enc_s = PopulationEncoder(n_neurons=2, mu=[C_0, C_1], sigma=s)
        w_s = nn.Parameter(torch.ones(2, dtype=torch.float64) * 0.1)
        w_s, _, _ = train_population_analytical(
            enc_s, w_s, train_lambdas, train_labels,
            epochs=15, lr=5e-4, beta=BETA_EXP, threshold=THRESHOLD, membrane_zero=MEM_ZERO, steps=STEPS
        )
        te_trains_s = torch.stack([
            torch.poisson(enc_s(l).squeeze(0).unsqueeze(0).expand(STEPS, -1))
            for l in test_lambdas
        ])
        net_s = PopulationNet(weights=w_s, steps=STEPS, beta=BETA_EXP, threshold=THRESHOLD, membrane_zero=MEM_ZERO)
        acc_s, none_s = evaluate_population_ttfs(net_s, te_trains_s, test_labels)
        sigma_results.append({
            'sigma': s,
            'w0': w_s[0].item(),
            'w1': w_s[1].item(),
            'accuracy': acc_s,
            'none_rate': none_s
        })
        print(f"    Sigma={s:<5}: w=({w_s[0].item():.3f}, {w_s[1].item():.3f}), Acc={acc_s*100:.1f}%, Silent={none_s*100:.1f}%")

    results['sigma_sweep'] = sigma_results

    # 2.4: Comparison: ExpNeuron vs LIF
    print("\n>>> Part 2.4: Multi-run Robustness: ExpNeuron vs LIF...")
    import snntorch as snn
    class LIFNet(nn.Module):
        def __init__(self, weights, steps, beta=0.95, threshold=1.0):
            super().__init__()
            self.weights = weights
            self.steps = steps
            self.neuron = snn.Leaky(beta=beta, threshold=threshold, learn_beta=False, learn_threshold=False)
        def forward(self, spk_input):
            mem = self.neuron.init_leaky()
            spk_out = []
            for t in range(self.steps):
                inp = spk_input[:, t, :].float() * self.weights.float()
                spk, mem = self.neuron(inp, mem)
                spk_out.append(spk)
            return torch.stack(spk_out, dim=1)

    # Train LIF weights via surrogate gradient presence loss
    lif_weights = nn.Parameter(torch.ones(2, dtype=torch.float32) * 0.5)
    opt_lif = torch.optim.Adam([lif_weights], lr=5e-3)
    target_c0 = torch.tensor([1.0, 0.0], dtype=torch.float32)
    target_c1 = torch.tensor([0.0, 1.0], dtype=torch.float32)

    # Train LIF on Poisson trains
    train_trains = torch.stack([
        torch.poisson(encoder(l).squeeze(0).unsqueeze(0).expand(STEPS, -1))
        for l in train_lambdas
    ])
    net_lif = LIFNet(weights=lif_weights, steps=STEPS, beta=0.95, threshold=THRESHOLD)

    for epoch in range(15):
        for i in range(len(train_lambdas)):
            opt_lif.zero_grad()
            spk_out = net_lif(train_trains[i:i+1])
            counts = spk_out.sum(dim=1)
            presence = 1.0 - torch.exp(-1.0 * counts).squeeze(0)
            tgt = target_c1 if train_labels[i].item() == 1.0 else target_c0
            loss = ((presence - tgt) ** 2).mean()
            loss.backward()
            opt_lif.step()
            with torch.no_grad():
                lif_weights.clamp_(min=0.01, max=5.0)

    # Multi-run evaluation
    n_runs = 10
    exp_accs, lif_accs = [], []
    for r in range(n_runs):
        torch.manual_seed(1000 + r)
        run_trains = torch.stack([
            torch.poisson(encoder(l).squeeze(0).unsqueeze(0).expand(STEPS, -1))
            for l in test_lambdas
        ])
        acc_e, _ = evaluate_population_ttfs(net_pop, run_trains, test_labels)
        # LIF TTFS evaluation
        with torch.no_grad():
            spk_l = net_lif(run_trains)
        corr_l = 0
        for idx in range(len(test_labels)):
            pred_l, _ = decode_first_spike_ttfs(spk_l[idx])
            if pred_l == int(test_labels[idx].item()):
                corr_l += 1
        acc_l = corr_l / len(test_labels)
        exp_accs.append(acc_e)
        lif_accs.append(acc_l)

    exp_mean, exp_std = np.mean(exp_accs), np.std(exp_accs)
    lif_mean, lif_std = np.mean(lif_accs), np.std(lif_accs)
    print(f"    Multi-Run (10 runs x {N_TEST} samples):")
    print(f"    ExpNeuron (proposed): {exp_mean*100:.2f}% +- {exp_std*100:.2f}%")
    print(f"    LIF (baseline):       {lif_mean*100:.2f}% +- {lif_std*100:.2f}%")

    results['comparison'] = {
        'exp_mean': exp_mean, 'exp_std': exp_std,
        'lif_mean': lif_mean, 'lif_std': lif_std
    }

    # Plot 6: Sigma sweep & Model Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    sigmas = [r['sigma'] for r in sigma_results]
    s_accs = [r['accuracy'] * 100 for r in sigma_results]
    s_nones = [r['none_rate'] * 100 for r in sigma_results]

    ax1.plot(sigmas, s_accs, 'o-', color='#059669', lw=2.5, label='TTFS Accuracy (%)')
    ax1.plot(sigmas, s_nones, 's--', color='#EF4444', lw=2, label='Silent Samples (%)')
    ax1.set_xlabel('Gaussian Tuning Width sigma', fontsize=12)
    ax1.set_ylabel('Percentage (%)', fontsize=12)
    ax1.set_title('Tuning Width sigma: Accuracy vs Silent Samples', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    models = ['ExpNeuron (Proposed)', 'LIF (Baseline)']
    means = [exp_mean * 100, lif_mean * 100]
    stds = [exp_std * 100, lif_std * 100]
    bars = ax2.bar(models, means, yerr=stds, capsize=8, color=['#2563EB', '#6B7280'], alpha=0.85, width=0.5)
    ax2.set_ylabel('Multi-Run Accuracy (%)', fontsize=12)
    ax2.set_ylim([70, 105])
    ax2.set_title('Robustness Comparison (10 Independent Runs)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, m, s in zip(bars, means, stds):
        ax2.text(bar.get_x() + bar.get_width()/2, m - 5, f"{m:.1f}%\n+- {s:.1f}%",
                 ha='center', va='center', color='white', fontweight='bold', fontsize=11)

    plt.subplots_adjust(wspace=0.3)
    plot6_path = os.path.join(PLOTS_DIR, 'plot6_population_sigma_sweep_and_lif_comparison.png')
    plt.savefig(plot6_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Saved Plot 6: {plot6_path}")

    # Save all raw numerical data
    results_json_path = os.path.join(os.path.dirname(__file__), 'experiment_results.json')
    with open(results_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to {results_json_path}")
    print("=" * 70)

if __name__ == '__main__':
    run_experiment_suite()

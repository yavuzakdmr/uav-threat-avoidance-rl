"""
Aynı senaryolarda (aynı seed listesi) kural tabanlı yöntem ile PPO politikalarını karşılaştırır.
Çıktılar: results/episodes.csv, results/summary.csv, results/disagreements.png
"""
import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from rule_based import PotentialFieldPolicy
from uav_env import ThreatAvoidanceEnv

# isim: (tür, model yolu, değerlendirmede zarf uygulansın mı?)
CONFIGS = {
    "kural_tabanli":        ("rule", None, True),
    "ppo_zarfli":           ("ppo", "models/ppo_shield_s0", True),
    "ppo_zarfsiz":          ("ppo", "models/ppo_none_s0", False),
    "ppo_zarfsiz_kalkanli": ("ppo", "models/ppo_none_s0", True),  # zarfsız eğitilip zarfla uçarsa?
}


def make_policy(kind, path):
    if kind == "rule":
        pf = PotentialFieldPolicy()
        return lambda obs, env: pf(env)
    model = PPO.load(path)
    return lambda obs, env: model.predict(obs, deterministic=True)[0]


def run_episode(env, policy, seed):
    obs, _ = env.reset(seed=seed)
    traj, threats = [env.pos.copy()], [env.threat_pos.copy()]
    done = False
    while not done:
        obs, _, terminated, truncated, info = env.step(policy(obs, env))
        traj.append(env.pos.copy())
        threats.append(env.threat_pos.copy())
        done = terminated or truncated
    ep = {"traj": np.array(traj), "threats": np.array(threats),
          "radii": env.threat_r.copy(), "goal": env.goal.copy()}
    return info, ep


def wilson_ci(k, n, z=1.96):
    """Bir oran için Wilson %95 güven aralığı."""
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half


def mcnemar_exact(a_ok, b_ok):
    """Eşleştirilmiş karşılaştırma: aynı senaryoda yalnız birinin başardığı durumlara bakar."""
    b = int(np.sum(a_ok & ~b_ok))  # yalnız A başarılı
    c = int(np.sum(~a_ok & b_ok))  # yalnız B başarılı
    n = b + c
    if n == 0:
        return b, c, 1.0
    p = 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return b, c, min(p, 1.0)


def summarize(g):
    n = len(g)
    s = int((g.outcome == "success").sum())
    c = int((g.outcome == "collision").sum())
    return pd.Series({
        "basari": s / n,
        "basari_ci95": "[{:.3f}, {:.3f}]".format(*wilson_ci(s, n)),
        "carpisma": c / n,
        "carpisma_ci95": "[{:.3f}, {:.3f}]".format(*wilson_ci(c, n)),
        "sinir_disi": (g.outcome == "out_of_bounds").mean(),
        "zaman_asimi": (g.outcome == "timeout").mean(),
        "zarf_ihlali_ep": g.envelope_violations.mean(),
        "kalkan_mudahale_adim": g.shield_interventions.sum() / g.steps.sum(),
        "min_bosluk_medyan": g.min_clearance.median(),
    })


def plot_disagreements(df, eps, a="ppo_zarfli", b="kural_tabanli", n=3,
                       out="results/disagreements.png"):
    oa = df[df.method == a].set_index("seed").outcome
    ob = df[df.method == b].set_index("seed").outcome
    cases = [(s, a) for s in oa.index if oa[s] == "success" and ob[s] != "success"][:n]
    cases += [(s, b) for s in oa.index if oa[s] != "success" and ob[s] == "success"][:n]
    if not cases:
        return
    fig, axes = plt.subplots(1, len(cases), figsize=(4 * len(cases), 4.4), squeeze=False)
    for ax, (s, winner) in zip(axes[0], cases):
        ea, eb = eps[(a, s)], eps[(b, s)]
        failed = eb if winner == a else ea
        for i, r in enumerate(failed["radii"]):
            tp = failed["threats"][:, i]
            ax.plot(tp[:, 0], tp[:, 1], color="tab:red", alpha=0.25, lw=1)
            ax.add_patch(plt.Circle(tp[-1], r, color="tab:red", alpha=0.3))
        ax.plot(ea["traj"][:, 0], ea["traj"][:, 1], label=f"{a}: {oa[s]}")
        ax.plot(eb["traj"][:, 0], eb["traj"][:, 1], label=f"{b}: {ob[s]}")
        ax.plot(*ea["goal"], marker="*", markersize=15, color="gold")
        ax.set(xlim=(0, 100), ylim=(0, 100), aspect="equal")
        ax.set_title(f"seed {s}: yalnız {winner} başarılı", fontsize=9)
        ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=500)
    args = p.parse_args()
    seeds = range(10_000, 10_000 + args.episodes)  # eğitimde kullanılmayan seed'ler

    rows, eps = [], {}
    for name, (kind, path, shield) in CONFIGS.items():
        if kind == "ppo" and not os.path.exists(path + ".zip"):
            print(f"[atlandı] {name}: {path}.zip yok, önce train.py çalıştır")
            continue
        env = ThreatAvoidanceEnv(enforce_envelope=shield)
        policy = make_policy(kind, path)
        for s in seeds:
            info, ep = run_episode(env, policy, s)
            rows.append({"method": name, "seed": s, **info})
            eps[(name, s)] = ep
        print(f"{name}: tamamlandı")

    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv("results/episodes.csv", index=False)

    summary = pd.DataFrame({m: summarize(g) for m, g in df.groupby("method", sort=False)}).T
    summary = summary.infer_objects()
    summary.to_csv("results/summary.csv")
    print("\n" + summary.round(3).to_string())

    if "kural_tabanli" in df.method.values:
        base = df[df.method == "kural_tabanli"].set_index("seed").outcome.eq("success")
        print("\nEşleştirilmiş karşılaştırma (kesin McNemar, kural tabanlıya karşı):")
        for m in df.method.unique():
            if m == "kural_tabanli":
                continue
            other = df[df.method == m].set_index("seed").outcome.eq("success").reindex(base.index)
            b, c, pval = mcnemar_exact(other.to_numpy(), base.to_numpy())
            print(f"  {m}: yalnız {m} başarılı={b}, yalnız kural başarılı={c}, p={pval:.4f}")
        if "ppo_zarfli" in df.method.values:
            plot_disagreements(df, eps)


if __name__ == "__main__":
    main()

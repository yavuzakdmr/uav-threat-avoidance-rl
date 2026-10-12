import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from uav_env import ThreatAvoidanceEnv

p = argparse.ArgumentParser()
p.add_argument("--envelope", choices=["shield", "none"], default="shield")
p.add_argument("--steps", type=int, default=500_000)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

env = make_vec_env(ThreatAvoidanceEnv, n_envs=8, seed=args.seed,
                   env_kwargs={"enforce_envelope": args.envelope == "shield"})
model = PPO("MlpPolicy", env, n_steps=1024, batch_size=256, gamma=0.99,
            seed=args.seed, verbose=1, tensorboard_log="runs")
model.learn(total_timesteps=args.steps, tb_log_name=f"ppo_{args.envelope}_s{args.seed}")

os.makedirs("models", exist_ok=True)
model.save(f"models/ppo_{args.envelope}_s{args.seed}")

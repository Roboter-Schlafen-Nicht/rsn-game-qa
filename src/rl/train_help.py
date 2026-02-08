# train_help_rl.py
"""
Train an RL policy on the simulated LastWarHelpEnv using PPO.

This is v3-alpha: policy learns when to CLICK_HELP vs NOOP vs RANDOM_SWIPE
on top of a tiny feature vector. Later, you can swap the env internals
to call the real game while keeping this script.
"""

from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from rl.env_help import LastWarHelpEnv


def make_env():
    return LastWarHelpEnv(
        max_episode_steps=200,
        help_appear_prob=0.05,
        help_disappear_prob=0.02,
        swipe_find_help_prob=0.4,
        since_last_clip=30.0,
    )


def main():
    # 1) Create and sanity-check env
    env = make_env()
    check_env(env, warn=True)

    # 2) Wrap for SB3 (Gymnasium is supported in SB3>=2)
    vec_env = gym.wrappers.TimeLimit(env, max_episode_steps=200)

    # 3) Define PPO model (small MLP)
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        learning_rate=3e-4,
        device="cpu",
    )

    # 4) Train
    total_timesteps = 200_000
    model.learn(total_timesteps=total_timesteps)

    # 5) Save
    out_dir = Path("runs/rl_help_v3")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "ppo_lastwar_help.zip"
    model.save(str(model_path))
    print(f"[RL HELP] Saved PPO policy to {model_path}")

    # 6) Quick qualitative test
    test_env = make_env()
    obs, _ = test_env.reset()
    for _ in range(50):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(int(action))
        print(f"action={action}, reward={reward:.3f}, info={info}, obs={obs}")
        if done or truncated:
            obs, _ = test_env.reset()


if __name__ == "__main__":
    main()

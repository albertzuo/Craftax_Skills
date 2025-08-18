#!/usr/bin/env python3
"""
Wandb sweep runner for random seed optimization.
This script is called by wandb sweep agents to run individual experiments.
"""

import wandb
import argparse
import numpy as np
from ppo_skills_kl import run_ppo


def main():
    # Initialize wandb run in sweep context
    wandb.init()
    
    # Get configuration from wandb sweep
    config = wandb.config
    
    # Convert wandb config to argparse Namespace to match existing run_ppo interface
    args = argparse.Namespace()
    
    # Set all parameters from wandb config
    args.env_name = config.env_name
    args.num_envs = config.num_envs
    args.total_timesteps = config.total_timesteps
    args.lr = float(config.lr)  # Fixed learning rate
    args.num_steps = config.num_steps
    args.update_epochs = config.update_epochs
    args.num_minibatches = config.num_minibatches
    args.gamma = config.gamma
    args.gae_lambda = config.gae_lambda
    args.clip_eps = config.clip_eps
    args.ent_coef = config.ent_coef
    args.vf_coef = config.vf_coef
    args.kl_coef = config.kl_coef
    args.use_forward_kl = config.use_forward_kl
    args.max_grad_norm = config.max_grad_norm
    args.activation = config.activation
    args.anneal_lr = config.anneal_lr
    args.debug = config.debug
    args.jit = config.jit
    args.use_wandb = config.use_wandb
    args.save_policy = config.save_policy
    args.num_repeats = config.num_repeats
    args.layer_size = config.layer_size
    args.use_optimistic_resets = config.use_optimistic_resets
    args.optimistic_reset_ratio = config.optimistic_reset_ratio
    args.max_num_skills = config.max_num_skills
    
    # Use seed from wandb sweep config
    args.seed = int(config.seed)
    
    # Set wandb project and entity if not already set by sweep
    if not hasattr(config, 'wandb_project') or config.wandb_project is None:
        args.wandb_project = "craftax-skills-seed-sweep"
    else:
        args.wandb_project = config.wandb_project
    
    if not hasattr(config, 'wandb_entity') or config.wandb_entity is None:
        args.wandb_entity = None
    else:
        args.wandb_entity = config.wandb_entity
    
    # Log the seed being tested
    print(f"Running experiment with seed: {args.seed}")
    
    # Run the training
    run_ppo(args)


if __name__ == "__main__":
    main()
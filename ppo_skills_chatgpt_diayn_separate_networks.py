import argparse
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from craftax.craftax_env import make_craftax_env_from_name # Added BlockType
from craftax.craftax_classic.constants import *

import wandb
from typing import NamedTuple

from flax.training import orbax_utils
from flax.training.train_state import TrainState
import orbax.checkpoint as ocp
from orbax.checkpoint import (
    PyTreeCheckpointer,
    CheckpointManagerOptions,
    CheckpointManager,
)

from logz.batch_logging import batch_log, create_log_dict
# from models.diayn_ac import DiaynAc
from models.actor_critic import (
    ActorCritic,
    ActorCriticConv,
)
from reward_fns.gemini_skill_rewards import (
    calculate_harvesting_reward,
    crafting_reward_fn,
    survival_reward_function,
)
from reward_fns.gemini_diverse_skills_rewards import (
    reward_broaden_horizons_stockpile,
    reward_execute_next_milestone_skill,
)
from reward_fns.gemini_personality_rewards import (
    cautious_reward_function,
    driven_reward_function,
    playful_reward_function,
)
from reward_fns.my_skill_rewards import (
    my_harvesting_reward_fn,
    my_crafting_reward_fn,
    my_survival_reward_fn,
    my_harvesting_reward_fn_v2,
    my_crafting_reward_fn_v2,
    my_survival_reward_fn_v2,
)
from reward_fns.my_skill_rewards_state import (
    my_harvesting_reward_fn_state,
    my_crafting_reward_fn_state,
    my_survival_reward_fn_state,
    my_harvesting_crafting_reward_fn_state,
)
from reward_fns.my_ppo_skill_rewards import (
    configurable_achievement_reward_fn,
)
from meta_policy.skill_training import (
    skill_selector,
    # skill_selector_v2,
    # skill_selector_v3,
    skill_selector_two_skills,
    skill_selector_my_two_skills,
    single_skill_selector_zero,
    single_skill_selector_one,
    single_skill_selector_two,
    terminate_harvest,
    terminate_craft,
    terminate_sustain,
)
# from meta_policy.gemini_diverse_skills_training import (
#     skill_selector,
#     skill_selector_v2,
# )
# from meta_policy.gemini_personality_training import (
#     skill_selector,
#     terminate_cautious,
#     terminate_driven,
#     terminate_playful,
# )
from wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
    BatchEnvWrapper,
    AutoResetEnvWrapper,
    HealthWrapper,
    HungerWrapper,
    ThirstWrapper,
    EnergyWrapper,
    AllVitalsWrapper,
)
import matplotlib.pyplot as plt

# Code adapted from the original implementation made by Chris Lu
# Original code located at https://github.com/luchris429/purejaxrl


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward_e: jnp.ndarray
    reward_i: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: jnp.ndarray
    skill_idx: jnp.ndarray


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    # Add EVAL_FRACTIONS to config if not present
    if "EVAL_FRACTIONS" not in config:
        config["EVAL_FRACTIONS"] = [0.15, 0.3, 0.45, 0.6, 0.75, 0.9]
    eval_steps = [int(config["NUM_UPDATES"] * frac) for frac in config["EVAL_FRACTIONS"]]
    already_evaluated = set()

    env = make_craftax_env_from_name(
        config["ENV_NAME"], not config["USE_OPTIMISTIC_RESETS"]
    )
    env_params = env.default_params

    env = EnergyWrapper(env)
    env = LogWrapper(env)
    if config["USE_OPTIMISTIC_RESETS"]:
        env = OptimisticResetVecEnvWrapper(
            env,
            num_envs=config["NUM_ENVS"],
            reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["NUM_ENVS"]),
        )
    else:
        env = AutoResetEnvWrapper(env)
        env = BatchEnvWrapper(env, num_envs=config["NUM_ENVS"])

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORKS - Create separate network for each skill
        def create_network():
            if "Symbolic" in config["ENV_NAME"]:
                return ActorCritic(env.action_space(env_params).n, config["LAYER_SIZE"])
            else:
                return ActorCriticConv(
                    env.action_space(env_params).n, config["LAYER_SIZE"]
                )
        
        # Create networks for each skill
        networks = [create_network() for _ in range(config["MAX_NUM_SKILLS"])]
        
        # Initialize network parameters for each skill
        init_x = jnp.zeros((1, env.observation_space(env_params).shape[0]))  # No skill concatenation
        network_params_list = []
        
        for i in range(config["MAX_NUM_SKILLS"]):
            rng, _rng = jax.random.split(rng)
            params = networks[i].init(_rng, init_x)
            network_params_list.append(params)
        
        # Create optimizer
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        
        # Create train states for each skill
        train_states = []
        for i in range(config["MAX_NUM_SKILLS"]):
            train_state = TrainState.create(
                apply_fn=networks[i].apply,
                params=network_params_list[i],
                tx=tx,
            )
            train_states.append(train_state)
        
        # Create network branch functions for jax.lax.switch
        def make_network_branch(network_idx):
            def branch_fn(train_states, obs):
                return networks[network_idx].apply(train_states[network_idx].params, obs)
            return branch_fn
        
        network_branches = [make_network_branch(i) for i in range(config["MAX_NUM_SKILLS"])]
        
        # Create network selector function
        def network_selector(skill_idx, train_states, obs):
            return jax.lax.switch(skill_idx, network_branches, train_states, obs)

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        obs, env_state = env.reset(_rng, env_params)

        rng, _rng = jax.random.split(rng)
        # Initialize skill indices (start with skill 0)
        skill_indices = jnp.full((config["NUM_ENVS"],), 0, dtype=jnp.int32)
        intrinsic_rewards = jnp.zeros((config["NUM_ENVS"], config["MAX_NUM_SKILLS"]))
        final_intrinsic_rewards = jnp.zeros((config["NUM_ENVS"], config["MAX_NUM_SKILLS"]))
        skill_timesteps = jnp.zeros((config["NUM_ENVS"], config["MAX_NUM_SKILLS"]))
        final_skill_timesteps = jnp.zeros((config["NUM_ENVS"], config["MAX_NUM_SKILLS"]))
        current_skill_durations = jnp.zeros(config["NUM_ENVS"])  # Track current duration of active skill
        damage_counters = jnp.zeros((config["NUM_ENVS"], 6))  # [thirst, hunger, energy, zombie, arrow, lava]
        final_damage_counters = jnp.zeros((config["NUM_ENVS"], 6))  # Episode-end damage counters

        # Helper function to detect damage sources
        def detect_damage_sources(prev_state, current_state, prev_obs, current_obs, action, done):
            """
            Detect damage sources based on state changes and observations.
            Returns array of shape (NUM_ENVS, 6) for [thirst, hunger, energy, zombie, arrow, lava]
            """
            # Extract health values
            prev_health = prev_state.player_health
            current_health = current_state.player_health
            
            # Count damage when health decreased OR when episode ended (instant death)
            health_decreased = jnp.logical_or(current_health < prev_health, done)
            
            # Extract intrinsic values from observations 
            # Observation structure: inventory + intrinsics + direction + misc + map
            inventory_size = 12  # From renderer: 12 inventory items
            intrinsics_start_idx = inventory_size
            
            # Current intrinsics (scaled 0-1, convert to 0-10)
            # Order from renderer: health, food, drink, energy
            current_food = jnp.round(current_obs[:, intrinsics_start_idx + 1] * 10.0)
            current_drink = jnp.round(current_obs[:, intrinsics_start_idx + 2] * 10.0)
            current_energy = jnp.round(current_obs[:, intrinsics_start_idx + 3] * 10.0)
            
            # Check for zero values (starvation conditions)
            thirst_damage = jnp.logical_and(health_decreased, current_drink == 0)
            hunger_damage = jnp.logical_and(health_decreased, current_food == 0)
            energy_damage = jnp.logical_and(health_decreased, current_energy == 0)
            
            # Calculate map observation parameters
            NUM_BLOCK_TYPES = len(BlockType)
            NUM_MOB_TYPES = 4
            all_map_flat_size = OBS_DIM[0] * OBS_DIM[1] * (NUM_BLOCK_TYPES + NUM_MOB_TYPES)
            
            # Extract map observations
            map_obs = current_obs[:, :all_map_flat_size]
            map_obs = map_obs.reshape(config["NUM_ENVS"], OBS_DIM[0], OBS_DIM[1], NUM_BLOCK_TYPES + NUM_MOB_TYPES)
            
            # Player is at center of observation
            center_x, center_y = OBS_DIM[0] // 2, OBS_DIM[1] // 2
            
            # Check for zombie attacks (adjacent zombies)
            zombie_idx = NUM_BLOCK_TYPES + 0  # Zombie is first mob type (index 0)
            local_area_zombies = map_obs[:, center_x-1:center_x+2, center_y-1:center_y+2, zombie_idx]
            nearby_zombies = jnp.sum(local_area_zombies, axis=(1, 2))
            zombie_damage = jnp.logical_and(health_decreased, nearby_zombies > 0)
            
            # Check for arrow attacks (skeletons damage through arrows)
            arrow_idx = NUM_BLOCK_TYPES + 3  # Arrow is fourth mob type (index 3)
            local_area_arrows = map_obs[:, center_x-1:center_x+2, center_y-1:center_y+2, arrow_idx]
            nearby_arrows = jnp.sum(local_area_arrows, axis=(1, 2))
            arrow_damage = jnp.logical_and(health_decreased, nearby_arrows > 0)
            
            # Check for lava damage (instant death from stepping on lava)
            lava_idx = BlockType.LAVA.value
            # Extract previous map observation to check where lava was
            prev_map_obs = prev_obs[:, :all_map_flat_size]
            prev_map_obs = prev_map_obs.reshape(config["NUM_ENVS"], OBS_DIM[0], OBS_DIM[1], NUM_BLOCK_TYPES + NUM_MOB_TYPES)
            
            # Check if agent was adjacent to lava in previous observation and moved toward it
            # Movement actions: 1=LEFT, 2=RIGHT, 3=UP, 4=DOWN (from Action enum)
            lava_up = prev_map_obs[:, center_x-1, center_y, lava_idx]
            lava_down = prev_map_obs[:, center_x+1, center_y, lava_idx]
            lava_left = prev_map_obs[:, center_x, center_y-1, lava_idx]
            lava_right = prev_map_obs[:, center_x, center_y+1, lava_idx]
            
            # Check if action moved toward lava and episode ended
            moved_into_lava = jnp.logical_or(
                jnp.logical_and(action == 3, lava_up > 0),      # moved up into lava
                jnp.logical_or(
                    jnp.logical_and(action == 4, lava_down > 0),    # moved down into lava
                    jnp.logical_or(
                        jnp.logical_and(action == 1, lava_left > 0),    # moved left into lava
                        jnp.logical_and(action == 2, lava_right > 0)    # moved right into lava
                    )
                )
            )
            
            lava_damage = jnp.logical_and(done, moved_into_lava)
            
            # Count all active damage sources (multiple sources can be active simultaneously)
            damage_array = jnp.zeros((config["NUM_ENVS"], 6))
            
            # Set damage counters for all active sources
            damage_array = damage_array.at[:, 0].set(thirst_damage.astype(jnp.int32))
            damage_array = damage_array.at[:, 1].set(hunger_damage.astype(jnp.int32))
            damage_array = damage_array.at[:, 2].set(energy_damage.astype(jnp.int32))
            damage_array = damage_array.at[:, 3].set(zombie_damage.astype(jnp.int32))
            damage_array = damage_array.at[:, 4].set(arrow_damage.astype(jnp.int32))
            damage_array = damage_array.at[:, 5].set(lava_damage.astype(jnp.int32))
            
            return damage_array

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                (
                    train_states,
                    env_state,
                    intrinsic_rewards,
                    final_intrinsic_rewards,
                    skill_timesteps,
                    final_skill_timesteps,
                    current_skill_durations,
                    damage_counters,
                    final_damage_counters,
                    last_obs,
                    last_skill_indices,
                    rng,
                    update_step,
                ) = runner_state
                
                # Apply network selection for all environments
                rng, _rng = jax.random.split(rng)
                network_outputs = jax.vmap(network_selector, in_axes=(0, None, 0))(
                    last_skill_indices, train_states, last_obs
                )
                pi, value = network_outputs
                
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                prev_env_state = env_state  # Save previous state for reward calculation
                base_obs, env_state, reward_e, done, info = env.step(
                    _rng, env_state, action, env_params
                )
                
                # Create dynamic termination function list
                terminate_fns = [terminate_harvest, terminate_craft, terminate_sustain][:config["MAX_NUM_SKILLS"]]
                
                # Get termination conditions for each skill
                def get_termination_single(index, last_b_obs_s, b_obs_s, duration, done_val):
                    return jax.lax.switch(index, terminate_fns, last_b_obs_s, b_obs_s, duration, done_val)
                
                # Check termination conditions
                should_terminate = jax.vmap(get_termination_single)(
                    last_skill_indices, last_obs, base_obs, current_skill_durations, done
                )
                
                # Only select new skills for environments that should terminate
                new_skill_indices = jax.vmap(skill_selector_my_two_skills)(base_obs)
                skill_indices = jnp.where(should_terminate, new_skill_indices, last_skill_indices)
                
                # Update current skill durations
                current_skill_durations = jnp.where(
                    should_terminate,
                    jnp.zeros_like(current_skill_durations),  # Reset to 0 for terminated skills
                    current_skill_durations + 1  # Increment for continuing skills
                )
                
                # No need for skill vector concatenation - use base observations directly
                obs = base_obs

                # reward_fns_single = [my_harvesting_reward_fn, my_crafting_reward_fn, my_survival_reward_fn]
                # def select_reward_single(index, prev_obs, cur_obs, done_val):
                #     return jax.lax.switch(index, reward_fns_single, prev_obs, cur_obs, done_val)
                # reward_i = jax.vmap(select_reward_single)(last_skill_indices, last_obs, base_obs, done)

                # reward_fns_single = [my_harvesting_reward_fn_v2, my_crafting_reward_fn_v2, my_survival_reward_fn_v2]
                # def select_reward_single(index, last_b_obs_s, b_obs_s, done_val, prev_state, cur_state):
                #     return jax.lax.switch(index, reward_fns_single, last_b_obs_s, b_obs_s, done_val, prev_state, cur_state)
                # reward_i = jax.vmap(select_reward_single)(last_skill_indices, last_obs, base_obs, done, prev_env_state.env_state, env_state.env_state)

                # Dynamic reward function selection based on MAX_NUM_SKILLS
                reward_fns_single = [my_harvesting_crafting_reward_fn_state, my_survival_reward_fn_state][:config["MAX_NUM_SKILLS"]]
                # reward_fns_single = [configurable_achievement_reward_fn]
                def select_reward_single(index, prev_state, cur_state, done_val):
                    return jax.lax.switch(index, reward_fns_single, prev_state, cur_state, done_val)
                reward_i = jax.vmap(select_reward_single)(last_skill_indices, prev_env_state.env_state, env_state.env_state, done)

                reward = reward_i#reward_i#reward_i + reward_e#+ reward_e
                current_env_indices = jnp.arange(config["NUM_ENVS"])
                updated_intrinsic_rewards = intrinsic_rewards.at[current_env_indices, last_skill_indices].add(reward_i)
                updated_skill_timesteps = skill_timesteps.at[current_env_indices, last_skill_indices].add(1)

                done_expanded = done[:, None]

                updated_final_intrinsic_rewards = final_intrinsic_rewards * (1 - done_expanded) + updated_intrinsic_rewards * done_expanded
                updated_intrinsic_rewards = updated_intrinsic_rewards * (1 - done_expanded)


                # updated_final_skill_timesteps = final_skill_timesteps * (1 - done_expanded) + updated_skill_timesteps * done_expanded
                # info["returned_episode_lengths"] is (NUM_ENVS,)
                # updated_skill_timesteps is (NUM_ENVS, MAX_NUM_SKILLS)
                episode_lengths = info["returned_episode_lengths"]
                # Use jnp.where to avoid division by zero for envs not done, or if length is 0 (though unlikely for done episodes)
                # The result for not-done envs will be masked out by (1 - done_expanded) anyway.
                safe_episode_lengths = jnp.where(done, episode_lengths, 1.0)
                
                # skill_proportions will be (NUM_ENVS, MAX_NUM_SKILLS)
                skill_proportions = updated_skill_timesteps / jnp.maximum(safe_episode_lengths[:, None], 1.0)

                updated_final_skill_timesteps = final_skill_timesteps * (1 - done_expanded) + skill_proportions * done_expanded
                updated_skill_timesteps = updated_skill_timesteps * (1 - done_expanded)

                # Detect damage sources and update counters
                damage_occurred = detect_damage_sources(prev_env_state.env_state, env_state.env_state, last_obs, base_obs, action, done)
                updated_damage_counters = damage_counters + damage_occurred
                
                # Update final damage counters for completed episodes
                updated_final_damage_counters = final_damage_counters * (1 - done_expanded) + updated_damage_counters * done_expanded
                updated_damage_counters = updated_damage_counters * (1 - done_expanded)

                transition = Transition(
                    done=done,
                    action=action,
                    value=value,
                    reward=reward,
                    reward_i=reward_i,
                    reward_e=reward_e,
                    log_prob=log_prob, 
                    obs=last_obs,
                    next_obs=obs,
                    info=info,
                    skill_idx=last_skill_indices,
                )

                runner_state = (
                    train_states,
                    env_state,
                    updated_intrinsic_rewards,
                    updated_final_intrinsic_rewards,
                    updated_skill_timesteps,
                    updated_final_skill_timesteps,
                    current_skill_durations,
                    updated_damage_counters,
                    updated_final_damage_counters,
                    obs, 
                    skill_indices,
                    rng,
                    update_step,
                )
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            (
                train_states,
                env_state,
                intrinsic_rewards,
                final_intrinsic_rewards,
                skill_timesteps,
                final_skill_timesteps,
                current_skill_durations,
                damage_counters,
                final_damage_counters,
                last_obs,
                skill_indices,
                rng,
                update_step, 
            ) = runner_state


            # CALCULATE ADVANTAGE - Use network selection for final value
            def network_value_selector(skill_idx, train_states, obs):
                pi, value = jax.lax.switch(skill_idx, network_branches, train_states, obs)
                return value
            
            last_val = jax.vmap(network_value_selector, in_axes=(0, None, 0))(skill_indices, train_states, last_obs) 

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORKS - Separate training for each skill
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_states, batch_info):
                    traj_batch, advantages, targets = batch_info
                    
                    # Helper function to get indices for each skill (more memory efficient)
                    def get_skill_indices(skill_id):
                        mask = (traj_batch.skill_idx == skill_id)
                        valid_samples = jnp.sum(mask)
                        return mask, valid_samples
                    
                    # Update function for a single network using masks instead of filtering
                    def update_single_network(skill_id, train_state):
                        mask, valid_samples = get_skill_indices(skill_id)
                        
                        def _loss_fn(params, batch, gae, tgts):
                            # RERUN NETWORK on all observations (no filtering)
                            pi, value = networks[skill_id].apply(params, batch.obs)
                            log_prob = pi.log_prob(batch.action)
                            
                            # CALCULATE VALUE LOSS - only on masked samples
                            value_pred_clipped = batch.value + (
                                value - batch.value
                            ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                            value_losses = jnp.square(value - tgts)
                            value_losses_clipped = jnp.square(value_pred_clipped - tgts)
                            value_loss_raw = 0.5 * jnp.maximum(value_losses, value_losses_clipped)
                            
                            # Apply mask and normalize by valid samples
                            value_loss = jnp.where(mask, value_loss_raw, 0).sum() / jnp.maximum(valid_samples, 1)

                            # CALCULATE ACTOR LOSS - only on masked samples
                            ratio = jnp.exp(log_prob - batch.log_prob)
                            # Normalize GAE only over masked samples
                            masked_gae = jnp.where(mask, gae, 0)
                            gae_mean = masked_gae.sum() / jnp.maximum(valid_samples, 1)
                            gae_var = jnp.where(mask, (gae - gae_mean)**2, 0).sum() / jnp.maximum(valid_samples, 1)
                            gae_normalized = (gae - gae_mean) / jnp.sqrt(gae_var + 1e-8)
                            
                            loss_actor1 = ratio * gae_normalized
                            loss_actor2 = (
                                jnp.clip(
                                    ratio,
                                    1.0 - config["CLIP_EPS"],
                                    1.0 + config["CLIP_EPS"],
                                )
                                * gae_normalized
                            )
                            loss_actor_raw = -jnp.minimum(loss_actor1, loss_actor2)
                            loss_actor = jnp.where(mask, loss_actor_raw, 0).sum() / jnp.maximum(valid_samples, 1)
                            
                            # CALCULATE ENTROPY - only on masked samples
                            entropy_raw = pi.entropy()
                            entropy = jnp.where(mask, entropy_raw, 0).sum() / jnp.maximum(valid_samples, 1)

                            total_loss = (
                                loss_actor
                                + config["VF_COEF"] * value_loss
                                - config["ENT_COEF"] * entropy
                            )
                            return total_loss, (value_loss, loss_actor, entropy)
                        
                        # Only update if we have valid samples
                        def update_fn():
                            grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                            total_loss, grads = grad_fn(
                                train_state.params, traj_batch, advantages, targets
                            )
                            updated_state = train_state.apply_gradients(grads=grads)
                            return updated_state, total_loss
                        
                        def no_update_fn():
                            # Return unchanged state and zero loss
                            return train_state, (jnp.array(0.0), (jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)))
                        
                        return jax.lax.cond(valid_samples > 0, update_fn, no_update_fn)
                    
                    # Update all networks
                    updated_train_states = []
                    losses = []
                    for skill_id in range(config["MAX_NUM_SKILLS"]):
                        updated_state, loss = update_single_network(skill_id, train_states[skill_id])
                        updated_train_states.append(updated_state)
                        losses.append(loss)
                    
                    return updated_train_states, losses

                (
                    train_states,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree.map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_states, losses = jax.lax.scan(
                    _update_minbatch, train_states, minibatches
                )
                update_state = (
                    train_states,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, losses

            update_state = (
                train_states,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )

            train_states = update_state[0]


            _ , _ , _ , final_intrinsic_rewards_log, _ , final_skill_timesteps_log, _, _, final_damage_counters_log, _ , _, _ , _ = runner_state
            
            info_extended = traj_batch.info
            # Dynamic skill metrics based on MAX_NUM_SKILLS
            for skill_id in range(config["MAX_NUM_SKILLS"]):
                info_extended[f"final_intrinsic_rewards_skill_{skill_id}"] = final_intrinsic_rewards_log[:, skill_id]
                info_extended[f"final_skill_timesteps_skill_{skill_id}"] = final_skill_timesteps_log[:, skill_id]
            info_extended["damage_thirst_total"] = final_damage_counters_log[:, 0]
            info_extended["damage_hunger_total"] = final_damage_counters_log[:, 1]
            info_extended["damage_energy_total"] = final_damage_counters_log[:, 2]
            info_extended["damage_zombie_total"] = final_damage_counters_log[:, 3]
            info_extended["damage_arrow_total"] = final_damage_counters_log[:, 4]
            info_extended["damage_lava_total"] = final_damage_counters_log[:, 5]


            metric = jax.tree.map(
                lambda x: (x * traj_batch.info["returned_episode"]).sum()
                / (traj_batch.info["returned_episode"].sum()),
                info_extended,
            )

            rng = update_state[-1]

            def _maybe_eval_callback(train_states, config, update_step, networks):
                for i, eval_step in enumerate(eval_steps):
                    if update_step >= eval_step and eval_step not in already_evaluated:
                        # jax.debug.breakpoint()
                        # print(f"[Eval] Running evaluation at update {update_step}")
                        run_eval_and_plot(train_states, config, update_step, int(100*config["EVAL_FRACTIONS"][i]), networks)
                        already_evaluated.add(eval_step)            
            jax.debug.callback(_maybe_eval_callback, train_states, config, update_step, networks)
            
            # wandb logging
            if config["DEBUG"] and config["USE_WANDB"]:

                def callback(metric, update_step):
                    to_log = create_log_dict(metric, config)
                    # Dynamic skill logging based on MAX_NUM_SKILLS
                    for skill_id in range(config["MAX_NUM_SKILLS"]):
                        to_log[f"final_intrinsic_rewards_skill_{skill_id}"] = metric[f"final_intrinsic_rewards_skill_{skill_id}"]
                        to_log[f"final_skill_timesteps_skill_{skill_id}"] = metric[f"final_skill_timesteps_skill_{skill_id}"]
                    to_log["damage_thirst_total"] = metric["damage_thirst_total"]
                    to_log["damage_hunger_total"] = metric["damage_hunger_total"]
                    to_log["damage_energy_total"] = metric["damage_energy_total"]
                    to_log["damage_zombie_total"] = metric["damage_zombie_total"]
                    to_log["damage_arrow_total"] = metric["damage_arrow_total"]
                    to_log["damage_lava_total"] = metric["damage_lava_total"]
                    batch_log(update_step, to_log, config)

                jax.debug.callback(
                    callback,
                    metric,
                    update_step,
                )

            runner_state = (
                train_states,
                env_state,
                intrinsic_rewards,
                final_intrinsic_rewards,
                skill_timesteps,
                final_skill_timesteps,
                current_skill_durations,
                damage_counters,
                final_damage_counters,
                last_obs,
                skill_indices,
                rng,
                update_step + 1,
            )

            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_states,
            env_state,
            intrinsic_rewards,
            final_intrinsic_rewards,
            skill_timesteps,
            final_skill_timesteps,
            current_skill_durations,
            damage_counters,
            final_damage_counters,
            obs,
            skill_indices,
            _rng,
            0, 
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state} 


    return train


def run_ppo(config):
    config = {k.upper(): v for k, v in config.__dict__.items()}

    if config["USE_WANDB"]:
        wandb.init(
            project=config["WANDB_PROJECT"],
            entity=config["WANDB_ENTITY"],
            config=config,
            name=config["ENV_NAME"]
            + "-"
            + str(int(config["TOTAL_TIMESTEPS"] // 1e6))
            + "M",
        )
        # wandb.define_metric("eval/active_skill*", step_metric="eval/timestep")

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_REPEATS"])

    train_jit = jax.jit(make_train(config))
    train_vmap = jax.vmap(train_jit)

    t0 = time.time()
    out = train_vmap(rngs)
    t1 = time.time()
    print("Time to run experiment", t1 - t0)
    print("SPS: ", config["TOTAL_TIMESTEPS"] / (t1 - t0))

    if config["USE_WANDB"]:

        def _save_network(rs_index, dir_name):
            runner_state = out["runner_state"][rs_index]
            train_states = jax.tree.map(lambda x: x[0], runner_state)  # Extract train_states from runner_state
            
            # Save each skill network separately
            for skill_id in range(config["MAX_NUM_SKILLS"]):
                skill_train_state = train_states[skill_id]
                orbax_checkpointer = PyTreeCheckpointer()
                options = CheckpointManagerOptions(max_to_keep=1, create=True)
                path = os.path.join(wandb.run.dir, f"{dir_name}_skill_{skill_id}")
                checkpoint_manager = CheckpointManager(path, orbax_checkpointer, options)
                print(f"saved skill {skill_id} network to {path}")
                save_args = orbax_utils.save_args_from_target(skill_train_state)
                checkpoint_manager.save(
                    int(config["TOTAL_TIMESTEPS"]),
                    skill_train_state,
                    save_kwargs={"save_args": save_args},
                )

        if config["SAVE_POLICY"]:
            _save_network(0, "policies")


def run_eval_and_plot(train_states, config, update_step, update_frac, networks):
    """
    Run a single-episode eval, track skill indices and vital stats, plot and save/log.
    """
    
    # Create evaluation network branch functions (same as in training)
    def make_eval_network_branch(network_idx):
        def branch_fn(train_states, obs):
            return networks[network_idx].apply(train_states[network_idx].params, obs)
        return branch_fn
    
    eval_network_branches = [make_eval_network_branch(i) for i in range(config["MAX_NUM_SKILLS"])]
    
    env = make_craftax_env_from_name(config["ENV_NAME"], not config["USE_OPTIMISTIC_RESETS"])
    env = EnergyWrapper(env)
    env = LogWrapper(env)
    env = AutoResetEnvWrapper(env)
    env_params = env.default_params

    # Calculate observation indices for mob counting
    NUM_BLOCK_TYPES = len(BlockType)
    NUM_MOB_TYPES = 4
    all_map_flat_size = OBS_DIM[0] * OBS_DIM[1] * (NUM_BLOCK_TYPES + NUM_MOB_TYPES)
    
    # Helper function to count mobs in 3x3 area around player
    def count_mobs_nearby(obs_flat):
        # Player is always at center of observation
        center_x, center_y = OBS_DIM[0] // 2, OBS_DIM[1] // 2
        
        # Extract and reshape map observation
        map_obs = obs_flat[:all_map_flat_size]
        map_obs = map_obs.reshape(OBS_DIM[0], OBS_DIM[1], NUM_BLOCK_TYPES + NUM_MOB_TYPES)
        
        # Extract 3x3 area around player, focus on mob channels
        local_area = map_obs[center_x-1:center_x+2, center_y-1:center_y+2, NUM_BLOCK_TYPES:]
        
        # Count total mobs in the 3x3 area (sum across all mob types and positions)
        total_mobs = jnp.sum(local_area)
        
        return total_mobs

    # Helper function to check if adjacent to lava
    def is_adjacent_to_lava(obs_flat):
        # Player is always at center of observation
        center_x, center_y = OBS_DIM[0] // 2, OBS_DIM[1] // 2
        
        # Extract and reshape map observation
        map_obs = obs_flat[:all_map_flat_size]
        map_obs = map_obs.reshape(OBS_DIM[0], OBS_DIM[1], NUM_BLOCK_TYPES + NUM_MOB_TYPES)
        
        # Check the 4 adjacent positions for lava
        adjacent_lava = (
            map_obs[center_x-1, center_y, BlockType.LAVA.value] +    # up
            map_obs[center_x+1, center_y, BlockType.LAVA.value] +    # down
            map_obs[center_x, center_y-1, BlockType.LAVA.value] +    # left
            map_obs[center_x, center_y+1, BlockType.LAVA.value]      # right
        )
        
        return adjacent_lava > 0

    rng = jax.random.PRNGKey(config["SEED"] + update_step)
    obs, env_state = env.reset(rng, env_params)
    done = False
    skill_trace = []
    health_trace = []
    food_trace = []
    drink_trace = []
    energy_trace = []
    reward_trace = []
    mobs_nearby_trace = []
    lava_adjacent_trace = []
    t = 0
    current_skill_duration = jnp.array(0)
    last_skill_index = 0
    last_state = env_state
    last_obs = obs
    should_terminate_skill = True

    while not done and t < 1000:
        last_obs = last_obs.flatten()
        if should_terminate_skill:
            curr_skill_index = skill_selector_my_two_skills(last_obs)
            current_skill_duration = jnp.array(0) # this isn't 0 since it could pick the same skill again.
        else:
            curr_skill_index = last_skill_index
            current_skill_duration = current_skill_duration + 1

        # Use network selection instead of skill vector concatenation
        last_obs_batch = jnp.expand_dims(last_obs, axis=0)
        
        # Select appropriate network based on skill
        pi, value = jax.lax.switch(curr_skill_index, eval_network_branches, train_states, last_obs_batch)

        rng, _rng = jax.random.split(rng)
        action = pi.sample(seed=_rng)[0]
        rng, _rng = jax.random.split(rng)
        base_obs, env_state, reward_e, done, info = env.step(_rng, env_state, action, env_params)
        def get_termination_single(index, last_b_obs_s, b_obs_s, duration, done_val):
            terminate_fns = [terminate_harvest, terminate_craft, terminate_sustain][:config["MAX_NUM_SKILLS"]]
            return jax.lax.switch(index, terminate_fns, last_b_obs_s, b_obs_s, duration, done_val)

        should_terminate_skill = get_termination_single(curr_skill_index, last_obs, base_obs, current_skill_duration, done)

        # Calculate active skill reward
        # reward_fns_single = [my_harvesting_reward_fn, my_crafting_reward_fn, my_survival_reward_fn]
        # def select_reward_single(index, last_b_obs_s, b_obs_s, done_val):
        #     return jax.lax.switch(index, reward_fns_single, last_b_obs_s, b_obs_s, done_val)
        # skill_reward = select_reward_single(curr_skill_index, last_obs[:-config["MAX_NUM_SKILLS"]], base_obs, done)

        # Dynamic reward function selection
        reward_fns_single = [my_harvesting_crafting_reward_fn_state, my_survival_reward_fn_state][:config["MAX_NUM_SKILLS"]]
        # reward_fns_single = [configurable_achievement_reward_fn]
        def select_reward_single(index, prev_state, cur_state, done_val):
            return jax.lax.switch(index, reward_fns_single, prev_state, cur_state, done_val)
        skill_reward = select_reward_single(curr_skill_index, last_state.env_state, env_state.env_state, done)
        

        # Count mobs nearby
        mobs_nearby = count_mobs_nearby(base_obs)
        # Check if adjacent to lava
        lava_adjacent = is_adjacent_to_lava(base_obs)
        # Track skill and vital stats
        skill_trace.append(curr_skill_index)
        health_trace.append(env_state.env_state.player_health)  # Direct access from state
        food_trace.append(env_state.env_state.player_food)
        drink_trace.append(env_state.env_state.player_drink)
        energy_trace.append(env_state.env_state.player_energy)
        reward_trace.append(skill_reward)
        mobs_nearby_trace.append(mobs_nearby)
        lava_adjacent_trace.append(lava_adjacent)
        # if config["USE_WANDB"]:
        #     wandb.log({
        #         "eval/timestep": t,
        #         f"eval/active_skill_{update_frac}": curr_skill_index
        #     }, commit=False)
        last_skill_index = curr_skill_index
        last_obs = base_obs
        last_state = env_state
        t += 1
        if hasattr(done, "item"):
            done = done.item()
    
    # Create combined subplot layout with all plots
    timesteps = list(range(len(skill_trace)))
    fig, axes = plt.subplots(5, 2, figsize=(12, 20))
    fig.suptitle(f'Agent Performance During Evaluation (Update {update_frac}%)', fontsize=16, y=0.98)
    
    # Skill plot (top left)
    axes[0, 0].step(timesteps, skill_trace, where='post', color='blue', linewidth=2)
    axes[0, 0].set_title('Active Skill')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(-0.5, 2.5)
    axes[0, 0].set_yticks([0, 1, 2])
    
    # Active skill reward plot (top right)
    axes[0, 1].plot(timesteps, reward_trace, color='purple', linewidth=2)
    axes[0, 1].set_title('Active Skill Reward')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Health plot (second row left)
    axes[1, 0].plot(timesteps, health_trace, color='red', linewidth=2)
    axes[1, 0].set_title('Health')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 10)
    
    # Food plot (second row right)
    axes[1, 1].plot(timesteps, food_trace, color='green', linewidth=2)
    axes[1, 1].set_title('Food')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 10)
    
    # Drink plot (third row left)
    axes[2, 0].plot(timesteps, drink_trace, color='cyan', linewidth=2)
    axes[2, 0].set_title('Drink')
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].set_ylim(0, 10)
    
    # Energy plot (third row right)
    axes[2, 1].plot(timesteps, energy_trace, color='orange', linewidth=2)
    axes[2, 1].set_title('Energy')
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].set_ylim(0, 10)
    
    # Mobs nearby plot (bottom left)
    axes[3, 0].plot(timesteps, mobs_nearby_trace, color='brown', linewidth=2)
    axes[3, 0].set_title('Mobs Nearby (3x3)')
    axes[3, 0].grid(True, alpha=0.3)
    axes[3, 0].set_ylim(-0.1, 3.1)
    axes[3, 0].set_yticks([0, 1, 2, 3])
    
    # Combined vital stats plot (bottom right)
    axes[3, 1].plot(timesteps, health_trace, color='red', linewidth=2)
    axes[3, 1].plot(timesteps, food_trace, color='green', linewidth=2)
    axes[3, 1].plot(timesteps, drink_trace, color='cyan', linewidth=2)
    axes[3, 1].plot(timesteps, energy_trace, color='orange', linewidth=2)
    axes[3, 1].set_title('Combined Vital Stats')
    axes[3, 1].grid(True, alpha=0.3)
    axes[3, 1].set_ylim(0, 10)
    
    # Lava adjacent plot (fifth row left)
    axes[4, 0].plot(timesteps, lava_adjacent_trace, color='red', linewidth=2)
    axes[4, 0].set_title('Adjacent to Lava')
    axes[4, 0].grid(True, alpha=0.3)
    axes[4, 0].set_ylim(-0.1, 1.1)
    axes[4, 0].set_yticks([0, 1])
    
    # Empty plot (fifth row right) - placeholder
    axes[4, 1].axis('off')
    
    # Adjust layout with proper spacing
    plt.tight_layout()
    
    # Log combined plot to wandb if enabled
    if config["USE_WANDB"]:
        wandb.log({f"Eval/agent_performance_{update_frac}": fig}, commit=False)
    
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Craftax-Classic-Symbolic-v1")
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--total_timesteps", type=lambda x: int(float(x)), default=1e8
    )  # Allow scientific notation
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_steps", type=int, default=64)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--num_minibatches", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.8)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--activation", type=str, default="tanh")
    parser.add_argument(
        "--anneal_lr", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--use_wandb", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--save_policy", action="store_true")
    parser.add_argument("--num_repeats", type=int, default=1)
    parser.add_argument("--layer_size", type=int, default=512)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument(
        "--use_optimistic_resets", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--optimistic_reset_ratio", type=int, default=16)

    # SKILLS
    parser.add_argument("--max_num_skills", type=int, default=2, help="Number of distinct skills (harvest/craft combined, sustain)") # Default to 2

    args, rest_args = parser.parse_known_args(sys.argv[1:])
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    if args.seed is None:
        args.seed = np.random.randint(2**31)

    if args.jit:
        run_ppo(args)
    else:
        with jax.disable_jit():
            run_ppo(args)

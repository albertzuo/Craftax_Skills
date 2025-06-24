import argparse
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from craftax.craftax_env import make_craftax_env_from_name # Added BlockType

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
from models.discrim import Discriminator
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
    my_survival_reward_function,
)
from meta_policy.skill_training import (
    skill_selector,
    # skill_selector_v2,
    # skill_selector_v3,
    skill_selector_two_skills,
    skill_selector_my_two_skills,
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
        # INIT NETWORK
        if "Symbolic" in config["ENV_NAME"]:
            network = ActorCritic(env.action_space(env_params).n, config["LAYER_SIZE"])
        else:
            network = ActorCriticConv(
                env.action_space(env_params).n, config["LAYER_SIZE"]
            )
        
        discriminator = Discriminator(config["MAX_NUM_SKILLS"], config["LAYER_SIZE"])

        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(
            (1, env.observation_space(env_params).shape[0] + config["MAX_NUM_SKILLS"])
        )
        network_params = network.init(_rng, init_x)
        
        init_discriminator_x = jnp.zeros(   
            (1, env.observation_space(env_params).shape[0])
        )
        rng, _rng = jax.random.split(rng)
        discriminator_params = discriminator.init(_rng, init_discriminator_x)
        
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
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )
        
        discriminator_state = TrainState.create(
            apply_fn=discriminator.apply,
            params=discriminator_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        obsv, env_state = env.reset(_rng, env_params)

        rng, _rng = jax.random.split(rng)
        # skill_indices = jax.random.randint(
        #     _rng, (config["NUM_ENVS"],), 0, config["MAX_NUM_SKILLS"]
        # )
        skill_indices = jnp.full((config["NUM_ENVS"],), 0, dtype=jnp.int32) # init to skill 0
        skill_vectors = jax.nn.one_hot(skill_indices, config["MAX_NUM_SKILLS"])

        def augment_obs_with_skill(obsv, skill_vec):
            return jnp.concatenate([obsv, skill_vec], axis=-1)

        obsv = jax.vmap(augment_obs_with_skill)(obsv, skill_vectors)
        intrinsic_rewards = jnp.zeros((config["NUM_ENVS"], config["MAX_NUM_SKILLS"]))
        final_intrinsic_rewards = jnp.zeros((config["NUM_ENVS"], config["MAX_NUM_SKILLS"]))
        skill_timesteps = jnp.zeros((config["NUM_ENVS"], config["MAX_NUM_SKILLS"]))
        final_skill_timesteps = jnp.zeros((config["NUM_ENVS"], config["MAX_NUM_SKILLS"]))
        current_skill_durations = jnp.zeros(config["NUM_ENVS"])  # Track current duration of active skill

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                (
                    train_state,
                    discriminator_state,
                    env_state,
                    intrinsic_rewards,
                    final_intrinsic_rewards,
                    skill_timesteps,
                    final_skill_timesteps,
                    current_skill_durations,
                    last_obs,
                    rng,
                    update_step,
                ) = runner_state

                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                base_obsv, env_state, reward_e, done, info = env.step(
                    _rng, env_state, action, env_params
                )

                last_skill_indices = jnp.argmax(last_obs[:, -config["MAX_NUM_SKILLS"]:], axis=-1)
                
                # Get termination conditions for each skill
                def get_termination_single(index, last_b_obs_s, b_obs_s, duration):
                    terminate_fns = [terminate_harvest, terminate_craft, terminate_sustain]
                    return jax.lax.switch(index, terminate_fns, last_b_obs_s, b_obs_s, duration)
                
                # Check termination conditions
                should_terminate = jax.vmap(get_termination_single)(
                    last_skill_indices, last_obs[:, :-config["MAX_NUM_SKILLS"]], base_obsv, current_skill_durations
                )
                
                # Only select new skills for environments that should terminate
                new_skill_indices = jax.vmap(skill_selector)(base_obsv)
                skill_indices = jnp.where(should_terminate, new_skill_indices, last_skill_indices)
                
                # Update current skill durations
                current_skill_durations = jnp.where(
                    should_terminate,
                    jnp.zeros_like(current_skill_durations),  # Reset to 0 for terminated skills
                    current_skill_durations + 1  # Increment for continuing skills
                )
                
                skill_vectors = jax.nn.one_hot(skill_indices, config["MAX_NUM_SKILLS"])
                obsv = jax.vmap(augment_obs_with_skill)(base_obsv, skill_vectors)
                last_base_obsv = last_obs[:, :-config["MAX_NUM_SKILLS"]]

                # Calculate discriminator reward
                discriminator_output = discriminator.apply(discriminator_state.params, last_base_obsv)
                logq_z = discriminator_output.log_prob(last_skill_indices)  # Log probability of true skill
                log_p_z = -jnp.log(config["MAX_NUM_SKILLS"])  # Log probability under uniform prior
                # discriminator_reward = (logq_z - log_p_z) * config["DIAYN_REWARD_COEF"]
                discriminator_reward = jnp.clip(logq_z - log_p_z, -10.0, 1.1) * config["DIAYN_REWARD_COEF"]

                reward_fns_single = [my_harvesting_reward_fn, my_crafting_reward_fn, my_survival_reward_function]
                def select_reward_single(index, last_b_obs_s, b_obs_s):
                    return jax.lax.switch(index, reward_fns_single, last_b_obs_s, b_obs_s)

                reward_i = jax.vmap(select_reward_single)(last_skill_indices, last_base_obsv, base_obsv)

                reward = reward_i #+ discriminator_reward #+ reward_e
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

                transition = Transition(
                    done=done,
                    action=action,
                    value=value,
                    reward=reward,
                    reward_i=reward_i,
                    reward_e=reward_e,
                    log_prob=log_prob, 
                    obs=last_obs,
                    next_obs=obsv, 
                    info=info,
                )

                runner_state = (
                    train_state,
                    discriminator_state,
                    env_state,
                    updated_intrinsic_rewards,
                    updated_final_intrinsic_rewards,
                    updated_skill_timesteps,
                    updated_final_skill_timesteps,
                    current_skill_durations,  # Add to runner state
                    obsv, 
                    rng,
                    update_step,
                )
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            (
                train_state,
                discriminator_state,
                env_state,
                intrinsic_rewards,
                final_intrinsic_rewards,
                skill_timesteps,
                final_skill_timesteps,
                current_skill_durations,  # Add to runner state
                last_obs, 
                rng,
                update_step, 
            ) = runner_state


            # CALCULATE ADVANTAGE
            _, last_val = network.apply(train_state.params, last_obs) 

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

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(combined_state, batch_info):
                    train_state, discriminator_state = combined_state
                    traj_batch, advantages, targets = batch_info

                    # Policy/value network
                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    # Discriminator network
                    def _discriminator_loss_fn(params, traj_batch):
                        # Extract skill indices from the previous timestep's observations
                        skill_indices = jnp.argmax(traj_batch.obs[:, -config["MAX_NUM_SKILLS"]:], axis=-1)
                        # Use the current observation (without skill encoding) to predict the previous skill
                        base_obs = traj_batch.next_obs[:, :-config["MAX_NUM_SKILLS"]]
                        discriminator_output = discriminator.apply(params, base_obs)
                        discriminator_loss = -discriminator_output.log_prob(skill_indices).mean()
                        return discriminator_loss

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)

                    # Update discriminator network
                    discriminator_grad_fn = jax.value_and_grad(_discriminator_loss_fn)
                    discriminator_loss, discriminator_grads = discriminator_grad_fn(
                        discriminator_state.params, traj_batch
                    )
                    discriminator_state = discriminator_state.apply_gradients(grads=discriminator_grads)

                    losses = (total_loss, discriminator_loss)
                    return (train_state, discriminator_state), losses

                (
                    train_state,
                    discriminator_state,
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
                (train_state, discriminator_state), losses = jax.lax.scan(
                    _update_minbatch, (train_state, discriminator_state), minibatches
                )
                update_state = (
                    train_state,
                    discriminator_state,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, losses

            update_state = (
                train_state,
                discriminator_state,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )


            train_state = update_state[0]


            _ , _, _ , _ , final_intrinsic_rewards_log, _ , final_skill_timesteps_log, _, _ , _ , _ = runner_state
            
            info_extended = traj_batch.info
            info_extended["final_intrinsic_rewards_skill_0"] = final_intrinsic_rewards_log[:, 0]
            info_extended["final_intrinsic_rewards_skill_1"] = final_intrinsic_rewards_log[:, 1]
            info_extended["final_intrinsic_rewards_skill_2"] = final_intrinsic_rewards_log[:, 2]
            info_extended["final_skill_timesteps_skill_0"] = final_skill_timesteps_log[:, 0]
            info_extended["final_skill_timesteps_skill_1"] = final_skill_timesteps_log[:, 1]
            info_extended["final_skill_timesteps_skill_2"] = final_skill_timesteps_log[:, 2]


            metric = jax.tree.map(
                lambda x: (x * traj_batch.info["returned_episode"]).sum()
                / (traj_batch.info["returned_episode"].sum()),
                info_extended,
            )

            rng = update_state[-1]

            def _maybe_eval_callback(train_state, config, update_step, network):
                # update_step = int(np.asarray(update_step))
                for i, eval_step in enumerate(eval_steps):
                    if update_step >= eval_step and eval_step not in already_evaluated:
                        # jax.debug.breakpoint()
                        # print(f"[Eval] Running evaluation at update {update_step}")
                        run_eval_and_plot(train_state, config, update_step, int(100*config["EVAL_FRACTIONS"][i]), network)
                        already_evaluated.add(eval_step)            
            jax.debug.callback(_maybe_eval_callback, train_state, config, update_step, network)
            
            # wandb logging
            if config["DEBUG"] and config["USE_WANDB"]:

                def callback(metric, update_step):
                    to_log = create_log_dict(metric, config)
                    to_log["final_intrinsic_rewards_skill_0"] = metric["final_intrinsic_rewards_skill_0"]
                    to_log["final_intrinsic_rewards_skill_1"] = metric["final_intrinsic_rewards_skill_1"]
                    to_log["final_intrinsic_rewards_skill_2"] = metric["final_intrinsic_rewards_skill_2"]
                    to_log["final_skill_timesteps_skill_0"] = metric["final_skill_timesteps_skill_0"]
                    to_log["final_skill_timesteps_skill_1"] = metric["final_skill_timesteps_skill_1"]
                    to_log["final_skill_timesteps_skill_2"] = metric["final_skill_timesteps_skill_2"]
                    batch_log(update_step, to_log, config)

                jax.debug.callback(
                    callback,
                    metric,
                    update_step,
                )

            runner_state = (
                train_state,
                discriminator_state,
                env_state,
                intrinsic_rewards,
                final_intrinsic_rewards,
                skill_timesteps,
                final_skill_timesteps,
                current_skill_durations,  # Add to runner state
                last_obs, 
                rng,
                update_step + 1,
            )

            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            discriminator_state,
            env_state,
            intrinsic_rewards,
            final_intrinsic_rewards,
            skill_timesteps,
            final_skill_timesteps,
            current_skill_durations,  # Add to runner state
            obsv, 
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
            train_states = out["runner_state"][rs_index]
            train_state = jax.tree.map(lambda x: x[0], train_states)
            orbax_checkpointer = PyTreeCheckpointer()
            options = CheckpointManagerOptions(max_to_keep=1, create=True)
            path = os.path.join(wandb.run.dir, dir_name)
            checkpoint_manager = CheckpointManager(path, orbax_checkpointer, options)
            print(f"saved runner state to {path}")
            save_args = orbax_utils.save_args_from_target(train_state)
            checkpoint_manager.save(
                int(config["TOTAL_TIMESTEPS"]),
                train_state,
                save_kwargs={"save_args": save_args},
            )

        if config["SAVE_POLICY"]:
            _save_network(0, "policies")


def run_eval_and_plot(train_state, config, update_step, update_frac, network):
    """
    Run a single-episode eval, track skill indices, plot and save/log.
    """
    
    env = make_craftax_env_from_name(config["ENV_NAME"], not config["USE_OPTIMISTIC_RESETS"])
    env = LogWrapper(env)
    env = AutoResetEnvWrapper(env)
    env_params = env.default_params

    rng = jax.random.PRNGKey(config["SEED"] + update_step)
    obsv, env_state = env.reset(rng, env_params)
    done = False
    skill_trace = []
    t = 0
    current_skill_duration = 0
    last_skill_index = 0
    last_obs = obsv
    should_terminate_skill = True

    while not done and t < 1000:
        last_obs = last_obs.flatten()
        if should_terminate_skill:
            curr_skill_index = skill_selector(last_obs)
            current_skill_duration = 0
        else:
            curr_skill_index = last_skill_index
            current_skill_duration += 1
        skill_vector = jax.nn.one_hot(curr_skill_index, config["MAX_NUM_SKILLS"])
        last_obs = jnp.concatenate([last_obs, skill_vector])

        
        last_obs_batch = jnp.expand_dims(last_obs, axis=0)
        pi, value = network.apply(train_state.params, last_obs_batch)

        rng, _rng = jax.random.split(rng)
        action = pi.sample(seed=_rng)[0]
        rng, _rng = jax.random.split(rng)
        base_obs, env_state, reward_e, done, info = env.step(_rng, env_state, action, env_params)

        
        def get_termination_single(index, last_b_obs_s, b_obs_s, duration):
            terminate_fns = [terminate_harvest, terminate_craft, terminate_sustain]
            return jax.lax.switch(index, terminate_fns, last_b_obs_s, b_obs_s, duration)

        should_terminate_skill = get_termination_single(curr_skill_index, last_obs[:-config["MAX_NUM_SKILLS"]], base_obs, current_skill_duration)

        skill_trace.append(curr_skill_index)
        # if config["USE_WANDB"]:
        #     wandb.log({
        #         "eval/timestep": t,
        #         f"eval/active_skill_{update_frac}": curr_skill_index
        #     }, commit=False)
        last_skill_index = curr_skill_index
        last_obs = base_obs
        t += 1
        if hasattr(done, "item"):
            done = done.item()
    
    # Plot step plot using matplotlib
    timesteps = list(range(len(skill_trace)))
    plt.step(timesteps, skill_trace, where='post')
    plt.xlabel('Timestep')
    plt.ylabel('Active Skill')
    plt.title(f'Active Skill Over Time (Update {update_frac})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Log step plot to wandb if enabled
    if config["USE_WANDB"]:
        wandb.log({f"Active Skill {update_frac}": plt}, commit=False)
        # wandb.log({f"eval/active_skill_{update_frac}": wandb.plot.line(table, "timestep", "skill_index", title=f"Active Skill {update_frac}")}, commit=False)
        # wandb.log({
        #     "eval/timestep": list(range(len(skill_trace))),
        #     f"eval/active_skill_{update_step+1}": skill_trace 
        # })
        # wandb.log({
        #     f"eval/skill_trace_{update_step+1}": wandb.plot.line_series(
        #         xs = list(range(len(skill_trace))),
        #         ys = [skill_trace],
        #         keys = ["Skill"],
        #         title = f"Skill Trace ({update_step+1})"
        #     ) #wandb.plot.line(table, "timestep", "skill_index"),
        # }, step=update_step+1)
        # jax.debug.print("Skill trace size: {}", len(list(enumerate(skill_trace))))
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
    parser.add_argument("--max_num_skills", type=int, default=3, help="Number of distinct skills (harvest, craft, sustain)") # Default to 3
    parser.add_argument("--diayn_reward_coef", type=float, default=0.1, help="Coefficient for the DIAYN discriminator reward")

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

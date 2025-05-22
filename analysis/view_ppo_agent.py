import argparse
import os
import sys

sys.path.append("./")
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
# print(sys.path)
from wrappers import AutoResetEnvWrapper
from flax.training.train_state import TrainState
from orbax.checkpoint import (
    PyTreeCheckpointer,
    CheckpointManagerOptions,
    CheckpointManager,
)
import orbax.checkpoint as ocp

from models.actor_critic import ActorCriticConv, ActorCritic


def main(args):

    with open(os.path.join(args.path, "config.yaml")) as f:
        raw_config = yaml.load(f, Loader=yaml.Loader)

        config = {}
        for key, value in raw_config.items():
            if isinstance(value, dict) and "value" in value:
                config[key] = value["value"]

    config["NUM_ENVS"] = 1

    orbax_checkpointer = PyTreeCheckpointer()
    options = CheckpointManagerOptions(max_to_keep=1, create=True)
    checkpoint_manager = CheckpointManager(
        os.path.join(args.path, "policies"), orbax_checkpointer, options
    )

    is_classic = False

    if config["ENV_NAME"] == "Craftax-Symbolic-v1":
        from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv
        from craftax.craftax.constants import Action

        env = CraftaxSymbolicEnv(CraftaxSymbolicEnv.default_static_params())
        network = ActorCritic(len(Action), config["LAYER_SIZE"])
    elif config["ENV_NAME"] == "Craftax-Pixels-v1":
        from craftax.craftax.envs.craftax_pixels_env import CraftaxPixelsEnv
        from craftax.craftax.constants import Action

        env = CraftaxPixelsEnv(CraftaxPixelsEnv.default_static_params())
        network = ActorCriticConv(len(Action), config["LAYER_SIZE"])
    elif config["ENV_NAME"] == "Craftax-Classic-Symbolic-v1":
        from craftax.craftax_classic.envs.craftax_symbolic_env import (
            CraftaxClassicSymbolicEnv,
        )
        from craftax.craftax_classic.constants import Action

        env = CraftaxClassicSymbolicEnv(
            CraftaxClassicSymbolicEnv.default_static_params()
        )
        network = ActorCritic(len(Action), config["LAYER_SIZE"])
        is_classic = True
    elif config["ENV_NAME"] == "Craftax-Classic-Pixels-v1":
        from craftax.craftax_classic.envs.craftax_pixels_env import (
            CraftaxClassicPixelsEnv,
        )
        from craftax.craftax_classic.constants import Action

        env = CraftaxClassicPixelsEnv(CraftaxClassicPixelsEnv.default_static_params())
        network = ActorCriticConv(len(Action), config["LAYER_SIZE"])
        is_classic = True
    else:
        raise ValueError(f"Unknown env: {config['ENV_NAME']}")

    env = AutoResetEnvWrapper(env)
    env_params = env.default_params

    init_x = jnp.zeros((config["NUM_ENVS"], *env.observation_space(env_params).shape))

    rng = jax.random.PRNGKey(np.random.randint(2**31))
    rng, _rng, __rng = jax.random.split(rng, 3)
    network_params = network.init(_rng, init_x)

    tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(config["LR"], eps=1e-5),
    )
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )

    train_state = checkpoint_manager.restore(
        int(float(config["TOTAL_TIMESTEPS"])), items=train_state
    )

    obs, env_state = env.reset(key=__rng)
    done = 0

    if is_classic:
        from craftax.craftax_classic.play_craftax_classic import CraftaxRenderer
        from craftax.craftax_classic.constants import Achievement
    else:
        from craftax.craftax.play_craftax import CraftaxRenderer
        from craftax.craftax.constants import Achievement

    renderer = CraftaxRenderer(env, env_params, pixel_render_size=1)

    # Initialize video recording if requested
    frames = [] if args.record_video else None

    frame_num = 0
    while not renderer.is_quit_requested() and frame_num < 1000:
        done = np.array([done], dtype=bool)
        obs = jnp.expand_dims(obs, axis=0)

        pi, value = network.apply(train_state.params, obs)
        rng, _rng = jax.random.split(rng)
        action = pi.sample(seed=_rng)[0]
        # action = jnp.argmax(pi.probs[0, 0])

        if action is not None:
            rng, _rng = jax.random.split(rng)
            old_achievements = env_state.achievements
            obs, env_state, reward, done, info = env.step(
                _rng, env_state, action, env_params
            )
            new_achievements = env_state.achievements
            print_new_achievements(Achievement, old_achievements, new_achievements)
            if done:
                print("\n")
        renderer.render(env_state)
        frame_num += 1

        # Capture frame for video if recording
        if args.record_video:
            # Get pygame surface data as a numpy array
            frame = pygame.surfarray.array3d(renderer.screen_surface)
            frame = np.transpose(frame, (1, 0, 2))  # Transpose to correct dimensions
            frames.append(frame)

    # Save video if recording
    if args.record_video and frames:
        print(f"Saving video to {args.record_video}...")
        if frames:
            height, width, _ = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # codec
            video_writer = cv2.VideoWriter(
                args.record_video, fourcc, 30, (width, height)
            )
            for frame in frames:
                # Convert RGB to BGR for OpenCV
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(bgr_frame)
            video_writer.release()
        print(f"Video saved to {args.record_video}")


def print_new_achievements(achievements_cls, old_achievements, new_achievements):
    for i in range(len(old_achievements)):
        if old_achievements[i] == 0 and new_achievements[i] == 1:
            print(f"{achievements_cls(i).name} ({new_achievements.sum()}/{22})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--record_video", type=str)

    args, rest_args = parser.parse_known_args(sys.argv[1:])
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    if args.debug:
        with jax.disable_jit():
            main(args)
    else:
        main(args)

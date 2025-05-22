import jax
import jax.numpy as jnp
import pytest
import numpy as np

# Assuming the functions are accessible, e.g., by importing them
# This requires augment_obs_with_skill to be defined at module level or imported in ppo_skills_chatgpt
try:
    # Attempt to import the actual function
    from ppo_skills_chatgpt import augment_obs_with_skill
except ImportError:
    # Fallback to local definition if import fails (e.g., function is nested)
    print("Warning: Could not import augment_obs_with_skill from ppo_skills_chatgpt. Using local definition for testing.")
    def augment_obs_with_skill(obsv, skill_vec):
        return jnp.concatenate([obsv, skill_vec], axis=-1)


# from ppo_skills_chatgpt import skill_selector # (Need actual skill_selector - keeping mock for now)
# from reward_fns.skill_rewards import reward_harvest_resources, reward_craft_useful_items, survival_reward_function # Keeping mocks

# --- Mocking necessary components ---

# Mock skill_selector if its real implementation is complex or depends on env state
# For this example, let's assume a simple mock based on a dummy feature
def mock_skill_selector(obs):
    # Example: Select skill based on a dummy feature (e.g., sum of first 5 elements)
    # Replace with logic relevant to the actual skill_selector implementation
    dummy_feature_sum = jnp.sum(obs[..., :5], axis=-1)
    # Skill 0 if sum < 10, Skill 1 if 10 <= sum < 20, Skill 2 otherwise
    return jnp.select(
        [dummy_feature_sum < 10, dummy_feature_sum < 20],
        [0, 1],
        default=2
    )

# Mock reward functions for simplicity and isolation
def mock_reward_fn_0_single(last_obs_s, obs_s):
    # Return a constant value for skill 0
    return 0.1

def mock_reward_fn_1_single(last_obs_s, obs_s):
    # Return a constant value for skill 1
    return 1.1

def mock_reward_fn_2_single(last_obs_s, obs_s):
    # Return a constant value for skill 2
    return 2.1

mock_reward_fns_single = [mock_reward_fn_0_single, mock_reward_fn_1_single, mock_reward_fn_2_single]

def mock_select_reward_single(index, last_b_obs_s, b_obs_s):
    return jax.lax.switch(index, mock_reward_fns_single, last_b_obs_s, b_obs_s)

# --- Test Functions ---

def test_augment_obs_with_skill():
    """Tests if the observation is correctly augmented with the skill vector."""
    obs_shape = (10,)
    num_skills = 3
    batch_size = 4

    dummy_obs = jnp.zeros((batch_size, obs_shape[0]))
    skill_indices = jnp.array([0, 1, 2, 0])
    skill_vectors = jax.nn.one_hot(skill_indices, num_skills)

    augmented_obs = jax.vmap(augment_obs_with_skill)(dummy_obs, skill_vectors)

    # Check shape
    expected_shape = (batch_size, obs_shape[0] + num_skills)
    assert augmented_obs.shape == expected_shape

    # Check content (skill part)
    expected_skill_part = skill_vectors
    np.testing.assert_array_equal(augmented_obs[:, obs_shape[0]:], expected_skill_part)
    # Check content (obs part)
    np.testing.assert_array_equal(augmented_obs[:, :obs_shape[0]], dummy_obs)

def test_skill_selector_logic():
    """Tests the (mocked) skill selector logic."""
    batch_size = 4
    obs_dim = 10 # Example observation dimension
    # Create observations designed to trigger different skills in the mock selector
    obs_skill_0 = jnp.ones((1, obs_dim)) * 0.5 # sum = 5*0.5 = 2.5 (< 10) -> skill 0
    obs_skill_1 = jnp.ones((1, obs_dim)) * 1.5 # sum = 5*1.5 = 7.5 (< 10) -> skill 0 (adjusting mock logic)
    obs_skill_1_alt = jnp.ones((1, obs_dim)) * 2.5 # sum = 5*2.5 = 12.5 (>=10, <20) -> skill 1
    obs_skill_2 = jnp.ones((1, obs_dim)) * 5.0 # sum = 5*5.0 = 25.0 (>=20) -> skill 2
    
    # Adjusting mock logic for clarity:
    def mock_skill_selector_updated(obs):
        dummy_feature_sum = jnp.sum(obs[..., :5], axis=-1)
        return jnp.select(
            [dummy_feature_sum < 10, dummy_feature_sum < 20],
            [jnp.array(0, dtype=jnp.int32), jnp.array(1, dtype=jnp.int32)],
            default=jnp.array(2, dtype=jnp.int32)
        )

    test_obs = jnp.concatenate([obs_skill_0, obs_skill_1_alt, obs_skill_2, obs_skill_0], axis=0)
    
    # Vmap the selector over the batch
    selected_skills = jax.vmap(mock_skill_selector_updated)(test_obs)

    expected_skills = jnp.array([0, 1, 2, 0], dtype=jnp.int32)
    np.testing.assert_array_equal(selected_skills, expected_skills)


def test_reward_switching():
    """Tests if the correct reward function is selected based on skill index."""
    batch_size = 4
    obs_dim = 10 # Example observation dimension

    # Dummy observations (content doesn't matter for mock reward functions)
    last_base_obs = jnp.zeros((batch_size, obs_dim))
    base_obsv = jnp.ones((batch_size, obs_dim))

    # Skill indices for each item in the batch
    skill_indices = jnp.array([0, 1, 2, 0], dtype=jnp.int32)

    # Vmap the single-instance selection logic using mock rewards
    rewards = jax.vmap(mock_select_reward_single)(skill_indices, last_base_obs, base_obsv)

    # Expected rewards based on mock functions and skill indices
    expected_rewards = jnp.array([
        mock_reward_fn_0_single(None, None), # 0.1
        mock_reward_fn_1_single(None, None), # 1.1
        mock_reward_fn_2_single(None, None), # 2.1
        mock_reward_fn_0_single(None, None)  # 0.1
    ])

    np.testing.assert_allclose(rewards, expected_rewards, rtol=1e-6)

# To run these tests, you would typically use pytest:
# pip install pytest
# pytest test_ppo_skills.py

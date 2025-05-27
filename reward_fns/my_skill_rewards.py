import jax
import jax.numpy as jnp
from craftax.craftax_classic.constants import *

NUM_BLOCK_TYPES = len(BlockType)     # Number of block types (e.g., len(BlockType))
NUM_MOB_TYPES = 4        # Number of mob types (zombie, cow, skeleton, arrow)

all_map_flat_size = OBS_DIM[0] * OBS_DIM[1] * (NUM_BLOCK_TYPES + NUM_MOB_TYPES)
inventory_size = 12
intrinsics_size = 4 # health, food, drink, energy
direction_size = 4
misc_size = 2 # light_level, is_sleeping

inventory_start_idx = all_map_flat_size
intrinsics_start_idx = inventory_start_idx + inventory_size
direction_start_idx = intrinsics_start_idx + intrinsics_size
misc_start_idx = direction_start_idx + direction_size

health_idx = intrinsics_start_idx + 0
food_idx = intrinsics_start_idx + 1
drink_idx = intrinsics_start_idx + 2
energy_idx = intrinsics_start_idx + 3
light_level_idx = misc_start_idx + 0
is_sleeping_idx = misc_start_idx + 1

# Define indices for relevant raw materials within the inventory part of the observation vector.
# These correspond to the order in the `render_craftax_symbolic` function's inventory array:
# [wood, stone, coal, iron, diamond, sapling, wood_pickaxe, stone_pickaxe, ...]
WOOD_IDX = 0
STONE_IDX = 1
COAL_IDX = 2
IRON_IDX = 3
DIAMOND_IDX = 4
SAPLING_IDX = 5
WOOD_PICKAXE_IDX = 6
STONE_PICKAXE_IDX = 7
IRON_PICKAXE_IDX = 8
WOOD_SWORD_IDX = 9
STONE_SWORD_IDX = 10
IRON_SWORD_IDX = 11

HARVESTING_REWARD_WEIGHTS = jnp.array([
    1,  # Wood
    5.0,  # Stone
    10.0,  # Coal
    10.0,  # Iron
    50.0, # Diamond
    0.1,  # Sapling
], dtype=jnp.float32)

CRAFTED_ITEM_INDICES = jnp.array([
    6,  # wood_pickaxe
    7,  # stone_pickaxe
    8,  # iron_pickaxe
    9,  # wood_sword
    10, # stone_sword
    11, # iron_sword
])

# Weights applied to the reward for crafting each corresponding item.
# Higher weights encourage crafting more advanced items.
# Order must match CRAFTED_ITEM_INDICES:
# [wood_p, stone_p, iron_p, wood_s, stone_s, iron_s]
CRAFTING_REWARD_WEIGHTS = jnp.array([
    1,  # Reward for wood_pickaxe
    5.0,  # Reward for stone_pickaxe
    10.0,  # Reward for iron_pickaxe
    1,  # Reward for wood_sword
    5.0,  # Reward for stone_sword
    10.0,  # Reward for iron_sword
])

def get_inventory_slice(obs: jnp.ndarray) -> jnp.ndarray:
    """
    Extracts the inventory part from the flattened observation vector.

    Assumes the inventory is a fixed-size slice located before the final
    'intrinsics', 'direction', 'light_level', and 'is_sleeping' elements.

    Args:
        obs: The flattened observation array for a single time step.

    Returns:
        The inventory slice of the observation array.
    """
    # The observation structure concatenates:
    # [flat_map, inventory, intrinsics, direction, light_level, is_sleeping]
    # Sizes:
    # inventory: 12 (wood, stone, coal, iron, diamond, sapling, 3 pickaxes, 3 swords)
    # intrinsics: 4 (health, food, drink, energy)
    # direction: 4 (one-hot encoded)
    # light_level: 1
    # is_sleeping: 1
    # Total elements after inventory = 4 + 4 + 1 + 1 = 10
    inventory_size = 12
    num_trailing_elements = 10

    # Calculate start and end indices dynamically based on the total length
    # Works correctly even if the map size changes, as long as the
    # inventory and trailing elements remain constant.
    start_idx = obs.shape[-1] - num_trailing_elements - inventory_size
    end_idx = obs.shape[-1] - num_trailing_elements

    # Ensure indices are valid (useful for debugging)
    # assert start_idx >= 0, f"Calculated start index {start_idx} is invalid."
    # assert end_idx <= obs.shape[-1], f"Calculated end index {end_idx} is invalid."
    # assert (end_idx - start_idx) == inventory_size, f"Slice size {(end_idx - start_idx)} != {inventory_size}"

    return obs[..., start_idx:end_idx]

@jax.jit
def my_harvesting_reward_fn(prev_obs: jnp.ndarray, current_obs: jnp.ndarray) -> jnp.ndarray:
    """
    Calculates a reward signal focused on harvesting raw materials in Craftax.

    The reward is based on the positive change in the quantity of specific raw
    materials (wood, stone, coal, iron, diamond, sapling) in the agent's
    inventory between the previous and current observation.

    Args:
        prev_obs: The flattened observation array from the previous time step.
        current_obs: The flattened observation array from the current time step.

    Returns:
        A scalar JAX array representing the reward for the current step.
    """
    # Extract inventory slices from both observations
    # Note: The observation stores inventory counts divided by 10.0,
    # so we multiply by 10.0 to get the actual counts.
    prev_inventory = get_inventory_slice(prev_obs) * 10.0
    current_inventory = get_inventory_slice(current_obs) * 10.0

    # Select only the raw materials we care about for this skill
    prev_raw_materials = prev_inventory[..., :len(HARVESTING_REWARD_WEIGHTS)] # Select first 6 items
    current_raw_materials = current_inventory[..., :len(HARVESTING_REWARD_WEIGHTS)] # Select first 6 items

    # Calculate the change in counts for each raw material
    delta_materials = current_raw_materials - prev_raw_materials

    # Only reward positive changes (i.e., increases in resources)
    # This prevents rewarding the agent for dropping items or using them in crafting (if that were possible here).
    positive_delta_materials = jnp.maximum(0.0, delta_materials)

    # Calculate the weighted reward
    # Multiply the positive change of each resource by its corresponding weight
    weighted_reward = positive_delta_materials * HARVESTING_REWARD_WEIGHTS

    # Sum the weighted rewards for all resources to get the final scalar reward
    total_reward = jnp.sum(weighted_reward, axis=-1)

    # Ensure the output is a scalar float32 array
    return jnp.array(total_reward, dtype=jnp.float32)

@jax.jit
def my_crafting_reward_fn(prev_obs_flat: jnp.ndarray, current_obs_flat: jnp.ndarray) -> jnp.float32:
    """
    Calculates a reward signal focused on the crafting skill in Craftax.

    This function rewards the agent based on the *increase* in the quantity
    of specific crafted items (tools and swords) in its inventory between
    two consecutive timesteps.

    Args:
        prev_obs_flat: The flattened observation vector from the previous timestep.
                       Expected shape: (total_flat_size,)
        current_obs_flat: The flattened observation vector from the current timestep.
                          Expected shape: (total_flat_size,)

    Returns:
        A scalar float32 reward value. Positive reward is given when a target
        item count increases, weighted by CRAFTING_REWARD_WEIGHTS.
    """
    # --- 1. Extract Inventory Data ---
    # Use jax.lax.dynamic_slice_in_dim for efficient slicing on accelerators.
    # Extracts the segment corresponding to the inventory from the flat observation vectors.
    prev_inventory_scaled = get_inventory_slice(prev_obs_flat) # CHANGED FROM DEFAULT LOGIC FROM GEMINI
    current_inventory_scaled = get_inventory_slice(current_obs_flat)

    
    center_x, center_y = OBS_DIM[0] // 2, OBS_DIM[1] // 2   
    crafting_table_idx = BlockType.CRAFTING_TABLE.value
    furnace_idx = BlockType.FURNACE.value
    
    map_obs = current_obs_flat[:all_map_flat_size]
    map_obs = map_obs.reshape(OBS_DIM[0], OBS_DIM[1], NUM_BLOCK_TYPES + NUM_MOB_TYPES)
    local_area = map_obs[center_x-1:center_x+2, center_y-1:center_y+2, :NUM_BLOCK_TYPES]    
    is_near_crafting_table = jnp.any(local_area[:, :, crafting_table_idx] > 0)
    is_near_furnace = jnp.any(local_area[:, :, furnace_idx] > 0)
    is_in_iron_crafting_pos = jnp.logical_and(is_near_crafting_table, is_near_furnace)

    prev_map_obs = prev_obs_flat[:all_map_flat_size]
    prev_map_obs = prev_map_obs.reshape(OBS_DIM[0], OBS_DIM[1], NUM_BLOCK_TYPES + NUM_MOB_TYPES)
    prev_local_area = prev_map_obs[center_x-1:center_x+2, center_y-1:center_y+2, :NUM_BLOCK_TYPES]
    prev_is_near_crafting_table = jnp.any(prev_local_area[:, :, crafting_table_idx] > 0)
    prev_is_near_furnace = jnp.any(prev_local_area[:, :, furnace_idx] > 0)
    prev_is_in_iron_crafting_pos = jnp.logical_and(prev_is_near_crafting_table, prev_is_near_furnace)

    # Reward for entering crafting position
    entered_crafting_pos_reward = jnp.where(
        jnp.logical_and(jnp.logical_not(prev_is_in_iron_crafting_pos), is_in_iron_crafting_pos),
        0.5,
        0.0
    )

    entered_crafting_pos_reward += jnp.where(
        jnp.logical_and(prev_is_in_iron_crafting_pos, jnp.logical_not(is_in_iron_crafting_pos)),
        -0.5,
        0.0
    )

    # --- 2. Convert to Item Counts ---
    # The inventory values in the observation are scaled (divided by 10.0).
    # Multiply by 10.0 and round to nearest integer to get actual counts.
    # Cast to int32 for count representation.
    prev_inventory_counts = jnp.round(prev_inventory_scaled * 10.0).astype(jnp.int32)
    current_inventory_counts = jnp.round(current_inventory_scaled * 10.0).astype(jnp.int32)

    # --- 3. Isolate Target Crafted Item Counts ---
    # Select only the counts of the items we want to reward (pickaxes, swords)
    # using the predefined indices.
    prev_crafted_item_counts = prev_inventory_counts[CRAFTED_ITEM_INDICES]
    current_crafted_item_counts = jnp.minimum(current_inventory_counts[CRAFTED_ITEM_INDICES], 1)  # Cap at 1

    # --- 4. Calculate Increase in Counts ---
    # Compute the difference between current and previous counts for target items.
    delta_counts = current_crafted_item_counts - prev_crafted_item_counts

    # We only care about *increases* (crafting events), so zero out any negative
    # or zero changes (e.g., item used, lost, or count unchanged).
    increase_in_counts = jnp.maximum(0, delta_counts)

    # --- 5. Apply Weights and Sum ---
    # Multiply the increase in count for each item by its corresponding weight.
    weighted_increase = increase_in_counts * CRAFTING_REWARD_WEIGHTS

    # Sum the weighted increases across all target items to get the final reward.
    reward = jnp.sum(weighted_increase) #+ entered_crafting_pos_reward - 0.01

    # --- 6. Return Reward ---
    # Ensure the reward is a float32, a common type for RL rewards.
    return reward.astype(jnp.float32)
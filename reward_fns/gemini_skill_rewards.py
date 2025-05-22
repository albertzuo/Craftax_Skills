import jax
import jax.numpy as jnp
from craftax.craftax_classic.constants import *

# ================================================================================================
# ================================================================================================
# THESE ARE FOR CRAFTAX CLASSIC
# ================================================================================================
# ================================================================================================

# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
# Gemini 2.5-Pro Reward Functions
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------

# --- Environment Constants (Replace with actual values from Craftax) ---
# Example values - ensure these match your environment configuration
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

# Define weights for each resource type. Rarer/more valuable resources get higher weights.
# These weights can be tuned based on experimental results.
REWARD_WEIGHTS = jnp.array([
    1,  # Wood
    10.0,  # Stone # TUNED
    10.0,  # Coal
    100.0,  # Iron
    200.0, # Diamond
    0.1,  # Sapling # TUNED
], dtype=jnp.float32)

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
def calculate_harvesting_reward(prev_obs: jnp.ndarray, current_obs: jnp.ndarray) -> jnp.ndarray:
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
    prev_raw_materials = prev_inventory[..., :len(REWARD_WEIGHTS)] # Select first 6 items
    current_raw_materials = current_inventory[..., :len(REWARD_WEIGHTS)] # Select first 6 items

    # Calculate the change in counts for each raw material
    delta_materials = current_raw_materials - prev_raw_materials

    # Only reward positive changes (i.e., increases in resources)
    # This prevents rewarding the agent for dropping items or using them in crafting (if that were possible here).
    positive_delta_materials = jnp.maximum(0.0, delta_materials)

    # Calculate the weighted reward
    # Multiply the positive change of each resource by its corresponding weight
    weighted_reward = positive_delta_materials * REWARD_WEIGHTS * 0.1

    # Sum the weighted rewards for all resources to get the final scalar reward
    total_reward = jnp.sum(weighted_reward, axis=-1)

    # Ensure the output is a scalar float32 array
    return jnp.array(total_reward, dtype=jnp.float32)

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
    10.0,  # Reward for stone_pickaxe
    100.0,  # Reward for iron_pickaxe
    1,  # Reward for wood_sword
    10.0,  # Reward for stone_sword
    100.0,  # Reward for iron_sword
])

@jax.jit
def crafting_reward_fn(prev_obs_flat: jnp.ndarray, current_obs_flat: jnp.ndarray) -> jnp.float32:
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
    current_crafted_item_counts = current_inventory_counts[CRAFTED_ITEM_INDICES]

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
    reward = jnp.sum(weighted_increase)

    # --- 6. Return Reward ---
    # Ensure the reward is a float32, a common type for RL rewards.
    return reward.astype(jnp.float32)

@jax.jit
def survival_reward_function(prev_obs, obs):
    """
    Calculates a reward signal focused on survival in the Craftax environment.

    Args:
        obs: The flattened observation vector for the current state
             (output of render_craftax_symbolic).
        prev_obs: The flattened observation vector for the previous state.

    Returns:
        A scalar reward value (float).
    """
    reward = 0.0

    # --- Extract Player Intrinsics (Current State) ---
    # Rescale from [0, 1] back to original values (e.g., 0-10)
    health = obs[health_idx] * 10.0
    food = obs[food_idx] * 10.0
    drink = obs[drink_idx] * 10.0
    energy = obs[energy_idx] * 10.0
    is_sleeping = obs[is_sleeping_idx] # Already 0 or 1

    # --- Extract Previous Health ---
    prev_health = prev_obs[health_idx] * 10.0

    # === Reward Components ===

    # 1. Health & Damage Penalty
    # Penalize taking damage (decrease in health)
    damage_taken = jnp.maximum(0.0, prev_health - health)
    # Scale penalty: large penalty for losing health
    reward -= damage_taken * 1.0

    # Small positive reward for having high health (encourages staying healthy)
    # Scaled to be small relative to damage penalty
    reward += (health / 10.0) * 0.01

    # Very large penalty for dying (reaching 0 health)
    # This strongly discourages terminal states for this skill.
    reward = jax.lax.cond(health <= 0.1, lambda: -10.0, lambda: reward) # Use a small threshold > 0

    # 2. Sustenance Penalties (Food, Drink, Energy)
    # Penalize being in critical states
    # Using thresholds (e.g., < 2 out of 10)
    reward -= jax.lax.cond(food < 2.0, lambda: 0.1, lambda: 0.0)   # Penalty for critical hunger
    reward -= jax.lax.cond(drink < 2.0, lambda: 0.1, lambda: 0.0)  # Penalty for critical thirst
    reward -= jax.lax.cond(energy < 1.0, lambda: 0.05, lambda: 0.0) # Penalty for critical energy

    # Small reward for high sustenance levels (encourages keeping them topped up)
    reward += (food / 10.0) * 0.005
    reward += (drink / 10.0) * 0.005
    # Energy reward is linked to sleeping below

    # 3. Energy & Sleep Reward
    # Give a small reward for being asleep, especially if energy is low.
    # This incentivizes using the sleep mechanic for recovery.
    sleep_reward_base = 0.01
    # Increase reward if sleeping when energy is low (< 5)
    sleep_reward_bonus = jax.lax.cond(energy < 5.0, lambda: 0.5, lambda: 0.0)
    reward += jax.lax.cond(is_sleeping > 0.5, lambda: sleep_reward_base + sleep_reward_bonus, lambda: 0.0)

    # 4. Mob Proximity Penalty
    # Extract the mob map from the observation
    map_start_idx = 0 # Map data is at the beginning
    all_map_flat = jax.lax.dynamic_slice(obs, (map_start_idx,), (all_map_flat_size,))
    all_map = all_map_flat.reshape((*OBS_DIM, NUM_BLOCK_TYPES + NUM_MOB_TYPES))
    mob_map = all_map[:, :, NUM_BLOCK_TYPES:] # Shape: (H, W, num_mob_types)

    # Identify hostile mobs (adjust indices if mob order changes)
    # Assuming: 0: Zombie, 1: Cow, 2: Skeleton, 3: Arrow
    hostile_mob_indices = jnp.array([0, 2]) # Zombies and Skeletons
    # Arrows (index 3) are projectiles, maybe handle separately or consider them hostile
    # hostile_mob_indices = jnp.array([0, 2, 3])

    hostile_mob_map = mob_map[:, :, hostile_mob_indices] # Shape: (H, W, num_hostile)

    # Check immediate proximity (e.g., 3x3 area around player)
    center_y, center_x = OBS_DIM[0] // 2, OBS_DIM[1] // 2
    prox_radius = 1 # Check adjacent cells (3x3)

    # Slice the 3x3 area centered on the player from the hostile mob map
    # Pad the map implicitly by ensuring slice indices are within bounds if needed,
    # but dynamic_slice handles this; ensure OBS_DIM is large enough for radius.
    proximity_area = jax.lax.dynamic_slice(
        hostile_mob_map,
        (center_y - prox_radius, center_x - prox_radius, 0),
        (2 * prox_radius + 1, 2 * prox_radius + 1, hostile_mob_indices.shape[0])
    )

    # Sum detections in the proximity area. Each detected hostile mob adds a penalty.
    # Exclude the center cell itself? Maybe not necessary, as being *on* a mob is bad.
    num_hostile_mobs_nearby = jnp.sum(proximity_area)

    # Apply penalty for each nearby hostile mob
    reward -= num_hostile_mobs_nearby * 0.2 # Tune this penalty factor

    # Optional: Consider light level?
    # light_level = obs[light_level_idx]
    # reward -= jax.lax.cond(light_level < 0.2, lambda: 0.02, lambda: 0.0) # Small penalty for darkness

    return reward
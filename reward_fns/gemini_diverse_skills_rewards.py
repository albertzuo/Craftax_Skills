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

# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------
# Gemini 2.5-Pro Reward Functions (TAKE 2)
# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------


# Additional constants previously in CraftaxEnvConstants, now global:
# Integer value representing "out of bounds" tiles in the numeric map view
BLOCKTYPE_OUT_OF_BOUNDS_VALUE = 1 # Assumes BlockType.OUT_OF_BOUNDS is an int or .value #BlockType.OUT_OF_BOUNDS = 1
# jax.debug.print("BLOCKTYPE_OUT_OF_BOUNDS_VALUE: {}", BLOCKTYPE_OUT_OF_BOUNDS_VALUE)

# A JAX array of all valid game block type integer values, EXCLUDING OutOfBounds.
# This depends on how BlockType values are defined.
# If BlockType values are contiguous and start from 0 or 1:
_all_types_temp = jnp.arange(NUM_BLOCK_TYPES) # NUM_BLOCK_TYPES should be len(BlockType)
ALL_GAME_BLOCK_TYPES_NO_OOB = _all_types_temp[_all_types_temp != BLOCKTYPE_OUT_OF_BOUNDS_VALUE]
# Or, if they are arbitrary, you'd construct this array manually from BlockType enum.
# Example: jnp.array([bt.value for bt in BlockType if bt != BlockType.OUT_OF_BOUNDS])

# Indices of raw materials in the inventory array (using the globally defined indices)
RAW_MATERIAL_INDICES = jnp.array([
    WOOD_IDX, STONE_IDX, COAL_IDX, IRON_IDX, DIAMOND_IDX, SAPLING_IDX
])
INVENTORY_SCALE_FACTOR = 10.0 # As per original observation spec

# Reward weights (these will require tuning)
W_UNIQUE_BLOCKTYPES_IN_VIEW_INCREASE = 0.5
W_NEW_ITEM_TYPE_GATHERED = 5.0
W_INVENTORY_VARIETY_INCREASE = 2.0
W_RAW_MATERIAL_STOCKPILE_LOG_INCREASE = 1.0
# --- End of Globally Defined Constants ---


# --- Observation Parsing Helper ---

def _parse_observation(flat_obs: jnp.ndarray):
    """
    Parses the flattened observation into its components using global constants.
    """
    # 1. All Map (Map View One-Hot + Mob Map)
    # The map_view_one_hot is the first part of the all_map_flat segment.
    # all_map_flat contains NUM_BLOCK_TYPES channels for block types, then NUM_MOB_TYPES channels for mobs.
    all_map_flat_segment = flat_obs[0 : all_map_flat_size]
    all_map_reshaped = all_map_flat_segment.reshape((
        OBS_DIM[0],
        OBS_DIM[1],
        NUM_BLOCK_TYPES + NUM_MOB_TYPES # Total channels in all_map
    ))
    # Extract only the block type one-hot encoding part
    map_view_one_hot = all_map_reshaped[:, :, :NUM_BLOCK_TYPES]
    # mob_map = all_map_reshaped[:, :, NUM_BLOCK_TYPES:] # Not used by this reward

    # 2. Inventory
    inventory_scaled = flat_obs[
        inventory_start_idx : inventory_start_idx + inventory_size
    ]
    inventory = inventory_scaled * INVENTORY_SCALE_FACTOR # Unscale

    return map_view_one_hot, inventory

# --- Reward Function ---

@jax.jit
def reward_broaden_horizons_stockpile(
    prev_obs_flat: jnp.ndarray,
    current_obs_flat: jnp.ndarray
) -> float:
    """
    Calculates the reward for the "Broaden Horizons & Stockpile" skill
    using globally defined constants.

    Args:
        prev_obs_flat: The flattened observation from the previous timestep.
        current_obs_flat: The flattened observation from the current timestep.

    Returns:
        A scalar reward value.
    """
    total_reward = 0.0

    # Parse observations
    prev_map_view_one_hot, prev_inventory = _parse_observation(prev_obs_flat)
    curr_map_view_one_hot, curr_inventory = _parse_observation(current_obs_flat)

    # 1. Reward for increasing unique block types visible in map view
    # Convert one-hot map view to numeric block types
    prev_map_numeric = jnp.argmax(prev_map_view_one_hot, axis=-1)
    curr_map_numeric = jnp.argmax(curr_map_view_one_hot, axis=-1)

    # Filter out "Out Of Bounds" tiles
    # prev_map_valid_tiles = prev_map_numeric[prev_map_numeric != BLOCKTYPE_OUT_OF_BOUNDS_VALUE]
    # curr_map_valid_tiles = curr_map_numeric[curr_map_numeric != BLOCKTYPE_OUT_OF_BOUNDS_VALUE]

    # Count unique block types present in view (JIT-friendly way)
    def count_unique_present(map_elements, all_possible_types):
        # Checks for each possible type if it's present in map_elements
        is_present = jax.vmap(lambda tile_type: jnp.any(map_elements == tile_type))(all_possible_types)
        return jnp.sum(is_present)
    # jax.debug.breakpoint()
    # jax.debug.print("all_game_block_types_no_oob: {}", ALL_GAME_BLOCK_TYPES_NO_OOB)
    # jax.debug.print("prev_map_valid_tiles: {}", prev_map_valid_tiles)
    # jax.debug.print("curr_map_valid_tiles: {}", curr_map_valid_tiles)
    num_unique_prev_view = count_unique_present(prev_map_numeric, ALL_GAME_BLOCK_TYPES_NO_OOB)
    num_unique_curr_view = count_unique_present(curr_map_numeric, ALL_GAME_BLOCK_TYPES_NO_OOB)

    reward_unique_blocktypes_increase = jnp.maximum(0.0, num_unique_curr_view - num_unique_prev_view)
    total_reward += W_UNIQUE_BLOCKTYPES_IN_VIEW_INCREASE * reward_unique_blocktypes_increase

    # 2. Reward for gathering a new type of item for the first time
    # (item was not in inventory before, but is now)
    first_time_gathered_mask = (curr_inventory > prev_inventory) & (prev_inventory == 0)
    reward_new_item_type = jnp.sum(first_time_gathered_mask)
    total_reward += W_NEW_ITEM_TYPE_GATHERED * reward_new_item_type

    # 3. Reward for increasing overall inventory variety (more types of items held)
    variety_prev = jnp.sum(prev_inventory > 0)
    variety_curr = jnp.sum(curr_inventory > 0)
    reward_inventory_variety_increase = jnp.maximum(0.0, variety_curr - variety_prev)
    total_reward += W_INVENTORY_VARIETY_INCREASE * reward_inventory_variety_increase

    # 4. Reward for increasing stockpile of RAW materials (log scale for diminishing returns on single items)
    prev_raw_materials = prev_inventory.take(RAW_MATERIAL_INDICES) # Use .take for JAX static array indexing
    curr_raw_materials = curr_inventory.take(RAW_MATERIAL_INDICES)

    # Using log1p to handle zero counts and provide diminishing returns
    # Only consider increases in stockpile
    delta_log_stockpile = jnp.sum(jnp.log1p(curr_raw_materials)) - jnp.sum(jnp.log1p(prev_raw_materials))
    reward_raw_material_stockpile = jnp.maximum(0.0, delta_log_stockpile) # Ensure non-negative reward
    total_reward += W_RAW_MATERIAL_STOCKPILE_LOG_INCREASE * reward_raw_material_stockpile

    return total_reward


# ------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------

def reward_execute_next_milestone_skill(
    prev_obs_flat: jnp.ndarray,
    current_obs_flat: jnp.ndarray,
) -> jnp.float32: #isn't this the extrinsic reward with some intermediate steps?
    """
    Calculates the reward for the 'Execute Next Milestone' skill.

    Assumes that constants like `inventory_start_idx`, `inventory_size`,
    and specific item indices (`WOOD_IDX`, `WOOD_PICKAXE_IDX`, etc.)
    are defined in the global scope.

    Focuses on achieving milestones (crafting key items) and avoids rewarding
    general exploration or stockpiling.
    """
    reward = jnp.array(0.0, dtype=jnp.float32)

    # --- Step 1: Extract Inventory Counts ---
    # Inventory values in observation are scaled by 10.0
    # These constants are now accessed from the global scope
    prev_inventory_obs = jax.lax.dynamic_slice(
        prev_obs_flat, (inventory_start_idx,), (inventory_size,)
    )
    current_inventory_obs = jax.lax.dynamic_slice(
        current_obs_flat, (inventory_start_idx,), (inventory_size,)
    )

    # Get actual counts
    # Ensure counts are at least 0, as observation might be float
    prev_inventory_counts = jnp.maximum(0.0, prev_inventory_obs * 10.0)
    current_inventory_counts = jnp.maximum(0.0, current_inventory_obs * 10.0)

    # --- Step 2: Define Key Item and Critical Resource Indices ---
    # These indices are now accessed from the global scope
    key_item_indices = jnp.array([
        WOOD_PICKAXE_IDX, STONE_PICKAXE_IDX, IRON_PICKAXE_IDX,
        WOOD_SWORD_IDX, STONE_SWORD_IDX, IRON_SWORD_IDX
    ])
    critical_raw_material_indices = jnp.array([
        WOOD_IDX, STONE_IDX, COAL_IDX, IRON_IDX, DIAMOND_IDX
    ])

    # --- Step 3: Reward for Crafting New Key Items (Milestone Completion) ---
    # Check if key items were previously absent and are now present
    prev_key_items_present = prev_inventory_counts[key_item_indices] > 0.5 # Use 0.5 to handle potential float inaccuracies
    current_key_items_present = current_inventory_counts[key_item_indices] > 0.5

    newly_crafted_key_items_mask = jnp.logical_and(
        jnp.logical_not(prev_key_items_present), current_key_items_present
    )
    num_new_key_items_crafted = jnp.sum(newly_crafted_key_items_mask.astype(jnp.float32))

    reward_milestone_completion = num_new_key_items_crafted * 50.0
    reward += reward_milestone_completion

    any_key_item_crafted_this_step = num_new_key_items_crafted > 0

    # --- Step 4: Small Reward for First Acquisition of Critical Raw Materials ---
    # This reward is only given if no key item was crafted in this step.
    def get_first_resource_reward():
        prev_raw_materials_present = prev_inventory_counts[critical_raw_material_indices] > 0.5
        current_raw_materials_present = current_inventory_counts[critical_raw_material_indices] > 0.5

        newly_acquired_raw_mask = jnp.logical_and(
            jnp.logical_not(prev_raw_materials_present), current_raw_materials_present
        )
        num_new_raw_material_types = jnp.sum(newly_acquired_raw_mask.astype(jnp.float32))
        return num_new_raw_material_types * 5.0

    # Only add this reward if no key item was crafted in this step
    reward_first_resource = jax.lax.cond(
        any_key_item_crafted_this_step,
        lambda: jnp.array(0.0, dtype=jnp.float32),  # If key item crafted, no reward for first resource
        get_first_resource_reward # Otherwise, calculate first resource reward
    )
    reward += reward_first_resource

    # --- Step 5: Step Penalty for Efficiency ---
    step_penalty = -0.1
    reward += step_penalty

    return reward
import jax
import jax.numpy as jnp
from enum import IntEnum # Using IntEnum for clarity and JAX compatibility

# Assume these constants are defined elsewhere, based on the environment specifics.
# These are illustrative values, adjust them based on the actual Craftax environment config.
OBS_MAP_WIDTH = 9
OBS_MAP_HEIGHT = 7
NUM_BLOCK_TYPES = 17
NUM_MOB_TYPES = 4
NUM_INVENTORY_SLOTS = 12 # wood, stone, coal, iron, diamond, sapling, wood_p, stone_p, iron_p, wood_s, stone_s, iron_s
NUM_INTRINSIC_STATS = 4  # health, food, drink, energy
NUM_DIRECTION_CLASSES = 4

# Calculate slice indices based on the assumed structure in render_craftax_symbolic
map_flat_size = OBS_MAP_HEIGHT * OBS_MAP_WIDTH * (NUM_BLOCK_TYPES + NUM_MOB_TYPES)
inventory_start_idx = map_flat_size
intrinsics_start_idx = inventory_start_idx + NUM_INVENTORY_SLOTS
direction_start_idx = intrinsics_start_idx + NUM_INTRINSIC_STATS
light_level_idx = direction_start_idx + NUM_DIRECTION_CLASSES
is_sleeping_idx = light_level_idx + 1

# Define Skill IDs using IntEnum for better readability and JAX compatibility
class SkillID(IntEnum):
    BROADEN_HORIZONS_STOCKPILE = 0
    EXECUTE_NEXT_MILESTONE = 1

# --- Inventory Indices (assuming the order from render_craftax_symbolic) ---
# Relative to the start of the inventory slice
INV_WOOD = 0
INV_STONE = 1
INV_COAL = 2
INV_IRON = 3
INV_DIAMOND = 4
INV_SAPLING = 5
INV_WOOD_PICKAXE = 6
INV_STONE_PICKAXE = 7
INV_IRON_PICKAXE = 8
INV_WOOD_SWORD = 9
INV_STONE_SWORD = 10
INV_IRON_SWORD = 11

# --- Intrinsic Stat Indices (assuming the order from render_craftax_symbolic) ---
# Relative to the start of the intrinsics slice
STAT_HEALTH = 0
STAT_FOOD = 1
STAT_DRINK = 2
STAT_ENERGY = 3

def skill_selector(obs: jnp.ndarray) -> SkillID:
    """
    Meta-policy to select a skill for the Craftax game based on environment observations,
    using jax.lax.cond for conditional logic.

    This function implements a heuristic decision tree to switch between exploration/stockpiling
    and focused milestone execution. It prioritizes survival, then follows a logical
    progression through tool tiers (wood, stone, iron pickaxes), and finally considers
    general resource levels for advanced play.

    Args:
        obs: A JAX numpy array representing the flattened environment observation.
             The structure is assumed to include local map data, mob data,
             followed by a fixed-size segment for inventory, player intrinsics,
             direction, and game state flags (light level, sleeping status).

    Returns:
        SkillID: The enum member representing the skill to be activated.
                 (SkillID.BROADEN_HORIZONS_STOCKPILE or SkillID.EXECUTE_NEXT_MILESTONE)
    """

    # --- Constants and Thresholds ---
    ACTUAL_WOOD_COST_WOOD_PICKAXE = 0.3  # e.g., 3 wood
    ACTUAL_STONE_COST_STONE_PICKAXE = 0.3 # e.g., 3 stone
    ACTUAL_WOOD_COST_PICKAXE_HANDLES = 0.2 # e.g., 2 wood for handles (sticks)
    ACTUAL_IRON_COST_IRON_PICKAXE = 0.3   # e.g., 3 iron
    ACTUAL_COAL_COST_FOR_IRON_SMELTING = 0.1 # e.g., 1 coal for smelting

    # --- MODIFIED THRESHOLDS TO MAKE SKILL 1 (ENM) MORE ACTIVE ---

    # 1. Survival threshold (slightly lower to be less dominant for BHS)
    LOW_STAT_THRESHOLD = 0.25  # Was 0.3 (Corresponds to 2.5 actual stat value)

    # 2. ENM Trigger Thresholds for crafting.
    # ENM will be triggered if resources meet this percentage of the actual cost.
    # This allows ENM to perform its "highly targeted gathering" for minor shortfalls.
    EAGERNESS_FACTOR_FOR_ENM = 0.70 # Agent needs 70% of resources to attempt ENM for a milestone

    ENM_TRIGGER_WOOD_FOR_WOOD_PICKAXE = ACTUAL_WOOD_COST_WOOD_PICKAXE * EAGERNESS_FACTOR_FOR_ENM
    ENM_TRIGGER_STONE_FOR_STONE_PICKAXE = ACTUAL_STONE_COST_STONE_PICKAXE * EAGERNESS_FACTOR_FOR_ENM
    ENM_TRIGGER_WOOD_FOR_PICKAXE_HANDLES = ACTUAL_WOOD_COST_PICKAXE_HANDLES * EAGERNESS_FACTOR_FOR_ENM # For both stone and iron pickaxes
    ENM_TRIGGER_IRON_FOR_IRON_PICKAXE = ACTUAL_IRON_COST_IRON_PICKAXE * EAGERNESS_FACTOR_FOR_ENM
    # Coal is crucial for iron; consider if it needs a higher trigger factor or if ENM is good at getting it.
    # For now, applying the same eagerness factor.
    ENM_TRIGGER_COAL_FOR_IRON_SMELTING = ACTUAL_COAL_COST_FOR_IRON_SMELTING * EAGERNESS_FACTOR_FOR_ENM


    # 3. Thresholds for triggering BHS during Post-Iron Pickaxe phase.
    # If current resources are LESS than these, BHS is triggered for general stockpiling.
    # Lowering these thresholds means ENM is preferred more often in the late game.
    LOW_GENERAL_WOOD_STOCK_BHS_TRIGGER = 0.7  # Was 1.0 (trigger BHS if wood < 7, was < 10)
    LOW_GENERAL_IRON_STOCK_BHS_TRIGGER = 0.3  # Was 0.5 (trigger BHS if iron < 3, was < 5)
    LOW_GENERAL_COAL_STOCK_BHS_TRIGGER = 0.3  # Was 0.5 (trigger BHS if coal < 3, was < 5)


    # --- Parse Observation (same as before) ---
    num_non_map_features = 22
    idx_inventory_start = obs.shape[0] - num_non_map_features
    
    REL_INV_WOOD = 0
    REL_INV_STONE = 1
    REL_INV_COAL = 2
    REL_INV_IRON = 3
    REL_INV_WOOD_PICKAXE = 6
    REL_INV_STONE_PICKAXE = 7
    REL_INV_IRON_PICKAXE = 8
    
    idx_intrinsics_start = idx_inventory_start + 12
    REL_INTR_PLAYER_HEALTH = 0
    REL_INTR_PLAYER_FOOD = 1
    REL_INTR_PLAYER_DRINK = 2

    current_wood = obs[idx_inventory_start + REL_INV_WOOD]
    current_stone = obs[idx_inventory_start + REL_INV_STONE]
    current_coal = obs[idx_inventory_start + REL_INV_COAL]
    current_iron = obs[idx_inventory_start + REL_INV_IRON]

    has_wood_pickaxe = obs[idx_inventory_start + REL_INV_WOOD_PICKAXE] > 0.0
    has_stone_pickaxe = obs[idx_inventory_start + REL_INV_STONE_PICKAXE] > 0.0
    has_iron_pickaxe = obs[idx_inventory_start + REL_INV_IRON_PICKAXE] > 0.0

    player_health = obs[idx_intrinsics_start + REL_INTR_PLAYER_HEALTH]
    player_food = obs[idx_intrinsics_start + REL_INTR_PLAYER_FOOD]
    player_drink = obs[idx_intrinsics_start + REL_INTR_PLAYER_DRINK]

    # --- Define SkillID constants for return values ---
    BHS_skill = SkillID.BROADEN_HORIZONS_STOCKPILE
    ENM_skill = SkillID.EXECUTE_NEXT_MILESTONE

    # --- Define conditions (JAX boolean arrays) using the new thresholds ---
    is_survival_critical = (player_health < LOW_STAT_THRESHOLD) | \
                           (player_food < LOW_STAT_THRESHOLD) | \
                           (player_drink < LOW_STAT_THRESHOLD)

    # Condition for attempting to craft wood pickaxe (triggers ENM)
    can_trigger_enm_for_wood_pickaxe = current_wood >= ENM_TRIGGER_WOOD_FOR_WOOD_PICKAXE
    
    # Condition for attempting to craft stone pickaxe (triggers ENM)
    can_trigger_enm_for_stone_pickaxe = (current_stone >= ENM_TRIGGER_STONE_FOR_STONE_PICKAXE) & \
                                        (current_wood >= ENM_TRIGGER_WOOD_FOR_PICKAXE_HANDLES)

    # Condition for attempting to craft iron pickaxe (triggers ENM)
    can_trigger_enm_for_iron_pickaxe = (current_iron >= ENM_TRIGGER_IRON_FOR_IRON_PICKAXE) & \
                                       (current_coal >= ENM_TRIGGER_COAL_FOR_IRON_SMELTING) & \
                                       (current_wood >= ENM_TRIGGER_WOOD_FOR_PICKAXE_HANDLES)
                             
    # Condition for needing general stockpiling (triggers BHS in post-iron phase)
    needs_general_stockpiling_cond = (current_wood < LOW_GENERAL_WOOD_STOCK_BHS_TRIGGER) | \
                                     (current_iron < LOW_GENERAL_IRON_STOCK_BHS_TRIGGER) | \
                                     (current_coal < LOW_GENERAL_COAL_STOCK_BHS_TRIGGER)

    # --- Nested conditional logic using jax.lax.cond (structure remains the same) ---

    def _post_iron_pickaxe_fn(_):
        return jax.lax.cond(needs_general_stockpiling_cond,
                            lambda __: BHS_skill,
                            lambda __: ENM_skill,
                            operand=None)

    def _iron_pickaxe_fn(_):
        return jax.lax.cond(can_trigger_enm_for_iron_pickaxe, # Use new trigger condition
                            lambda __: ENM_skill,
                            lambda __: BHS_skill,
                            operand=None)

    def _manage_iron_or_post_iron_fn(_):
        return jax.lax.cond(jnp.logical_not(has_iron_pickaxe),
                            _iron_pickaxe_fn,
                            _post_iron_pickaxe_fn,
                            operand=None)

    def _stone_pickaxe_fn(_):
        return jax.lax.cond(can_trigger_enm_for_stone_pickaxe, # Use new trigger condition
                            lambda __: ENM_skill,
                            lambda __: BHS_skill,
                            operand=None)

    def _manage_stone_or_further_fn(_):
        return jax.lax.cond(jnp.logical_not(has_stone_pickaxe),
                            _stone_pickaxe_fn,
                            _manage_iron_or_post_iron_fn,
                            operand=None)

    def _wood_pickaxe_fn(_):
        return jax.lax.cond(can_trigger_enm_for_wood_pickaxe, # Use new trigger condition
                            lambda __: ENM_skill,
                            lambda __: BHS_skill,
                            operand=None)

    def _main_milestone_logic_fn(_):
        return jax.lax.cond(jnp.logical_not(has_wood_pickaxe),
                            _wood_pickaxe_fn,
                            _manage_stone_or_further_fn,
                            operand=None)

    selected_skill = jax.lax.cond(
        is_survival_critical,
        lambda _: BHS_skill,
        _main_milestone_logic_fn,
        operand=None
    )
    
    return selected_skill

def skill_selector_v2(obs: jnp.ndarray) -> SkillID:
    """
    Revised meta-policy using a discovery-driven and preparedness-based approach,
    with Python 'not' replaced by 'jnp.logical_not()' for JAX compatibility.

    Args:
        obs: A JAX numpy array representing the flattened environment observation.

    Returns:
        SkillID: The enum member representing the skill to be activated.
    """

    # --- Define State Thresholds (scaled by /10.0 where applicable) ---
    LOW_STAT_THRESHOLD = 0.25
    MIN_WOOD_TO_ATTEMPT_WOOD_PICKAXE = 0.1
    MIN_STONE_TO_ATTEMPT_STONE_PICKAXE = 0.1
    COMFORTABLE_WOOD_POST_IRON = 0.8
    COMFORTABLE_STONE_POST_IRON = 0.5
    COMFORTABLE_IRON_POST_IRON = 0.4
    COMFORTABLE_COAL_POST_IRON = 0.3

    # --- Parse Observation ---
    num_non_map_features = 22
    idx_inventory_start = obs.shape[0] - num_non_map_features
    
    REL_INV_WOOD = 0
    REL_INV_STONE = 1
    REL_INV_COAL = 2
    REL_INV_IRON = 3
    REL_INV_DIAMOND = 4
    REL_INV_WOOD_PICKAXE = 6
    REL_INV_STONE_PICKAXE = 7
    REL_INV_IRON_PICKAXE = 8
    
    idx_intrinsics_start = idx_inventory_start + 12
    REL_INTR_PLAYER_HEALTH = 0
    REL_INTR_PLAYER_FOOD = 1
    REL_INTR_PLAYER_DRINK = 2

    current_wood = obs[idx_inventory_start + REL_INV_WOOD]
    current_stone = obs[idx_inventory_start + REL_INV_STONE]
    current_coal = obs[idx_inventory_start + REL_INV_COAL]
    current_iron = obs[idx_inventory_start + REL_INV_IRON]
    current_diamond = obs[idx_inventory_start + REL_INV_DIAMOND]

    has_wood_pickaxe = obs[idx_inventory_start + REL_INV_WOOD_PICKAXE] > 0.0
    has_stone_pickaxe = obs[idx_inventory_start + REL_INV_STONE_PICKAXE] > 0.0
    has_iron_pickaxe = obs[idx_inventory_start + REL_INV_IRON_PICKAXE] > 0.0

    player_health = obs[idx_intrinsics_start + REL_INTR_PLAYER_HEALTH]
    player_food = obs[idx_intrinsics_start + REL_INTR_PLAYER_FOOD]
    player_drink = obs[idx_intrinsics_start + REL_INTR_PLAYER_DRINK]

    # --- Define SkillID constants for return values ---
    BHS_skill = SkillID.BROADEN_HORIZONS_STOCKPILE
    ENM_skill = SkillID.EXECUTE_NEXT_MILESTONE

    # --- Define Predicates for ENM Activation ---
    is_survival_critical = (player_health < LOW_STAT_THRESHOLD) | \
                           (player_food < LOW_STAT_THRESHOLD) | \
                           (player_drink < LOW_STAT_THRESHOLD)

    trigger_enm_for_wood_pickaxe = current_wood >= MIN_WOOD_TO_ATTEMPT_WOOD_PICKAXE
    trigger_enm_for_stone_pickaxe = current_stone >= MIN_STONE_TO_ATTEMPT_STONE_PICKAXE
    trigger_enm_for_iron_pickaxe = (current_iron > 0.0) & (current_coal > 0.0)
                             
    has_discovered_diamond = current_diamond > 0.0
    resources_are_comfortable = (current_wood >= COMFORTABLE_WOOD_POST_IRON) & \
                                (current_stone >= COMFORTABLE_STONE_POST_IRON) & \
                                (current_iron >= COMFORTABLE_IRON_POST_IRON) & \
                                (current_coal >= COMFORTABLE_COAL_POST_IRON)
    trigger_enm_post_iron = has_discovered_diamond | resources_are_comfortable

    # --- Nested conditional logic using jax.lax.cond ---

    def _post_iron_pickaxe_fn(_):
        return jax.lax.cond(trigger_enm_post_iron,
                            lambda __: ENM_skill,
                            lambda __: BHS_skill,
                            operand=None)

    def _iron_pickaxe_fn(_):
        return jax.lax.cond(trigger_enm_for_iron_pickaxe,
                            lambda __: ENM_skill,
                            lambda __: BHS_skill,
                            operand=None)

    def _manage_iron_or_post_iron_fn(_):
        # Use jnp.logical_not for negation
        return jax.lax.cond(jnp.logical_not(has_iron_pickaxe),
                            _iron_pickaxe_fn,
                            _post_iron_pickaxe_fn,
                            operand=None)

    def _stone_pickaxe_fn(_):
        return jax.lax.cond(trigger_enm_for_stone_pickaxe,
                            lambda __: ENM_skill,
                            lambda __: BHS_skill,
                            operand=None)

    def _manage_stone_or_further_fn(_):
        # Use jnp.logical_not for negation
        return jax.lax.cond(jnp.logical_not(has_stone_pickaxe),
                            _stone_pickaxe_fn,
                            _manage_iron_or_post_iron_fn,
                            operand=None)

    def _wood_pickaxe_fn(_):
        return jax.lax.cond(trigger_enm_for_wood_pickaxe,
                            lambda __: ENM_skill,
                            lambda __: BHS_skill,
                            operand=None)

    def _main_milestone_logic_fn(_):
        # Use jnp.logical_not for negation
        return jax.lax.cond(jnp.logical_not(has_wood_pickaxe),
                            _wood_pickaxe_fn,
                            _manage_stone_or_further_fn,
                            operand=None)

    # Top-level decision
    selected_skill = jax.lax.cond(
        is_survival_critical,
        lambda _: BHS_skill,
        _main_milestone_logic_fn,
        operand=None
    )
    
    return selected_skill
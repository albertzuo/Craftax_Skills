import jax
import jax.numpy as jnp
from enum import IntEnum
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

# Define Skill IDs using IntEnum for better readability and JAX compatibility
class SkillID(IntEnum):
    CAUTIOUS = 0
    DRIVEN = 1
    PLAYFUL = 2

def skill_selector(obs: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.int32:
    """
    Selects a skill based on the current observation and game progression.

    Args:
        obs: The flattened environment observation vector.
        key: A JAX pseudo-random number generator key for stochastic choices.

    Returns:
        An integer representing the SkillID (0: CAUTIOUS, 1: DRIVEN, 2: PLAYFUL).
    """

    # --- Extract Key Information from Observation ---
    # Vitals (obs values are scaled by 10, so 1.0 is max health/food/etc.)
    current_health = obs[health_idx]
    current_food = obs[food_idx]
    current_drink = obs[drink_idx]
    current_energy = obs[energy_idx]

    # Tool Possession (obs values are scaled by 10, so > 0 means has at least 1)
    has_wood_pickaxe = obs[WOOD_PICKAXE_IDX] > 0.05 # Greater than 0.05 (i.e. at least 0.1 which is 1 pickaxe)
    has_wood_sword = obs[WOOD_SWORD_IDX] > 0.05
    has_stone_pickaxe = obs[STONE_PICKAXE_IDX] > 0.05
    has_stone_sword = obs[STONE_SWORD_IDX] > 0.05
    has_iron_pickaxe = obs[IRON_PICKAXE_IDX] > 0.05
    has_iron_sword = obs[IRON_SWORD_IDX] > 0.05
    has_diamond_resource = obs[DIAMOND_IDX] > 0.05 # Has collected raw diamond

    has_any_wood_tools = jnp.logical_or(has_wood_pickaxe, has_wood_sword)
    has_any_stone_tools = jnp.logical_or(has_stone_pickaxe, has_stone_sword)
    has_any_iron_tools = jnp.logical_or(has_iron_pickaxe, has_iron_sword)

    # --- Critical State Overrides ---
    # If health, food, or drink are critically low, prioritize Cautious skill.
    # Thresholds are based on the 0-1 scaled values.
    is_health_critical = current_health < 0.3  # Health below 30%
    is_food_critical = current_food < 0.2      # Food below 20%
    is_drink_critical = current_drink < 0.2    # Drink below 20%
    is_in_critical_state = jnp.logical_or(is_health_critical, 
                                      jnp.logical_or(is_food_critical, is_drink_critical))

    # --- Phase-based Skill Determination (if not in critical state) ---
    def determine_skill_non_critical(obs_nc, key_nc):
        # Re-extracting from obs_nc for clarity within this function's scope if preferred,
        # or rely on captures from the outer scope. For JAX, captured variables are fine.
        # Using outer scope variables: current_health, current_energy,
        # has_any_wood_tools, has_any_stone_tools, has_any_iron_tools, has_diamond_resource

        # --- Define Probabilities for (CAUTIOUS, DRIVEN, PLAYFUL) for each phase ---
        # These are baseline probabilities, specific conditions can further adjust them.
        
        # Phase 1: Early Game (No stone tools yet)
        # Goal: Survival, get basic wood/stone tools.
        # Base: Cautious=0.3, Driven=0.5, Playful=0.2
        prob_early = jnp.array([0.3, 0.5, 0.2], dtype=jnp.float32)
        # If health is somewhat low (e.g., < 60% but not critical), increase Cautious.
        prob_early = jax.lax.select(current_health < 0.6, 
                                    jnp.array([0.6, 0.3, 0.1], dtype=jnp.float32), prob_early)
        # If no wood tools yet and health is good (>70%), slightly more playful/driven to find wood.
        prob_early = jax.lax.select(jnp.logical_and(jnp.logical_not(has_any_wood_tools), current_health >= 0.7),
                                    jnp.array([0.2, 0.4, 0.4], dtype=jnp.float32), prob_early)

        # Phase 2: Mid Game (Has stone tools, but not iron tools)
        # Goal: Accumulate resources, get iron tools.
        # Base: Cautious=0.25, Driven=0.55, Playful=0.2
        prob_mid = jnp.array([0.25, 0.55, 0.2], dtype=jnp.float32)
        # If health is somewhat low (e.g., < 60% but not critical), increase Cautious.
        prob_mid = jax.lax.select(current_health < 0.6, 
                                  jnp.array([0.5, 0.4, 0.1], dtype=jnp.float32), prob_mid)
        # If energy is low (<40%) but health okay, might lean cautious (rest) or playful (less demanding).
        prob_mid = jax.lax.select(jnp.logical_and(current_energy < 0.4, current_health >= 0.6),
                                  jnp.array([0.4, 0.2, 0.4], dtype=jnp.float32), prob_mid)

        # Phase 3: Late Game (Has iron tools or has found diamond)
        # Goal: Advanced objectives, exploration, sustained play.
        # Base: Cautious=0.25, Driven=0.35, Playful=0.4
        prob_late = jnp.array([0.25, 0.35, 0.4], dtype=jnp.float32)
        # If health is somewhat low (e.g., < 60% but not critical), increase Cautious.
        prob_late = jax.lax.select(current_health < 0.6, 
                                   jnp.array([0.5, 0.3, 0.2], dtype=jnp.float32), prob_late)
        # If has iron tools but no diamond yet, slightly more Driven to find diamond.
        prob_late = jax.lax.select(jnp.logical_and(has_any_iron_tools, jnp.logical_not(has_diamond_resource)),
                                   jnp.array([0.2, 0.5, 0.3], dtype=jnp.float32), prob_late)
        # If has diamond and good health, strongly Playful.
        prob_late = jax.lax.select(jnp.logical_and(has_diamond_resource, current_health >= 0.7),
                                   jnp.array([0.15, 0.25, 0.6], dtype=jnp.float32), prob_late)


        # --- Select Probabilities Based on Current Game Phase ---
        # Default to early game probabilities.
        current_probs = prob_early
        # If in mid_game phase (has stone tools, but not iron tools).
        is_mid_game = jnp.logical_and(has_any_stone_tools, jnp.logical_not(has_any_iron_tools))
        current_probs = jax.lax.select(is_mid_game, prob_mid, current_probs)
        # If in late_game phase (has iron tools or has found diamond).
        is_late_game = jnp.logical_or(has_any_iron_tools, has_diamond_resource)
        current_probs = jax.lax.select(is_late_game, prob_late, current_probs)
        
        # Choose a skill index (0, 1, or 2) based on the selected probabilities.
        # These indices correspond to SkillID.CAUTIOUS, SkillID.DRIVEN, SkillID.PLAYFUL.
        skill_indices = jnp.array([SkillID.CAUTIOUS.value, SkillID.DRIVEN.value, SkillID.PLAYFUL.value])
        chosen_skill_idx = jax.random.choice(key_nc, skill_indices, p=current_probs)
        
        return chosen_skill_idx

    # --- Main Conditional Logic: Critical Override or Phase-Based Choice ---
    # If in a critical state, return CAUTIOUS. Otherwise, use the phase-based probabilistic choice.
    # The lambda for the false_fun (determine_skill_non_critical) needs an operand.
    # We pass `(obs, key)` as the operand, which it can use.
    selected_skill_idx = jax.lax.cond(
        is_in_critical_state,
        lambda _: SkillID.CAUTIOUS.value,  # True branch: critical state -> Cautious
        lambda op: determine_skill_non_critical(op[0], op[1]), # False branch: non-critical -> phase logic
        operand=(obs, key)  # Pass obs and key to the false branch's function
    )
    
    return selected_skill_idx.astype(jnp.int32)

MAX_DURATION_CAUTIOUS = 150  # Timesteps
MAX_DURATION_DRIVEN = 250    # Timesteps
MAX_DURATION_PLAYFUL = 200   # Timesteps

# --- Thresholds for termination ---
# Cautious
CAUTIOUS_SAFE_HEALTH = 0.85     # Health > 85%
CAUTIOUS_SAFE_FOOD = 0.7      # Food > 70%
CAUTIOUS_SAFE_DRINK = 0.7     # Drink > 70%
CAUTIOUS_RESTED_ENERGY = 0.9  # Energy > 90% after sleeping

# Driven
DRIVEN_CRITICAL_HEALTH = 0.3  # Health < 30%

# Playful
PLAYFUL_LOW_HEALTH = 0.5      # Health < 50%
PLAYFUL_LOW_FOOD = 0.35       # Food < 35%
PLAYFUL_LOW_DRINK = 0.35      # Drink < 35%
PLAYFUL_LOW_ENERGY = 0.25     # Energy < 25%


def terminate_cautious(prev_obs: jnp.ndarray, current_obs: jnp.ndarray, current_skill_duration: int) -> jnp.bool_:
    """
    Determines if the CAUTIOUS skill should terminate.
    Terminates if vitals are restored, agent is rested, or max duration is reached.
    """
    # Extract current vitals
    current_health = current_obs[health_idx]
    current_food = current_obs[food_idx]
    current_drink = current_obs[drink_idx]
    current_energy = current_obs[energy_idx]
    is_currently_sleeping = current_obs[is_sleeping_idx] > 0.5

    # Condition 1: Vitals are at safe levels
    vitals_restored = jnp.logical_and(current_health >= CAUTIOUS_SAFE_HEALTH,
                                     jnp.logical_and(current_food >= CAUTIOUS_SAFE_FOOD,
                                                     current_drink >= CAUTIOUS_SAFE_DRINK))

    # Condition 2: Agent is well-rested (e.g., after sleeping)
    # This could be if they were sleeping and now have high energy, or simply high energy and good vitals.
    well_rested = jnp.logical_and(is_currently_sleeping, current_energy >= CAUTIOUS_RESTED_ENERGY)
    well_rested_general = jnp.logical_and(current_energy >= CAUTIOUS_RESTED_ENERGY, vitals_restored) # Broader definition
    
    # Condition 3: Maximum duration for Cautious skill reached
    max_duration_reached = current_skill_duration >= MAX_DURATION_CAUTIOUS

    # Terminate if any of these conditions are met
    should_terminate = jnp.logical_or(vitals_restored, well_rested_general)
    should_terminate = jnp.logical_or(should_terminate, max_duration_reached)
    
    return should_terminate.astype(jnp.bool_)


def terminate_driven(prev_obs: jnp.ndarray, current_obs: jnp.ndarray, current_skill_duration: int) -> jnp.bool_:
    """
    Determines if the DRIVEN skill should terminate.
    Terminates if a major goal is achieved, health becomes critical, or max duration is reached.
    """
    # Extract current health and inventory changes
    current_health = current_obs[health_idx]

    # Check for newly crafted high-tier tools or acquired rare resources
    # (comparing current inventory count > 0 and previous inventory count == 0)
    # Note: obs inventory values are scaled by 10.0, so 0.1 means 1 item.
    # A simple check for existence ( > 0.05) is used.
    
    # Stone Pickaxe (assuming wood is a lower tier and not a "major" goal for termination)
    newly_got_stone_pickaxe = jnp.logical_and(current_obs[STONE_PICKAXE_IDX] > 0.05, prev_obs[STONE_PICKAXE_IDX] < 0.05)
    # Iron Pickaxe
    newly_got_iron_pickaxe = jnp.logical_and(current_obs[IRON_PICKAXE_IDX] > 0.05, prev_obs[IRON_PICKAXE_IDX] < 0.05)
    # Diamond Resource
    newly_got_diamond = jnp.logical_and(current_obs[DIAMOND_IDX] > 0.05, prev_obs[DIAMOND_IDX] < 0.05)

    # Condition 1: Major milestone achieved
    # For simplicity, we consider getting an iron pickaxe or diamond as major.
    # Getting a stone pickaxe could also be a milestone if starting from nothing.
    milestone_achieved = jnp.logical_or(newly_got_iron_pickaxe, newly_got_diamond)
    milestone_achieved_early = newly_got_stone_pickaxe # Early game milestone

    # Condition 2: Health has become critical
    health_is_critical = current_health < DRIVEN_CRITICAL_HEALTH
    
    # Condition 3: Maximum duration for Driven skill reached
    max_duration_reached = current_skill_duration >= MAX_DURATION_DRIVEN

    # Terminate if any of these conditions are met
    should_terminate = jnp.logical_or(milestone_achieved, health_is_critical)
    should_terminate = jnp.logical_or(should_terminate, milestone_achieved_early) # Add early milestone
    should_terminate = jnp.logical_or(should_terminate, max_duration_reached)
    
    return should_terminate.astype(jnp.bool_)


def terminate_playful(prev_obs: jnp.ndarray, current_obs: jnp.ndarray, current_skill_duration: int) -> jnp.bool_:
    """
    Determines if the PLAYFUL skill should terminate.
    Terminates if vitals become too low or max duration is reached.
    """
    # Extract current vitals
    current_health = current_obs[health_idx]
    current_food = current_obs[food_idx]
    current_drink = current_obs[drink_idx]
    current_energy = current_obs[energy_idx]

    # Condition 1: Vitals are too low for continued play
    health_too_low = current_health < PLAYFUL_LOW_HEALTH
    food_too_low = current_food < PLAYFUL_LOW_FOOD
    drink_too_low = current_drink < PLAYFUL_LOW_DRINK
    energy_too_low = current_energy < PLAYFUL_LOW_ENERGY
    
    vitals_too_low = jnp.logical_or(health_too_low, 
                                   jnp.logical_or(food_too_low, 
                                                  jnp.logical_or(drink_too_low, energy_too_low)))

    # Condition 2: Maximum duration for Playful skill reached
    max_duration_reached = current_skill_duration >= MAX_DURATION_PLAYFUL

    # Terminate if any of these conditions are met
    should_terminate = jnp.logical_or(vitals_too_low, max_duration_reached)
    
    return should_terminate.astype(jnp.bool_)


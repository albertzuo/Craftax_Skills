import jax
import jax.numpy as jnp
from enum import IntEnum # Using IntEnum for clarity and JAX compatibility

# Assume these constants are defined elsewhere, based on the environment specifics.
# These are illustrative values, adjust them based on the actual Craftax environment config.
OBS_MAP_WIDTH = 9
OBS_MAP_HEIGHT = 7 # HAD TO MANUALLY CHANGE 9 -> 7
NUM_BLOCK_TYPES = 17 # Example: Adjust based on actual BlockType enum size # HAD TO MANUALLY CHANGE 16 -> 17
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
    HARVEST = 0
    CRAFT = 1
    SUSTAIN = 2

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
# Swords are indices 9, 10, 11 - not used in this simple policy's crafting logic

# --- Intrinsic Stat Indices (assuming the order from render_craftax_symbolic) ---
# Relative to the start of the intrinsics slice
STAT_HEALTH = 0
STAT_FOOD = 1
STAT_DRINK = 2
STAT_ENERGY = 3
# --------------------------------------------------------------------------

def skill_selector(obs: jnp.ndarray) -> SkillID:
    """
    Selects the active skill based on the current observation.

    Priorities:
    1. Sustain: If health, food, or drink are low, or if energy is low at night.
    2. Craft: If essential tools (pickaxes) can be crafted and are not yet owned.
    3. Harvest: Default skill if Sustain or Craft are not triggered.

    Args:
        obs: The flattened environment observation tensor.

    Returns:
        The SkillID (integer enum) representing the selected skill.
    """
    # Extract relevant parts of the observation vector
    # Note: These slices assume the flattened structure from render_craftax_symbolic
    inventory = jax.lax.dynamic_slice_in_dim(
        obs, inventory_start_idx, NUM_INVENTORY_SLOTS, axis=0
    )
    intrinsics = jax.lax.dynamic_slice_in_dim(
        obs, intrinsics_start_idx, NUM_INTRINSIC_STATS, axis=0
    )
    light_level = obs[light_level_idx]
    # is_sleeping = obs[is_sleeping_idx] # Not directly used for selection logic here

    # --- 1. Check Sustain Conditions ---
    # Thresholds are based on the normalized values (scaled by 10 in obs)
    low_health_threshold = 0.3 # e.g., health < 3
    low_food_threshold = 0.3   # e.g., food < 3
    low_drink_threshold = 0.3  # e.g., drink < 3
    low_energy_threshold = 0.3 # e.g., energy < 3
    night_threshold = 0.4      # Example threshold for darkness

    is_low_health = intrinsics[STAT_HEALTH] <= low_health_threshold
    is_low_food = intrinsics[STAT_FOOD] <= low_food_threshold
    is_low_drink = intrinsics[STAT_DRINK] <= low_drink_threshold
    is_low_energy = intrinsics[STAT_ENERGY] <= low_energy_threshold
    is_night = light_level < night_threshold

    # Need to sleep if energy is low and it's dark enough
    needs_sleep = is_low_energy & is_night

    # Activate sustain if any critical need is met
    # jax.debug.print("is_low_health: {}", is_low_health)
    # jax.debug.print("inventory start idx: {}", inventory_start_idx)
    # jax.debug.print("intrinsics start idx: {}", intrinsics_start_idx)
    # jax.debug.print("light level idx: {}", light_level_idx)
    # jax.debug.breakpoint()
    sustain_needed = is_low_health | is_low_food | is_low_drink | needs_sleep

    # --- 2. Check Crafting Conditions (if Sustain is not needed) ---
    # Check inventory quantities (remember they are scaled by 10)
    # We only need > 0, so checking > 0.01 is safe for float comparison
    has_wood = inventory[INV_WOOD] > 0.01
    has_stone = inventory[INV_STONE] > 0.01
    has_iron = inventory[INV_IRON] > 0.01
    has_coal = inventory[INV_COAL] > 0.01

    has_wood_pick = inventory[INV_WOOD_PICKAXE] > 0.01
    has_stone_pick = inventory[INV_STONE_PICKAXE] > 0.01
    has_iron_pick = inventory[INV_IRON_PICKAXE] > 0.01

    has_wood_sword = inventory[INV_WOOD_SWORD] > 0.01
    has_stone_sword = inventory[INV_STONE_SWORD] > 0.01
    has_iron_sword = inventory[INV_IRON_SWORD] > 0.01

    # Define simplified conditions for crafting pickaxes
    # Assumes 1 wood for wood pickaxe, 1 wood + 1 stone for stone, 1 wood + 1 iron for iron
    # More complex recipes would require checking actual quantities (e.g., >= 0.1 for 1 item)
    can_craft_wood_pick = has_wood
    can_craft_stone_pick = has_wood & has_stone
    can_craft_iron_pick = has_wood & has_iron & has_coal 

    can_craft_wood_sword = has_wood
    can_craft_stone_sword = has_wood & has_stone
    can_craft_iron_sword = has_wood & has_iron & has_coal # Simplification: assumes wood handle needed

    # Prioritize crafting better tools if materials are available and tool isn't owned
    should_craft_iron_pick = can_craft_iron_pick & ~has_iron_pick
    should_craft_stone_pick = can_craft_stone_pick & ~has_stone_pick & ~should_craft_iron_pick # Only if not going for iron
    should_craft_wood_pick = can_craft_wood_pick & ~has_wood_pick & ~has_stone_pick & ~should_craft_iron_pick # Only if not going for stone/iron

    should_craft_iron_sword = can_craft_iron_sword & ~has_iron_sword
    should_craft_stone_sword = can_craft_stone_sword & ~has_stone_sword & ~should_craft_iron_sword
    should_craft_wood_sword = can_craft_wood_sword & ~has_wood_sword & ~has_stone_sword & ~should_craft_iron_sword

    craft_needed = should_craft_wood_pick | should_craft_stone_pick | should_craft_iron_pick | \
                   should_craft_wood_sword | should_craft_stone_sword | should_craft_iron_sword

    # --- 3. Determine Final Skill ---
    # Use jnp.where for JAX-compatible conditional logic
    # Priority: Sustain > Craft > Harvest
    selected_skill_id = jnp.where(
        sustain_needed,
        SkillID.SUSTAIN,
        jnp.where(
            craft_needed,
            SkillID.CRAFT,
            SkillID.HARVEST # Default to Harvest
        )
    )

    return selected_skill_id



# ==============================================================================================
# ==============================================================================================
# ==============================================================================================
# ==============================================================================================

def skill_selector_v2(obs: jnp.ndarray) -> IntEnum:
    """
    Selects the active skill based on the current observation. V2.

    Priorities:
    1. Sustain: If health, food, or drink are low, or if energy is low at night, or currently sleeping.
    2. Craft: If materials are sufficient to craft the *next logical upgrade* (Table -> Wood Tools -> Stone Tools -> Iron Tools).
    3. Harvest: Default skill if Sustain or Craft are not triggered.

    Args:
        obs: The flattened environment observation tensor.

    Returns:
        The SkillID (integer enum) representing the selected skill.
    """

    # --- Constants (Defined internally) ---
    # Thresholds & Recipes (Normalized values, scaled by 10 in obs)
    # Sustain
    low_health_threshold = 0.5 # Slightly higher threshold (5 health)
    low_food_threshold = 0.4   # (4 food)
    low_drink_threshold = 0.4  # (4 drink)
    low_energy_threshold = 0.3 # (3 energy)
    night_threshold = 0.4      # Threshold for darkness

    # Crafting - Material Quantities needed (Value / 10)
    # These are *examples*, adjust to actual Craftax recipes!
    WOOD_FOR_TABLE = 0.4        # 4 wood for crafting table
    WOOD_FOR_WOOD_TOOL = 0.2    # 2 wood for wood pick/sword handle/planks
    STONE_FOR_STONE_TOOL = 0.3  # 3 stone for stone pick/sword head
    IRON_FOR_IRON_TOOL = 0.3    # 3 iron for iron pick/sword head
    COAL_FOR_SMELT = 0.1        # Assume 1 coal needed near iron for smelting trigger? (Simplification)

    # --- Extract Observation Slices ---
    inventory = jax.lax.dynamic_slice_in_dim(
        obs, inventory_start_idx, NUM_INVENTORY_SLOTS, axis=0
    )
    intrinsics = jax.lax.dynamic_slice_in_dim(
        obs, intrinsics_start_idx, NUM_INTRINSIC_STATS, axis=0
    )
    light_level = obs[light_level_idx]
    is_sleeping = obs[is_sleeping_idx] > 0.5 # Check if is_sleeping flag is true

    # --- 1. Check Sustain Conditions ---
    is_low_health = intrinsics[STAT_HEALTH] < low_health_threshold
    is_low_food = intrinsics[STAT_FOOD] < low_food_threshold
    is_low_drink = intrinsics[STAT_DRINK] < low_drink_threshold
    is_low_energy = intrinsics[STAT_ENERGY] < low_energy_threshold
    is_night = light_level < night_threshold

    needs_sleep = is_low_energy & is_night
    sustain_needed = is_low_health | is_low_food | is_low_drink | needs_sleep | is_sleeping

    # --- 2. Check Crafting Conditions (if Sustain is not needed) ---
    # Check current inventory quantities & tools
    wood_count = inventory[INV_WOOD]
    stone_count = inventory[INV_STONE]
    coal_count = inventory[INV_COAL] # May not be directly used for crafting trigger here
    iron_count = inventory[INV_IRON]

    has_wood_pick = inventory[INV_WOOD_PICKAXE] > 0.01
    has_stone_pick = inventory[INV_STONE_PICKAXE] > 0.01
    has_iron_pick = inventory[INV_IRON_PICKAXE] > 0.01

    has_wood_sword = inventory[INV_WOOD_SWORD] > 0.01
    has_stone_sword = inventory[INV_STONE_SWORD] > 0.01
    has_iron_sword = inventory[INV_IRON_SWORD] > 0.01

    # Crafting Logic: Progress up the tech tree. Check highest tier first.
    # Assumes crafting table is needed for stone/iron, wood tools can be made directly.
    # This logic determines *if* crafting *should* be prioritized now.
    # The actual crafting *action* is handled by the Craft sub-policy.

    # Can we make Iron tools? (Requires Iron + Wood + (Implicitly Coal/Furnace))
    can_make_iron_pick = (wood_count >= WOOD_FOR_WOOD_TOOL) & (iron_count >= IRON_FOR_IRON_TOOL)
    can_make_iron_sword = (wood_count >= WOOD_FOR_WOOD_TOOL) & (iron_count >= IRON_FOR_IRON_TOOL) # Same ingredients, diff recipe
    should_craft_iron = (can_make_iron_pick & ~has_iron_pick) | (can_make_iron_sword & ~has_iron_sword)

    # Can we make Stone tools? (Requires Stone + Wood + (Implicitly Table))
    can_make_stone_pick = (wood_count >= WOOD_FOR_WOOD_TOOL) & (stone_count >= STONE_FOR_STONE_TOOL)
    can_make_stone_sword = (wood_count >= WOOD_FOR_WOOD_TOOL) & (stone_count >= STONE_FOR_STONE_TOOL)
    # Only consider stone if not already aiming for iron, AND missing a stone tool
    should_craft_stone = ~should_craft_iron & ( (can_make_stone_pick & ~has_stone_pick) | (can_make_stone_sword & ~has_stone_sword) )

    # Can we make Wood tools? (Requires Wood)
    can_make_wood_pick = wood_count >= WOOD_FOR_WOOD_TOOL # Simplified: assumes 2 wood sufficient
    can_make_wood_sword = wood_count >= WOOD_FOR_WOOD_TOOL
    # Only consider wood if not aiming for stone/iron, AND missing a wood tool
    should_craft_wood = ~should_craft_iron & ~should_craft_stone & ( (can_make_wood_pick & ~has_wood_pick) | (can_make_wood_sword & ~has_wood_sword) )

    # *** Added: Need for Crafting Table prerequisite? ***
    # If we don't have stone/iron tools yet, but have wood, maybe we need to make the table?
    # This is a proxy: if we have wood but no advanced tools, trigger Craft skill
    # to potentially make the table or wood tools first.
    needs_basics = wood_count >= WOOD_FOR_TABLE # Check if enough wood for a table OR first tools
    is_early_game = ~has_stone_pick & ~has_iron_pick & ~has_stone_sword & ~has_iron_sword # No advanced tools yet
    should_craft_basics = needs_basics & is_early_game & ~should_craft_wood # If we have wood for table, but not decided to craft wood tools yet

    # Combine crafting triggers
    craft_needed = should_craft_iron | should_craft_stone | should_craft_wood | should_craft_basics

    # --- 3. Determine Final Skill ---
    # Priority: Sustain > Craft > Harvest
    selected_skill_id = jnp.where(
        sustain_needed,
        SkillID.SUSTAIN,
        jnp.where(
            craft_needed,
            SkillID.CRAFT,
            SkillID.HARVEST # Default to Harvest
        )
    )

    return selected_skill_id


# ==============================================================================================
# ==============================================================================================
# ==============================================================================================
# ==============================================================================================

# Start index of the inventory in the flattened observation
_INVENTORY_START_IDX = inventory_start_idx
_NUM_INVENTORY_ITEMS = NUM_INVENTORY_SLOTS

# Indices for items within the (un-normalized) inventory array
# These match the order in render_craftax_symbolic
INV_WOOD_IDX = 0
INV_STONE_IDX = 1
INV_COAL_IDX = 2
INV_IRON_IDX = 3
INV_DIAMOND_IDX = 4
INV_SAPLING_IDX = 5
INV_WOOD_PICKAXE_IDX = 6
INV_STONE_PICKAXE_IDX = 7
INV_IRON_PICKAXE_IDX = 8
INV_WOOD_SWORD_IDX = 9
INV_STONE_SWORD_IDX = 10
INV_IRON_SWORD_IDX = 11

# Material requirements for crafting (examples, can be tuned)
# Assumes sticks are implicitly made from wood if wood is available.
# Crafting logic in Craftax usually involves a recipe like "3 wood -> 1 wood pickaxe"
# and "1 wood -> 2 planks, 2 planks -> 4 sticks, 2 sticks + 3 wood_planks -> pickaxe"
# For simplicity, we'll check for primary material and some wood for sticks.

# Pickaxes
MAT_WOOD_FOR_WOOD_PICKAXE = 3
MAT_STONE_FOR_STONE_PICKAXE = 3
MAT_IRON_FOR_IRON_PICKAXE = 3

# Swords
MAT_WOOD_FOR_WOOD_SWORD = 2
MAT_STONE_FOR_STONE_SWORD = 2
MAT_IRON_FOR_IRON_SWORD = 2

# Wood for sticks (a general proxy for secondary ingredients)
MAT_WOOD_FOR_STICKS = 1 # Need at least 1 wood to make sticks for tools/swords


# --- Skill Selector Meta-Policy ---

def skill_selector_two_skills(obs: jnp.ndarray) -> SkillID:
    """
    Manually defined meta-policy to select between HARVEST and CRAFT skills.
    This policy guides the agent to progress through the Overworld by acquiring
    essential tools and weapons in a prioritized manner.

    Args:
        obs: The flattened environment observation from render_craftax_symbolic.

    Returns:
        SkillID: The skill to be activated (HARVEST or CRAFT).
    """

    # 1. Extract and un-normalize inventory from the observation
    # inventory_normalized = jax.lax.dynamic_slice(
    #     obs,
    #     (_INVENTORY_START_IDX,),
    #     (_NUM_INVENTORY_ITEMS,)
    # )
    inventory_normalized = jax.lax.dynamic_slice_in_dim(
        obs, inventory_start_idx, NUM_INVENTORY_SLOTS, axis=0
    )
    inventory = jnp.round(inventory_normalized * 10.0).astype(jnp.int32)

    # 2. Get current counts of key materials and tools (as JAX scalar arrays)
    wood_count = inventory[INV_WOOD_IDX]
    stone_count = inventory[INV_STONE_IDX]
    iron_count = inventory[INV_IRON_IDX]

    has_wood_pickaxe = inventory[INV_WOOD_PICKAXE_IDX] > 0
    has_stone_pickaxe = inventory[INV_STONE_PICKAXE_IDX] > 0
    has_iron_pickaxe = inventory[INV_IRON_PICKAXE_IDX] > 0
    has_wood_sword = inventory[INV_WOOD_SWORD_IDX] > 0
    has_stone_sword = inventory[INV_STONE_SWORD_IDX] > 0
    has_iron_sword = inventory[INV_IRON_SWORD_IDX] > 0
    
    sufficient_wood_for_sticks = wood_count >= MAT_WOOD_FOR_STICKS

    # 3. Implement prioritized crafting/harvesting logic using jax.lax.cond
    # The logic is a chain of checks. If an item is missing, we decide to
    # CRAFT it or HARVEST for it. If it's present, we move to the next
    # item in the priority list.

    # --- Tier 6: Iron Sword (Lowest explicit priority before default HARVEST) ---
    def f6_iron_sword_logic():
        # True branch: iron sword is missing. Decide to craft or harvest for it.
        # False branch: iron sword is present. Default to HARVEST (ultimate fallback).
        return jax.lax.cond(
            ~has_iron_sword,  # If iron sword is NOT present
            lambda: jax.lax.cond(
                (iron_count >= MAT_IRON_FOR_IRON_SWORD) & sufficient_wood_for_sticks,
                lambda: SkillID.CRAFT,  # Sufficient materials: Craft Iron Sword
                lambda: SkillID.HARVEST # Insufficient materials: Harvest for Iron Sword
            ),
            lambda: SkillID.HARVEST     # Iron sword is present: Default to HARVEST
        )

    # --- Tier 5: Iron Pickaxe ---
    def f5_iron_pickaxe_logic():
        return jax.lax.cond(
            ~has_iron_pickaxe, # If iron pickaxe is NOT present
            lambda: jax.lax.cond(
                (iron_count >= MAT_IRON_FOR_IRON_PICKAXE) & sufficient_wood_for_sticks,
                lambda: SkillID.CRAFT,  # Craft Iron Pickaxe
                lambda: SkillID.HARVEST # Harvest for Iron Pickaxe
            ),
            lambda: f6_iron_sword_logic() # Iron pickaxe is present: proceed to iron sword logic
        )

    # --- Tier 4: Stone Sword ---
    def f4_stone_sword_logic():
        return jax.lax.cond(
            ~has_stone_sword, # If stone sword is NOT present
            lambda: jax.lax.cond(
                (stone_count >= MAT_STONE_FOR_STONE_SWORD) & sufficient_wood_for_sticks,
                lambda: SkillID.CRAFT,  # Craft Stone Sword
                lambda: SkillID.HARVEST # Harvest for Stone Sword
            ),
            lambda: f5_iron_pickaxe_logic() # Stone sword is present: proceed to iron pickaxe logic
        )

    # --- Tier 3: Stone Pickaxe ---
    def f3_stone_pickaxe_logic():
        return jax.lax.cond(
            ~has_stone_pickaxe, # If stone pickaxe is NOT present
            lambda: jax.lax.cond(
                (stone_count >= MAT_STONE_FOR_STONE_PICKAXE) & sufficient_wood_for_sticks,
                lambda: SkillID.CRAFT,  # Craft Stone Pickaxe
                lambda: SkillID.HARVEST # Harvest for Stone Pickaxe
            ),
            lambda: f4_stone_sword_logic() # Stone pickaxe is present: proceed to stone sword logic
        )

    # --- Tier 2: Wood Pickaxe ---
    def f2_wood_pickaxe_logic():
        return jax.lax.cond(
            ~has_wood_pickaxe, # If wood pickaxe is NOT present
            lambda: jax.lax.cond(
                wood_count >= MAT_WOOD_FOR_WOOD_PICKAXE,
                lambda: SkillID.CRAFT,  # Craft Wood Pickaxe
                lambda: SkillID.HARVEST # Harvest for Wood Pickaxe
            ),
            lambda: f3_stone_pickaxe_logic() # Wood pickaxe is present: proceed to stone pickaxe logic
        )

    # --- Tier 1: Wood Sword (Highest explicit priority) ---
    def f1_wood_sword_logic():
        return jax.lax.cond(
            ~has_wood_sword, # If wood sword is NOT present
            lambda: jax.lax.cond(
                wood_count >= MAT_WOOD_FOR_WOOD_SWORD, # Wood sword recipe assumed simpler (e.g., no separate sticks needed beyond main wood)
                lambda: SkillID.CRAFT,  # Craft Wood Sword
                lambda: SkillID.HARVEST # Harvest for Wood Sword
            ),
            lambda: f2_wood_pickaxe_logic() # Wood sword is present: proceed to wood pickaxe logic
        )

    # Start the chain of conditional logic
    return f1_wood_sword_logic()

def breakpoint_if_craft_needed(craft_needed):
    craft_needed = jnp.asarray(craft_needed, dtype=jnp.bool_)
    craft_needed = jnp.any(craft_needed)
    def true_fn(x):
        jax.debug.breakpoint()
        return x
    def false_fn(x):
        return x
    jax.lax.cond(craft_needed, true_fn, false_fn, craft_needed)

def skill_selector_my_two_skills(obs: jnp.ndarray) -> SkillID:
    """
    Selects the active skill based on the current observation.

    Priorities:
    1. Craft: If essential tools (pickaxes) can be crafted and are not yet owned.
    2. Harvest: Default skill if Craft is not triggered.

    Args:
        obs: The flattened environment observation tensor.

    Returns:
        The SkillID (integer enum) representing the selected skill.
    """
    # Extract relevant parts of the observation vector
    # Note: These slices assume the flattened structure from render_craftax_symbolic
    inventory = jax.lax.dynamic_slice_in_dim(
        obs, inventory_start_idx, NUM_INVENTORY_SLOTS, axis=0
    )
    intrinsics = jax.lax.dynamic_slice_in_dim(
        obs, intrinsics_start_idx, NUM_INTRINSIC_STATS, axis=0
    )

    # jax.debug.print("is_low_health: {}", is_low_health)
    # jax.debug.print("inventory start idx: {}", inventory_start_idx)
    # jax.debug.print("intrinsics start idx: {}", intrinsics_start_idx)
    # jax.debug.print("light level idx: {}", light_level_idx)
    # jax.debug.breakpoint()

    # --- 2. Check Crafting Conditions (if Sustain is not needed) ---
    # Check inventory quantities (remember they are scaled by 10)
    # We only need > 0, so checking > 0.01 is safe for float comparison
    has_wood = inventory[INV_WOOD] > 0.01
    has_stone = inventory[INV_STONE] > 0.01
    has_iron = inventory[INV_IRON] > 0.01
    has_coal = inventory[INV_COAL] > 0.01
    
    has_wood_pick = inventory[INV_WOOD_PICKAXE] > 0.01
    has_stone_pick = inventory[INV_STONE_PICKAXE] > 0.01
    has_iron_pick = inventory[INV_IRON_PICKAXE] > 0.01

    has_wood_sword = inventory[INV_WOOD_SWORD] > 0.01
    has_stone_sword = inventory[INV_STONE_SWORD] > 0.01
    has_iron_sword = inventory[INV_IRON_SWORD] > 0.01

    # Define simplified conditions for crafting pickaxes
    # Assumes 1 wood for wood pickaxe, 1 wood + 1 stone for stone, 1 wood + 1 iron for iron
    # More complex recipes would require checking actual quantities (e.g., >= 0.1 for 1 item)
    can_craft_wood_pick = has_wood
    can_craft_stone_pick = has_wood & has_stone
    can_craft_iron_pick = has_wood & has_iron & has_coal 

    can_craft_wood_sword = has_wood
    can_craft_stone_sword = has_wood & has_stone
    can_craft_iron_sword = has_wood & has_iron & has_coal # Simplification: assumes wood handle needed

    # Prioritize crafting better tools if materials are available and tool isn't owned
    should_craft_iron_pick = can_craft_iron_pick & ~has_iron_pick
    should_craft_stone_pick = can_craft_stone_pick & ~has_stone_pick & ~should_craft_iron_pick # Only if not going for iron
    should_craft_wood_pick = can_craft_wood_pick & ~has_wood_pick & ~has_stone_pick & ~should_craft_iron_pick # Only if not going for stone/iron

    should_craft_iron_sword = can_craft_iron_sword & ~has_iron_sword
    should_craft_stone_sword = can_craft_stone_sword & ~has_stone_sword & ~should_craft_iron_sword
    should_craft_wood_sword = can_craft_wood_sword & ~has_wood_sword & ~has_stone_sword & ~should_craft_iron_sword

    craft_needed = should_craft_wood_pick | should_craft_stone_pick | should_craft_iron_pick | \
                   should_craft_wood_sword | should_craft_stone_sword | should_craft_iron_sword

    # --- 3. Determine Final Skill ---
    # Use jnp.where for JAX-compatible conditional logic
    # Priority: Craft > Harvest
    # breakpoint_if_craft_needed(craft_needed)
    # jax.lax.cond(jnp.any(craft_needed), lambda: jax.debug.print("inventory: {}", inventory), lambda: None)
    # jax.debug.print("is_crafting:    {craft},   inv:    {inv}", craft=craft_needed, inv=inventory)
    # jax.debug.breakpoint()
    selected_skill_id = jnp.where(
        craft_needed,
        SkillID.CRAFT,
        SkillID.HARVEST
    )

    return selected_skill_id

def terminate_harvest(prev_obs: jnp.ndarray, current_obs: jnp.ndarray, current_skill_duration: int) -> jnp.bool_:
    """
    Determines if the HARVEST skill should terminate.
    Terminates if a major goal is achieved, health becomes critical, or max duration is reached.
    """
    # Terminate if max duration for harvesting is reached
    max_duration_reached = current_skill_duration >= 1
    return max_duration_reached

def terminate_craft(prev_obs: jnp.ndarray, current_obs: jnp.ndarray, current_skill_duration: int) -> jnp.bool_:
    """
    Determines if the CRAFT skill should terminate.
    Terminates if max duration is reached AND either:
    1. A new tool has been crafted (comparing inventory states), or
    2. No new tools can be crafted (all tools are present)
    """
    # Extract inventory slices from both observations
    prev_inventory = jax.lax.dynamic_slice_in_dim(
        prev_obs, inventory_start_idx, NUM_INVENTORY_SLOTS, axis=0
    )
    current_inventory = jax.lax.dynamic_slice_in_dim(
        current_obs, inventory_start_idx, NUM_INVENTORY_SLOTS, axis=0
    )

    # Check for changes in tools (pickaxes and swords)
    tools_indices = [INV_WOOD_PICKAXE, INV_STONE_PICKAXE, INV_IRON_PICKAXE,
                    INV_WOOD_SWORD, INV_STONE_SWORD, INV_IRON_SWORD]
    
    # Check if any tool was newly crafted (went from 0 to >0)
    new_tool_crafted = jnp.any(
        jnp.array([
            (prev_inventory[idx] <= 0.01) & (current_inventory[idx] > 0.01)
            for idx in tools_indices
        ])
    )

    # Check if all tools are present
    all_tools_present = jnp.all(
        jnp.array([
            current_inventory[idx] > 0.01
            for idx in tools_indices
        ])
    )

    # Maximum duration reached
    max_duration_reached = current_skill_duration >= 1

    # Terminate if max duration reached AND either a new tool was crafted or all tools are present
    should_terminate = max_duration_reached & (new_tool_crafted | all_tools_present)
    
    return should_terminate.astype(jnp.bool_)

def terminate_sustain(prev_obs: jnp.ndarray, current_obs: jnp.ndarray, current_skill_duration: int) -> jnp.bool_:
    """
    Determines if the SUSTAIN skill should terminate.
    Terminates if a major goal is achieved, health becomes critical, or max duration is reached.
    """

    health_idx = intrinsics_start_idx + 0
    food_idx = intrinsics_start_idx + 1
    drink_idx = intrinsics_start_idx + 2
    energy_idx = intrinsics_start_idx + 3
    health_safe = current_obs[health_idx] >= 0.7  # 70% of max health
    food_safe = current_obs[food_idx] >= 0.6      # 60% of max food
    drink_safe = current_obs[drink_idx] >= 0.6    # 60% of max drink
    energy_safe = current_obs[energy_idx] >= 0.6  # 60% of max energy

    # All stats are safe
    stats_safe = jnp.logical_and(
        health_safe,
        jnp.logical_and(
            food_safe,
            jnp.logical_and(drink_safe, energy_safe)
        )
    )

    # Maximum duration reached
    max_duration_reached = current_skill_duration >= 1

    # Terminate if both conditions are met
    should_terminate = jnp.logical_and(max_duration_reached, stats_safe)
    
    return should_terminate.astype(jnp.bool_)
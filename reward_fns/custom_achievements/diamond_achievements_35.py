import jax
import jax.numpy as jnp
from flax import struct

# -----------------------------
# Custom achievements (35 total)
# -----------------------------
# Index legend:
#  0  wood>=1            1  max wood>=5          2  max wood>=10
#  3  craft wood pick    4  craft wood sword
#  5  stone>=1           6  max stone>=10        7  max stone>=20
#  8  craft stone pick   9  craft stone sword
# 10  coal>=1           11  max coal>=5         12  max coal>=10
# 13  iron>=1           14  max iron>=3         15  max iron>=5
# 16  craft iron pick   17  craft iron sword
# 18  cave light<0.30   19  deep cave<0.15      20  very deep<0.08
# 21  drink +2          22  food +2             23  heal +1
# 24  sleep once
# 25  get sapling>=1    26  plant a sapling     27  grown plant age>=3
# 28  gain iron this step while holding stone pick
# 29  have iron pick & deep cave (<0.15)
# 30  diamond>=1        31  max diamond>=2      32  max diamond>=3
# 33  quick diamond (gain diamond and timestep<=2000)
# 34  have stone AND iron pickaxes (completed pickaxe ladder)

# TODO: Number of custom achievements
NUM_CUSTOM_ACHIEVEMENTS = 35


@struct.dataclass
class CustomAchievementTracker:
    achievements: jnp.ndarray  # (NUM_CUSTOM_ACHIEVEMENTS,) bool
    # Track resource maxima so crafting doesn't "undo" progress when inventory is spent
    max_wood: jnp.int32 = jnp.int32(0)
    max_stone: jnp.int32 = jnp.int32(0)
    max_coal: jnp.int32 = jnp.int32(0)
    max_iron: jnp.int32 = jnp.int32(0)
    max_diamond: jnp.int32 = jnp.int32(0)


def init_single_tracker():
    return CustomAchievementTracker(
        achievements=jnp.zeros(NUM_CUSTOM_ACHIEVEMENTS, dtype=jnp.bool_)
    )


def update_custom_achievements(prev_state, cur_state, prev_tracker, done):
    def update_achievements():
        inv_prev, inv_cur = prev_state.inventory, cur_state.inventory

        # Update maxima
        max_wood   = jnp.maximum(prev_tracker.max_wood,   jnp.int32(inv_cur.wood))
        max_stone  = jnp.maximum(prev_tracker.max_stone,  jnp.int32(inv_cur.stone))
        max_coal   = jnp.maximum(prev_tracker.max_coal,   jnp.int32(inv_cur.coal))
        max_iron   = jnp.maximum(prev_tracker.max_iron,   jnp.int32(inv_cur.iron))
        max_diamond= jnp.maximum(prev_tracker.max_diamond,jnp.int32(inv_cur.diamond))

        # Helper deltas
        d_food  = jnp.int32(cur_state.player_food  - prev_state.player_food)
        d_drink = jnp.int32(cur_state.player_drink - prev_state.player_drink)
        d_health= jnp.int32(cur_state.player_health- prev_state.player_health)
        d_iron  = jnp.int32(inv_cur.iron    - inv_prev.iron)
        d_diam  = jnp.int32(inv_cur.diamond - inv_prev.diamond)
        d_sap   = jnp.int32(inv_cur.sapling - inv_prev.sapling)

        # Growing plants stats
        prev_plants = jnp.sum(prev_state.growing_plants_mask.astype(jnp.int32))
        cur_plants  = jnp.sum(cur_state.growing_plants_mask.astype(jnp.int32))
        # Max grown age among currently planted
        masked_age = cur_state.growing_plants_age * cur_state.growing_plants_mask.astype(cur_state.growing_plants_age.dtype)
        max_plant_age = jnp.max(jnp.where(cur_state.growing_plants_mask, masked_age, 0))

        # Light thresholds (encourage cave exploration where iron/diamond spawn)
        L = jnp.asarray(cur_state.light_level)

        # Conditions (only set if not already achieved)
        already = prev_tracker.achievements

        conds = [
            # Wood phase
            (inv_cur.wood > 0),
            (max_wood >= 5),
            (max_wood >= 10),
            (inv_cur.wood_pickaxe > 0),
            (inv_cur.wood_sword   > 0),

            # Stone phase
            (inv_cur.stone > 0),
            (max_stone >= 10),
            (max_stone >= 20),
            (inv_cur.stone_pickaxe > 0),
            (inv_cur.stone_sword   > 0),

            # Coal milestones
            (inv_cur.coal > 0),
            (max_coal >= 5),
            (max_coal >= 10),

            # Iron milestones
            (inv_cur.iron > 0),
            (max_iron >= 3),
            (max_iron >= 5),
            (inv_cur.iron_pickaxe > 0),
            (inv_cur.iron_sword   > 0),

            # Exploration depth via light level
            (L < 0.30),
            (L < 0.15),
            (L < 0.08),

            # Survival micro-skills
            (d_drink >= 2),
            (d_food  >= 2),
            (d_health >= 1),
            jnp.logical_and(jnp.logical_not(prev_state.is_sleeping), cur_state.is_sleeping),

            # Plants
            (inv_cur.sapling > 0),
            jnp.logical_and(d_sap < 0, cur_plants > prev_plants),  # planted a sapling
            (max_plant_age >= 3),

            # Tighter shaping toward iron->diamond
            jnp.logical_and(d_iron > 0, inv_cur.stone_pickaxe > 0),  # mined iron this step w/ stone pick
            jnp.logical_and(inv_cur.iron_pickaxe > 0, L < 0.15),      # geared explorer

            # Diamond milestones
            (inv_cur.diamond > 0),
            (max_diamond >= 2),
            (max_diamond >= 3),
            jnp.logical_and(d_diam > 0, jnp.asarray(cur_state.timestep) <= 2000),  # quick diamond
            jnp.logical_and(inv_cur.stone_pickaxe > 0, inv_cur.iron_pickaxe > 0),  # complete pickaxe ladder
        ]

        conds = jnp.array(conds, dtype=jnp.bool_)
        new_ach = jnp.logical_or(already, jnp.logical_and(~already, conds))

        return CustomAchievementTracker(
            achievements=new_ach,
            max_wood=max_wood,
            max_stone=max_stone,
            max_coal=max_coal,
            max_iron=max_iron,
            max_diamond=max_diamond,
        )

    return jax.lax.cond(done, init_single_tracker, update_achievements)


def get_custom_achievement_reward(prev_tracker, current_tracker):
    achievement_deltas = current_tracker.achievements.astype(jnp.float32) - prev_tracker.achievements.astype(jnp.float32)
    return jnp.sum(achievement_deltas)

import jax
import jax.numpy as jnp
from flax import struct

# Number of custom achievements
NUM_CUSTOM_ACHIEVEMENTS = 25

# Indices for readability
A_TIME_50, A_TIME_100, A_TIME_200, A_TIME_300, A_TIME_400, A_TIME_600, A_TIME_800, A_TIME_1000 = range(8)
A_EAT, A_DRINK, A_SLEEP, A_ENERGY_UP, A_NO_DMG_100 = 8, 9, 10, 11, 12
A_HEAL_ONCE, A_HEAL_5 = 13, 14
A_WOOD_PX, A_STONE_PX, A_IRON_PX = 15, 16, 17
A_WOOD_SW, A_STONE_SW, A_IRON_SW = 18, 19, 20
A_WOOD_5, A_STONE_10, A_COAL_1, A_IRON_1 = 21, 22, 23, 24

@struct.dataclass
class CustomAchievementTracker:
    achievements: jnp.ndarray  # (NUM_CUSTOM_ACHIEVEMENTS,), bool
    heal_cum: jnp.int32 = jnp.int32(0)
    no_damage_streak: jnp.int32 = jnp.int32(0)

def init_single_tracker():
    return CustomAchievementTracker(
        achievements=jnp.zeros(NUM_CUSTOM_ACHIEVEMENTS, dtype=jnp.bool_),
        heal_cum=jnp.int32(0),
        no_damage_streak=jnp.int32(0),
    )

def _mark(ach, idx, cond):
    # Set achievement idx if condition holds (idempotent)
    return ach.at[idx].set(jnp.logical_or(ach[idx], cond))

def update_custom_achievements(prev_state, cur_state, prev_tracker, done):
    def update_achievements():
        tracker = prev_tracker
        ach = tracker.achievements

        # --- Survival time milestones ---
        thresholds = [50, 100, 200, 300, 400, 600, 800, 1000]
        for i, t in enumerate(thresholds):
            cond = (prev_state.timestep < t) & (cur_state.timestep >= t)
            ach = _mark(ach, i, cond)

        # --- Vitals / usage ---
        ach = _mark(ach, A_EAT,   cur_state.player_food  > prev_state.player_food)
        ach = _mark(ach, A_DRINK, cur_state.player_drink > prev_state.player_drink)
        # sleep toggled on this step
        ach = _mark(ach, A_SLEEP, jnp.logical_and(jnp.logical_not(prev_state.is_sleeping), cur_state.is_sleeping))
        # any energy gain (rest/sleep/other)
        ach = _mark(ach, A_ENERGY_UP, cur_state.player_energy > prev_state.player_energy)

        # No-damage streak (100)
        no_dmg_inc = jnp.where(cur_state.player_health >= prev_state.player_health, 1, 0)
        new_no_dmg = jnp.where(no_dmg_inc == 1, tracker.no_damage_streak + 1, 0)
        ach = _mark(ach, A_NO_DMG_100, new_no_dmg >= 100)

        # --- Healing ---
        heal_step = jnp.maximum(cur_state.player_health - prev_state.player_health, 0)
        new_heal_cum = tracker.heal_cum + heal_step
        ach = _mark(ach, A_HEAL_ONCE, heal_step > 0)
        ach = _mark(ach, A_HEAL_5, new_heal_cum >= 5)

        # --- Tools acquired (inventory counters increased) ---
        inv_prev, inv_cur = prev_state.inventory, cur_state.inventory
        ach = _mark(ach, A_WOOD_PX,  inv_cur.wood_pickaxe  > inv_prev.wood_pickaxe)
        ach = _mark(ach, A_STONE_PX, inv_cur.stone_pickaxe > inv_prev.stone_pickaxe)
        ach = _mark(ach, A_IRON_PX,  inv_cur.iron_pickaxe  > inv_prev.iron_pickaxe)

        ach = _mark(ach, A_WOOD_SW,  inv_cur.wood_sword  > inv_prev.wood_sword)
        ach = _mark(ach, A_STONE_SW, inv_cur.stone_sword > inv_prev.stone_sword)
        ach = _mark(ach, A_IRON_SW,  inv_cur.iron_sword  > inv_prev.iron_sword)

        # --- Basic resource plateaus (first time reaching) ---
        ach = _mark(ach, A_WOOD_5,   jnp.logical_and(inv_prev.wood  < 5,  inv_cur.wood  >= 5))
        ach = _mark(ach, A_STONE_10, jnp.logical_and(inv_prev.stone < 10, inv_cur.stone >= 10))
        ach = _mark(ach, A_COAL_1,   jnp.logical_and(inv_prev.coal  < 1,  inv_cur.coal  >= 1))
        ach = _mark(ach, A_IRON_1,   jnp.logical_and(inv_prev.iron  < 1,  inv_cur.iron  >= 1))

        return CustomAchievementTracker(
            achievements=ach,
            heal_cum=new_heal_cum.astype(jnp.int32),
            no_damage_streak=new_no_dmg.astype(jnp.int32),
        )

    return jax.lax.cond(done, init_single_tracker, update_achievements)

def get_custom_achievement_reward(prev_tracker, current_tracker):
    achievement_deltas = current_tracker.achievements.astype(jnp.float32) - prev_tracker.achievements.astype(jnp.float32)
    return jnp.sum(achievement_deltas)

# Time alive milestones (8): 50, 100, 200, 300, 400, 600, 800, 1000 timesteps.
# Vitals/use (5): first eat (food↑), first drink (drink↑), first sleep, first energy gain, 100-step no-damage streak.
# Healing (2): heal at least once; cumulative +5 HP recovered.
# Tools (6): obtain wood/stone/iron pickaxe; obtain wood/stone/iron sword.
# Resources (4): reach {wood ≥5, stone ≥10, coal ≥1, iron ≥1} at least once.
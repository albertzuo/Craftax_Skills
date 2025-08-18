import jax
import jax.numpy as jnp
from flax import struct

# ===============================
# Custom survival achievements
# ===============================

# Index layout for readability
ACH_FIRST_MEAL   = 0
ACH_FIRST_DRINK  = 1
ACH_TOOLSMITH    = 2
ACH_NIGHT_WATCH  = 3
ACH_THOUSAND     = 4

# TODO: Number of custom achievements
NUM_CUSTOM_ACHIEVEMENTS = 5

# Tunables
NIGHT_LIGHT_THRESHOLD = 0.40  # consider "dark" if light_level < 0.40
NIGHT_STEPS_REQUIRED  = 120   # ~40% of a 300-step day; adjust if env differs
TARGET_TIMESTEPS      = 1000  # survival goal

@struct.dataclass
class CustomAchievementTracker:
    # Shape: (NUM_CUSTOM_ACHIEVEMENTS,)
    achievements: jnp.ndarray
    # Track a darkness streak that resets on damage or when it's bright
    dark_streak: jnp.int32 = jnp.int32(0)

def init_single_tracker():
    return CustomAchievementTracker(
        achievements=jnp.zeros(NUM_CUSTOM_ACHIEVEMENTS, dtype=jnp.bool_),
        dark_streak=jnp.int32(0),
    )

def _sum_tools(inv):
    # Count any pickaxe/sword across tiers
    vals = jnp.array([
        inv.wood_pickaxe, inv.stone_pickaxe, inv.iron_pickaxe,
        inv.wood_sword,   inv.stone_sword,   inv.iron_sword
    ], dtype=jnp.int32)
    return jnp.sum(vals)

def update_custom_achievements(prev_state, cur_state, prev_tracker, done):
    def update_achievements():
        tracker = prev_tracker
        ach = tracker.achievements

        # --- 1) First Bite: food increased
        ate_now = (cur_state.player_food - prev_state.player_food) > 0
        ach = ach.at[ACH_FIRST_MEAL].set(jnp.where(ach[ACH_FIRST_MEAL], True, ate_now))

        # --- 2) First Sip: drink increased
        drank_now = (cur_state.player_drink - prev_state.player_drink) > 0
        ach = ach.at[ACH_FIRST_DRINK].set(jnp.where(ach[ACH_FIRST_DRINK], True, drank_now))

        # --- 3) Toolsmith: any pickaxe or sword for the first time
        tools_prev = _sum_tools(prev_state.inventory)
        tools_cur  = _sum_tools(cur_state.inventory)
        got_tools_now = jnp.logical_and(tools_prev == 0, tools_cur > 0)
        ach = ach.at[ACH_TOOLSMITH].set(jnp.where(ach[ACH_TOOLSMITH], True, got_tools_now))

        # --- 4) Night Watch: long dark streak without taking damage
        is_dark      = cur_state.light_level < NIGHT_LIGHT_THRESHOLD
        took_damage  = cur_state.player_health < prev_state.player_health

        # increment streak if dark and no damage; else reset
        dark_streak_next = jnp.where(
            jnp.logical_and(is_dark, jnp.logical_not(took_damage)),
            tracker.dark_streak + jnp.int32(1),
            jnp.int32(0)
        )
        night_ok_now = dark_streak_next >= jnp.int32(NIGHT_STEPS_REQUIRED)
        ach = ach.at[ACH_NIGHT_WATCH].set(jnp.where(ach[ACH_NIGHT_WATCH], True, night_ok_now))

        # If achieved, no need to keep counting
        dark_streak_next = jnp.where(ach[ACH_NIGHT_WATCH], jnp.int32(0), dark_streak_next)

        # --- 5) Thousand Club: reach target timestep
        reached_target = cur_state.timestep >= jnp.int32(TARGET_TIMESTEPS)
        ach = ach.at[ACH_THOUSAND].set(jnp.where(ach[ACH_THOUSAND], True, reached_target))

        return CustomAchievementTracker(achievements=ach, dark_streak=dark_streak_next)

    return jax.lax.cond(done, init_single_tracker, update_achievements)

def get_custom_achievement_reward(prev_tracker, current_tracker):
    achievement_deltas = current_tracker.achievements.astype(jnp.float32) - prev_tracker.achievements.astype(jnp.float32)
    return jnp.sum(achievement_deltas)

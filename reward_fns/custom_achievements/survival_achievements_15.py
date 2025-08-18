import jax
import jax.numpy as jnp
from flax import struct

# 15 survival-focused achievements
NUM_CUSTOM_ACHIEVEMENTS = 15

@struct.dataclass
class CustomAchievementTracker:
    achievements: jnp.ndarray  # shape: (NUM_CUSTOM_ACHIEVEMENTS,), bool
    # simple counters to support multi-use behaviors
    eat_events: jnp.int32
    drink_events: jnp.int32

def init_single_tracker():
    return CustomAchievementTracker(
        achievements=jnp.zeros(NUM_CUSTOM_ACHIEVEMENTS, dtype=jnp.bool_),
        eat_events=jnp.int32(0),
        drink_events=jnp.int32(0),
    )

def update_custom_achievements(prev_state, cur_state, prev_tracker, done):
    def update_achievements():
        ps, cs, tr = prev_state, cur_state, prev_tracker

        # Event detection (rising edges / thresholds)
        drank_now = cs.player_drink > ps.player_drink
        ate_now   = cs.player_food  > ps.player_food
        sleep_start = jnp.logical_and(cs.is_sleeping, jnp.logical_not(ps.is_sleeping))

        # Update counters
        new_drink_count = tr.drink_events + drank_now.astype(jnp.int32)
        new_eat_count   = tr.eat_events   + ate_now.astype(jnp.int32)

        # Inventory / crafting helpers
        inv = cs.inventory
        has_sword = (inv.wood_sword + inv.stone_sword + inv.iron_sword) > 0

        # Time milestones
        t = cs.timestep

        # 15 achievements: ordered for clarity
        conds = jnp.array([
            inv.wood >= 1,                        # 0) First wood
            inv.wood_pickaxe >= 1,               # 1) Craft wooden pickaxe
            inv.stone >= 1,                      # 2) First stone
            inv.stone_pickaxe >= 1,              # 3) Craft stone pickaxe
            has_sword,                           # 4) Craft any sword (wood/stone/iron)

            drank_now,                           # 5) First drink
            ate_now,                             # 6) First eat
            sleep_start,                         # 7) First sleep
            new_drink_count >= 3,                # 8) Drink at least 3 times total
            new_eat_count >= 3,                  # 9) Eat at least 3 times total

            t >= 100,                            # 10) Survive 100 steps
            t >= 300,                            # 11) Survive 300 (≈ first full day)
            t >= 600,                            # 12) Survive 600 (≈ two days)
            t >= 900,                            # 13) Survive 900 (≈ three days)
            t >= 1000,                           # 14) Goal: survive 1000 steps
        ], dtype=jnp.bool_)

        new_ach = jnp.logical_or(tr.achievements, conds)

        return CustomAchievementTracker(
            achievements=new_ach,
            eat_events=new_eat_count,
            drink_events=new_drink_count,
        )

    return jax.lax.cond(done, init_single_tracker, update_achievements)

def get_custom_achievement_reward(prev_tracker, current_tracker):
    achievement_deltas = current_tracker.achievements.astype(jnp.float32) - prev_tracker.achievements.astype(jnp.float32)
    return jnp.sum(achievement_deltas)

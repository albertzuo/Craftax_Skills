import jax
import jax.numpy as jnp
from flax import struct

# -------------------- Config --------------------
NUM_CUSTOM_ACHIEVEMENTS = 35
_DAY_THR = 0.5   # light_level > 0.5 => day
_SAFE3   = 3     # "safely high" for vitals/health
_OK2     = 2     # "okay" level for vitals

@struct.dataclass
class CustomAchievementTracker:
    # One-shot flags
    achievements: jnp.ndarray  # (NUM_CUSTOM_ACHIEVEMENTS,), bool
    # Streaks / counters
    safe_vitals_streak: jnp.int32 = jnp.int32(0)    # food>=3 & drink>=3 & energy>=3
    food2_streak: jnp.int32       = jnp.int32(0)    # food>=2
    drink2_streak: jnp.int32      = jnp.int32(0)    # drink>=2
    energy2_streak: jnp.int32     = jnp.int32(0)    # energy>=2
    health_ok_streak: jnp.int32   = jnp.int32(0)    # health>=3
    no_damage_streak: jnp.int32   = jnp.int32(0)    # health_t >= health_{t-1}
    night_no_dmg_streak: jnp.int32= jnp.int32(0)    # (is_night & no damage) consecutive
    day_starts: jnp.int32         = jnp.int32(0)    # night->day transitions survived
    last_is_day: jnp.bool_        = jnp.bool_(True)
    # Risk bookkeeping
    ever_critical: jnp.bool_      = jnp.bool_(False)  # saw health < 2 at any time

def init_single_tracker():
    return CustomAchievementTracker(
        achievements=jnp.zeros(NUM_CUSTOM_ACHIEVEMENTS, dtype=jnp.bool_),
        safe_vitals_streak=jnp.int32(0),
        food2_streak=jnp.int32(0),
        drink2_streak=jnp.int32(0),
        energy2_streak=jnp.int32(0),
        health_ok_streak=jnp.int32(0),
        no_damage_streak=jnp.int32(0),
        night_no_dmg_streak=jnp.int32(0),
        day_starts=jnp.int32(0),
        last_is_day=jnp.bool_(True),
        ever_critical=jnp.bool_(False),
    )

def _set_ach(ach, idx, cond):
    return ach.at[idx].set(jnp.logical_or(ach[idx], cond))

def update_custom_achievements(prev_state, cur_state, prev_tracker, done):
    def update_achievements():
        tr = prev_tracker
        ach = tr.achievements
        t  = cur_state.timestep

        # ---------- Derived signals ----------
        is_day   = cur_state.light_level > _DAY_THR
        is_night = jnp.logical_not(is_day)

        # night->day rising edge => survived a night
        day_starts = tr.day_starts + jnp.where(
            jnp.logical_and(jnp.logical_not(tr.last_is_day), is_day), 1, 0
        )

        # Vitals
        safe_vitals = jnp.logical_and(
            jnp.logical_and(cur_state.player_food  >= _SAFE3,
                            cur_state.player_drink >= _SAFE3),
            cur_state.player_energy >= _SAFE3
        )
        safe_vitals_streak = jnp.where(safe_vitals, tr.safe_vitals_streak + 1, 0)

        food2_streak   = jnp.where(cur_state.player_food   >= _OK2, tr.food2_streak   + 1, 0)
        drink2_streak  = jnp.where(cur_state.player_drink  >= _OK2, tr.drink2_streak  + 1, 0)
        energy2_streak = jnp.where(cur_state.player_energy >= _OK2, tr.energy2_streak + 1, 0)

        # Health & damage
        health_ok = cur_state.player_health >= _SAFE3
        health_ok_streak = jnp.where(health_ok, tr.health_ok_streak + 1, 0)

        no_health_loss = cur_state.player_health >= prev_state.player_health
        no_damage_streak = jnp.where(no_health_loss, tr.no_damage_streak + 1, 0)

        night_no_dmg_streak = jnp.where(
            jnp.logical_and(is_night, no_health_loss),
            tr.night_no_dmg_streak + 1,
            jnp.int32(0)  # reset during day or if damaged at night
        )

        # Critical-history tracking
        ever_critical = jnp.logical_or(tr.ever_critical, cur_state.player_health < 2)

        # Sleep & recovery events
        slept_now = jnp.logical_and(jnp.logical_not(prev_state.is_sleeping), cur_state.is_sleeping)
        healed_now = cur_state.player_health > prev_state.player_health
        healed_while_sleeping = jnp.logical_and(prev_state.is_sleeping, healed_now)
        energy_up_while_sleep = jnp.logical_and(
            prev_state.is_sleeping, cur_state.player_energy > prev_state.player_energy
        )

        # "Safe sleep" if no hostiles active when falling asleep
        any_z = jnp.any(cur_state.zombies.mask)
        any_sk = jnp.any(cur_state.skeletons.mask)
        safe_sleep_now = jnp.logical_and(slept_now, jnp.logical_not(jnp.logical_or(any_z, any_sk)))

        # ---------- Achievements: all directly promote survival ----------
        # A. Time alive milestones (10)
        ach = _set_ach(ach,  0, t >=  50)
        ach = _set_ach(ach,  1, t >= 100)
        ach = _set_ach(ach,  2, t >= 150)
        ach = _set_ach(ach,  3, t >= 200)
        ach = _set_ach(ach,  4, t >= 300)
        ach = _set_ach(ach,  5, t >= 400)
        ach = _set_ach(ach,  6, t >= 500)
        ach = _set_ach(ach,  7, t >= 650)
        ach = _set_ach(ach,  8, t >= 800)
        ach = _set_ach(ach,  9, t >=1000)

        # B. Vitals management (9)
        ach = _set_ach(ach, 10, safe_vitals_streak >=  25)
        ach = _set_ach(ach, 11, safe_vitals_streak >=  75)
        ach = _set_ach(ach, 12, safe_vitals_streak >= 150)

        ach = _set_ach(ach, 13, food2_streak   >= 100)
        ach = _set_ach(ach, 14, food2_streak   >= 200)
        ach = _set_ach(ach, 15, drink2_streak  >= 100)
        ach = _set_ach(ach, 16, drink2_streak  >= 200)
        ach = _set_ach(ach, 17, energy2_streak >= 100)
        ach = _set_ach(ach, 18, energy2_streak >= 200)

        # C. Health safety & no-damage (7)
        ach = _set_ach(ach, 19, health_ok_streak  >=  50)
        ach = _set_ach(ach, 20, health_ok_streak  >= 100)

        ach = _set_ach(ach, 21, no_damage_streak  >=  50)
        ach = _set_ach(ach, 22, no_damage_streak  >= 100)
        ach = _set_ach(ach, 23, no_damage_streak  >= 200)

        ach = _set_ach(ach, 24, healed_now)  # first time you recover health
        ach = _set_ach(ach, 25, jnp.logical_and(t >= 300, jnp.logical_not(ever_critical)))  # no critical dip in first 300 steps

        # D. Sleep & recovery (4)
        ach = _set_ach(ach, 26, slept_now)
        ach = _set_ach(ach, 27, safe_sleep_now)
        ach = _set_ach(ach, 28, healed_while_sleeping)
        ach = _set_ach(ach, 29, energy_up_while_sleep)

        # E. Night survival & discipline (5)
        ach = _set_ach(ach, 30, day_starts >= 1)  # survived first night
        ach = _set_ach(ach, 31, day_starts >= 2)
        ach = _set_ach(ach, 32, day_starts >= 3)
        ach = _set_ach(ach, 33, night_no_dmg_streak >= 30)
        ach = _set_ach(ach, 34, night_no_dmg_streak >= 60)

        # ---------- Return ----------
        return CustomAchievementTracker(
            achievements=ach,
            safe_vitals_streak=safe_vitals_streak.astype(jnp.int32),
            food2_streak=food2_streak.astype(jnp.int32),
            drink2_streak=drink2_streak.astype(jnp.int32),
            energy2_streak=energy2_streak.astype(jnp.int32),
            health_ok_streak=health_ok_streak.astype(jnp.int32),
            no_damage_streak=no_damage_streak.astype(jnp.int32),
            night_no_dmg_streak=night_no_dmg_streak.astype(jnp.int32),
            day_starts=day_starts.astype(jnp.int32),
            last_is_day=is_day,
            ever_critical=ever_critical,
        )

    return jax.lax.cond(done, init_single_tracker, update_achievements)

def get_custom_achievement_reward(prev_tracker, current_tracker):
    achievement_deltas = current_tracker.achievements.astype(jnp.float32) - prev_tracker.achievements.astype(jnp.float32)
    return jnp.sum(achievement_deltas)

import jax
import jax.numpy as jnp
from flax import struct

# =========================
# Custom Achievements (15)
# =========================
NUM_CUSTOM_ACHIEVEMENTS = 15

@struct.dataclass
class CustomAchievementTracker:
    achievements: jnp.ndarray  # (NUM_CUSTOM_ACHIEVEMENTS,) bool
    enclosed_streak: jnp.int32 = jnp.int32(0)      # consecutive steps fully sealed
    sleeping_streak: jnp.int32 = jnp.int32(0)      # consecutive steps sleeping while sealed
    placed_nearby_total: jnp.int32 = jnp.int32(0)  # cumulative local map edits around agent

def init_single_tracker():
    return CustomAchievementTracker(
        achievements=jnp.zeros(NUM_CUSTOM_ACHIEVEMENTS, dtype=jnp.bool_)
    )

def _int_pos(pos):
    # robust int cast (Craftax positions are ints but we guard anyway)
    return jnp.int32(pos[0]), jnp.int32(pos[1])

def _within_bounds(x, y, H, W):
    return (x >= 1) & (x < W - 1) & (y >= 1) & (y < H - 1)

def _neighbor_coords(x, y):
    # 8-neighborhood around (x,y): [N, S, W, E, NW, NE, SW, SE]
    xs = jnp.array([x, x, x-1, x+1, x-1, x+1, x-1, x+1], dtype=jnp.int32)
    ys = jnp.array([y-1, y+1, y,   y,   y-1, y-1, y+1, y+1], dtype=jnp.int32)
    return xs, ys

def _cardinal_coords(x, y):
    # N, W, E, S
    xs = jnp.array([x,   x-1, x+1, x  ], dtype=jnp.int32)
    ys = jnp.array([y-1, y,   y,   y+1], dtype=jnp.int32)
    return xs, ys

def _min_hostile_dist(cur_state, x, y):
    # Manhattan distance to nearest hostile (zombies + skeletons)
    px = jnp.float32(x); py = jnp.float32(y)

    def mindist(mobs):
        # mobs.position: (N,2); mobs.mask: (N,)
        pos = mobs.position
        msk = mobs.mask
        dx = jnp.abs(pos[:, 0].astype(jnp.float32) - px)
        dy = jnp.abs(pos[:, 1].astype(jnp.float32) - py)
        d  = dx + dy
        big = jnp.full_like(d, 1e6)
        d_masked = jnp.where(msk, d, big)
        return jnp.min(d_masked)

    d_z = mindist(cur_state.zombies)
    d_s = mindist(cur_state.skeletons)
    return jnp.minimum(d_z, d_s)

def update_custom_achievements(prev_state, cur_state, prev_tracker, done):
    def update_achievements():
        tracker = prev_tracker
        H, W = cur_state.map.shape[0], cur_state.map.shape[1]
        x, y = _int_pos(cur_state.player_position)
        valid_center = _within_bounds(x, y, H, W)

        # Neighborhood values
        n8_x, n8_y = _neighbor_coords(x, y)
        n4_x, n4_y = _cardinal_coords(x, y)

        # Guard out-of-bounds reads (near edges declare not sealed)
        safe_n8 = valid_center
        safe_n4 = valid_center

        player_tile = jnp.where(valid_center, cur_state.map[x, y], jnp.int32(-1))

        cur_n8_vals  = jnp.where(safe_n8, cur_state.map[n8_x, n8_y], jnp.int32(-1))
        cur_n4_vals  = jnp.where(safe_n4, cur_state.map[n4_x, n4_y], jnp.int32(-1))
        prev_n8_vals = jnp.where(safe_n8, prev_state.map[n8_x, n8_y], jnp.int32(-1))

        # Map edits near player
        edits_n8 = jnp.sum((prev_n8_vals != cur_n8_vals).astype(jnp.int32))
        placed_nearby_total = tracker.placed_nearby_total + edits_n8

        # Walls vs floor: neighbors that are different from player tile
        diff8 = (cur_n8_vals != player_tile)
        diff4 = (cur_n4_vals != player_tile)

        four_sides = safe_n4 & jnp.all(diff4)
        two_sides  = safe_n4 & (jnp.sum(diff4.astype(jnp.int32)) >= 2)
        sealed8    = safe_n8 & jnp.all(diff8)

        # Light heuristics
        light_now = cur_state.light_level
        light_prev = prev_state.light_level
        light_drop = (light_prev - light_now) >= 0.10
        low_light  = light_now <= 0.30
        indoor_light = sealed8 & (low_light | light_drop)

        # Enclosure + sleep streaks
        enclosed_streak = jnp.where(sealed8, tracker.enclosed_streak + 1, jnp.int32(0))
        sleeping_in_fort = sealed8 & cur_state.is_sleeping
        sleeping_streak  = jnp.where(sleeping_in_fort, tracker.sleeping_streak + 1, jnp.int32(0))

        # Safety, preparedness, location checks
        min_hostile_d = _min_hostile_dist(cur_state, x, y)
        safe_sleep    = sleeping_in_fort & (min_hostile_d >= 2.0)
        edge_dist = jnp.minimum(jnp.minimum(jnp.int32(x), jnp.int32(W - 1 - x)),
                                jnp.minimum(jnp.int32(y), jnp.int32(H - 1 - y)))
        cozy_location = sleeping_in_fort & (edge_dist >= 3)
        prepared = sleeping_in_fort & (cur_state.player_food >= 5) & (cur_state.player_drink >= 5)

        # Inventory shaping
        blocks_total = (cur_state.inventory.wood + cur_state.inventory.stone)

        # === Set achievements ===
        ach = tracker.achievements

        # 0 Stockpile I (>=10)
        ach = ach.at[0].set(ach[0] | (blocks_total >= 10))
        # 1 Stockpile II (>=20)
        ach = ach.at[1].set(ach[1] | (blocks_total >= 20))
        # 2 First Wall (>=1 local edit)
        ach = ach.at[2].set(ach[2] | (placed_nearby_total >= 1))
        # 3 Two Sides (>=2 cardinals differ)
        ach = ach.at[3].set(ach[3] | two_sides)
        # 4 Four Sides (all 4 cardinals differ)
        ach = ach.at[4].set(ach[4] | four_sides)
        # 5 Fully Sealed Ring (all 8 neighbors differ)
        ach = ach.at[5].set(ach[5] | sealed8)
        # 6 Indoor Light (sealed + drop or low)
        ach = ach.at[6].set(ach[6] | indoor_light)
        # 7 Hold the Fort (sealed for >=10)
        ach = ach.at[7].set(ach[7] | (enclosed_streak >= 10))
        # 8 Any Sleep
        ach = ach.at[8].set(ach[8] | (cur_state.is_sleeping == True))
        # 9 Sleep in Fort
        ach = ach.at[9].set(ach[9] | sleeping_in_fort)
        # 10 Safe Sleep (no hostiles within 2)
        ach = ach.at[10].set(ach[10] | safe_sleep)
        # 11 Nighty-Night (low light while sleeping in fort)
        ach = ach.at[11].set(ach[11] | (sleeping_in_fort & low_light))
        # 12 Cozy Location (edge distance >=3)
        ach = ach.at[12].set(ach[12] | cozy_location)
        # 13 Well-Prepared (food/drink >=5)
        ach = ach.at[13].set(ach[13] | prepared)
        # 14 Good Nap (sleeping in fort for >=5 steps)
        ach = ach.at[14].set(ach[14] | (sleeping_streak >= 5))

        tracker = tracker.replace(
            achievements=ach,
            enclosed_streak=enclosed_streak,
            sleeping_streak=sleeping_streak,
            placed_nearby_total=placed_nearby_total
        )
        return tracker

    return jax.lax.cond(done, init_single_tracker, update_achievements)

def get_custom_achievement_reward(prev_tracker, current_tracker):
    achievement_deltas = current_tracker.achievements.astype(jnp.float32) - prev_tracker.achievements.astype(jnp.float32)
    return jnp.sum(achievement_deltas)

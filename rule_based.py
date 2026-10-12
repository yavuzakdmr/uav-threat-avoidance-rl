"""
Kural tabanlı karşılaştırma yöntemi: yapay potansiyel alanı + yön P-kontrolcüsü.
Hedef çeker; tehditler (şimdiki ve kısa vadede tahmin edilen konumları) ve duvarlar iter.
Bilinen zayıflıklar: yerel minimumlar, kafa kafaya simetrik durumlar, salınım.
"""
import numpy as np
from uav_env import UAV_RADIUS, WORLD, wrap_angle


class PotentialFieldPolicy:
    def __init__(self, detect_radius=20.0, k_rep=10.0, k_wall=5.0,
                 wall_margin=6.0, lookahead=5, k_p=1.5):
        self.detect_radius = detect_radius
        self.k_rep = k_rep
        self.k_wall = k_wall
        self.wall_margin = wall_margin
        self.lookahead = lookahead
        self.k_p = k_p

    def __call__(self, env):
        e = env.unwrapped
        to_goal = e.goal - e.pos
        force = to_goal / (np.linalg.norm(to_goal) + 1e-8)  # çekici kuvvet

        for c, v, r in zip(e.threat_pos, e.threat_vel, e.threat_r):
            for center in (c, c + v * self.lookahead):         # şimdiki + tahmini konum
                d_vec = e.pos - center
                d = np.linalg.norm(d_vec) + 1e-8
                clearance = max(d - r - UAV_RADIUS, 0.5)
                if clearance < self.detect_radius:
                    force = force + self.k_rep * (1 / clearance - 1 / self.detect_radius) * d_vec / d

        for k in range(2):                                      # duvarlardan itme
            lo, hi = e.pos[k], WORLD - e.pos[k]
            if lo < self.wall_margin:
                force[k] += self.k_wall * (1 / max(lo, 0.5) - 1 / self.wall_margin)
            if hi < self.wall_margin:
                force[k] -= self.k_wall * (1 / max(hi, 0.5) - 1 / self.wall_margin)

        desired = np.arctan2(force[1], force[0])
        omega = np.clip(self.k_p * wrap_angle(desired - e.psi), -e.omega_max, e.omega_max)
        return np.array([omega / e.cmd_limit], dtype=np.float32)

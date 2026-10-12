"""
2B İHA tehdit kaçınma ortamı (Gymnasium).

İHA sabit hızla uçar, ajan yalnızca dönüş hızını (ω) seçer.
Uçuş zarfı iki basit sınırla temsil edilir:
  1) |ω| <= omega_max               -> en küçük dönüş yarıçapı (yatış açısı / yük faktörü sınırı)
  2) |ω_t - ω_(t-1)| <= domega_max  -> dönüşe ne kadar hızlı girilebileceği (yatış hızı sınırı)

Ajanın komut aralığı zarfın 3 katıdır.
enforce_envelope=True  : komutlar zarfa kırpılır (güvenlik kalkanı), müdahaleler sayılır
enforce_envelope=False : komutlar olduğu gibi uygulanır, zarf ihlalleri sayılır
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

WORLD = 100.0        # alan boyutu
SPEED = 1.0          # İHA hızı (birim/adım)
UAV_RADIUS = 1.0
GOAL_RADIUS = 3.0
SAFE_MARGIN = 5.0    # tehdide bu boşluktan yakın geçiş cezalandırılır
SENSOR_RANGE = 30.0  # gözlem normalizasyonu
THREAT_VMAX = 0.6


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class ThreatAvoidanceEnv(gym.Env):
    def __init__(self, n_threats=5, enforce_envelope=True, omega_max=0.15,
                 domega_max=0.05, cmd_scale=3.0, max_steps=300):
        super().__init__()
        self.n_threats = n_threats
        self.enforce_envelope = enforce_envelope
        self.omega_max = omega_max
        self.domega_max = domega_max
        self.cmd_limit = omega_max * cmd_scale
        self.max_steps = max_steps
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        obs_dim = 8 + 5 * n_threats
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(obs_dim,), dtype=np.float32)

    # ---------- yardımcılar ----------
    def _to_body(self, v):
        """Dünya eksenindeki vektörü İHA gövde eksenine çevirir (x: burun yönü)."""
        c, s = np.cos(self.psi), np.sin(self.psi)
        return np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1]])

    def _clearances(self):
        """Her tehdide boşluk: merkez mesafesi - yarıçaplar. <= 0 ise çarpışma."""
        d = np.linalg.norm(self.threat_pos - self.pos, axis=1)
        return d - self.threat_r - UAV_RADIUS

    def _get_obs(self):
        to_goal = self.goal - self.pos
        g_dist = np.linalg.norm(to_goal)
        g_body = self._to_body(to_goal) / (g_dist + 1e-8)
        base = [self.pos[0] / WORLD, self.pos[1] / WORLD,
                np.cos(self.psi), np.sin(self.psi),
                self.omega / self.omega_max,
                g_dist / (WORLD * np.sqrt(2)),
                g_body[0], g_body[1]]
        clear = self._clearances()
        uav_vel = SPEED * np.array([np.cos(self.psi), np.sin(self.psi)])
        feats = []
        for i in np.argsort(clear):  # en yakın tehdit önce
            rel_p = self._to_body(self.threat_pos[i] - self.pos) / SENSOR_RANGE
            rel_v = self._to_body(self.threat_vel[i] - uav_vel) / (SPEED + THREAT_VMAX)
            feats += [rel_p[0], rel_p[1], rel_v[0], rel_v[1], clear[i] / SENSOR_RANGE]
        return np.clip(np.array(base + feats, dtype=np.float32), -10.0, 10.0)

    # ---------- Gymnasium arayüzü ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        self.pos = np.array([rng.uniform(5, 10), rng.uniform(20, 80)])
        self.goal = np.array([rng.uniform(88, 95), rng.uniform(20, 80)])
        to_goal = self.goal - self.pos
        self.psi = float(np.arctan2(to_goal[1], to_goal[0]) + rng.uniform(-0.5, 0.5))
        self.omega = 0.0

        self.threat_r = rng.uniform(4.0, 7.0, size=self.n_threats)
        self.threat_pos = np.zeros((self.n_threats, 2))
        for i in range(self.n_threats):
            while True:
                p = np.array([rng.uniform(25, 75), rng.uniform(10, 90)])
                if (np.linalg.norm(p - self.pos) > self.threat_r[i] + 15 and
                        np.linalg.norm(p - self.goal) > self.threat_r[i] + 8):
                    break
            self.threat_pos[i] = p
        ang = rng.uniform(0, 2 * np.pi, size=self.n_threats)
        spd = rng.uniform(0.2, THREAT_VMAX, size=self.n_threats)
        self.threat_vel = np.stack([np.cos(ang) * spd, np.sin(ang) * spd], axis=1)

        self.steps = 0
        self.prev_goal_dist = float(np.linalg.norm(self.goal - self.pos))
        self.envelope_violations = 0
        self.shield_interventions = 0
        self.min_clearance = float(self._clearances().min())
        return self._get_obs(), {}

    def step(self, action):
        a = float(np.clip(np.asarray(action, dtype=np.float64).reshape(-1)[0], -1.0, 1.0))
        omega_cmd = a * self.cmd_limit
        prev = self.omega

        if self.enforce_envelope:
            omega = float(np.clip(omega_cmd, prev - self.domega_max, prev + self.domega_max))
            omega = float(np.clip(omega, -self.omega_max, self.omega_max))
            if abs(omega - omega_cmd) > 1e-6:
                self.shield_interventions += 1
        else:
            omega = omega_cmd
        if abs(omega) > self.omega_max + 1e-6 or abs(omega - prev) > self.domega_max + 1e-6:
            self.envelope_violations += 1
        self.omega = omega

        # İHA kinematiği (sabit hız, Dubins benzeri)
        self.psi = wrap_angle(self.psi + omega)
        self.pos = self.pos + SPEED * np.array([np.cos(self.psi), np.sin(self.psi)])

        # Tehditler: sabit hızla hareket, alan sınırından sekme
        self.threat_pos += self.threat_vel
        for k in range(2):
            out = (self.threat_pos[:, k] < 0) | (self.threat_pos[:, k] > WORLD)
            self.threat_vel[out, k] *= -1
            self.threat_pos[:, k] = np.clip(self.threat_pos[:, k], 0, WORLD)
        self.steps += 1

        # Ödül
        clear = self._clearances()
        self.min_clearance = min(self.min_clearance, float(clear.min()))
        goal_dist = float(np.linalg.norm(self.goal - self.pos))
        reward = (self.prev_goal_dist - goal_dist) - 0.01          # ilerleme - zaman cezası
        near = clear[clear < SAFE_MARGIN]
        reward -= 0.1 * float(np.sum(1.0 - near / SAFE_MARGIN))    # yakın geçiş cezası
        self.prev_goal_dist = goal_dist

        # Bitiş koşulları
        terminated, outcome = False, None
        if clear.min() <= 0:
            reward, terminated, outcome = reward - 10.0, True, "collision"
        elif not (0 <= self.pos[0] <= WORLD and 0 <= self.pos[1] <= WORLD):
            reward, terminated, outcome = reward - 10.0, True, "out_of_bounds"
        elif goal_dist < GOAL_RADIUS:
            reward, terminated, outcome = reward + 10.0, True, "success"
        truncated = (not terminated) and self.steps >= self.max_steps
        if truncated:
            outcome = "timeout"

        info = {}
        if terminated or truncated:
            info = {"outcome": outcome, "steps": self.steps,
                    "min_clearance": self.min_clearance,
                    "envelope_violations": self.envelope_violations,
                    "shield_interventions": self.shield_interventions}
        return self._get_obs(), float(reward), terminated, truncated, info

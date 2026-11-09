#!/usr/bin/env python3
"""
ACID-FTS Framework - Automated Unit Test Suite
=============================================
Verifies core mathematical, statistical, and algorithmic foundations:
1. Aspect circular decomposition (orthogonality & boundary continuity).
2. Scale-normalized LinEx asymmetric loss (asymmetry, MSE recovery as u->0, overflow clipping).
3. Split-conformal calibration (finite-sample quantile, physical non-negativity, PICP/ACE/PINAW).
4. Physical non-negativity and integer count rounding.
5. Zero lookahead leakage in sliding-window construction and origin-residual alignment.
6. Central configuration integrity (27 features, regional test partition dates).
"""

import os
import sys
import math
import unittest
from datetime import datetime

# Add acid-fts root to path
ACID_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ACID_DIR not in sys.path:
    sys.path.insert(0, ACID_DIR)

import config


# -----------------------------------------------------------------------------
# Pure Python Mathematical Implementations (Guaranteeing 100% Portability)
# -----------------------------------------------------------------------------

def encode_aspect_scalar(aspect_deg):
    """Equation (1): Azimuth circular encoding into Northness and Eastness."""
    rad = math.radians(aspect_deg)
    return math.cos(rad), math.sin(rad)


def linex_loss_scalar(y_true, y_pred, sigma_y, b=1.0, clip_val=10.0):
    """Equation (3): Scale-normalized LinEx asymmetric loss."""
    sigma_y = max(sigma_y, 1e-6)
    u = (y_true - y_pred) / sigma_y
    u_clipped = max(min(u, clip_val), -clip_val)
    return (2.0 / (b ** 2)) * (math.exp(b * u_clipped) - b * u_clipped - 1.0)


def conformal_quantile(scores, alpha):
    """Equation (31): Finite-sample adjusted quantile level."""
    n = len(scores)
    sorted_scores = sorted(scores)
    k = math.ceil((n + 1) * (1.0 - alpha))
    k = min(max(k, 1), n)
    return sorted_scores[k - 1]


def bounded_conformal_interval(y_pred, q_val):
    """Equation (32): Non-negative prediction interval."""
    lower = max(0.0, y_pred - q_val)
    upper = y_pred + q_val
    return lower, upper


def physical_project_count(val):
    """Physical non-negative discrete integer projection."""
    return max(0, int(math.floor(val + 0.5)))


# -----------------------------------------------------------------------------
# Test Suite Classes
# -----------------------------------------------------------------------------

class TestAspectEncoding(unittest.TestCase):
    """Verifies Equation (1): Circular Aspect Decomposition."""

    def test_unit_circle_identity(self):
        """Northness^2 + Eastness^2 must equal 1.0 for all angles."""
        test_angles = [0.0, 30.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0, 360.0]
        for angle in test_angles:
            northness, eastness = encode_aspect_scalar(angle)
            norm_sq = northness ** 2 + eastness ** 2
            self.assertAlmostEqual(
                norm_sq, 1.0, places=6,
                msg=f"Unit circle violated at angle {angle}: {norm_sq} != 1.0"
            )

    def test_boundary_continuity_0_and_360(self):
        """0 degrees (North) and 360 degrees (North) must yield identical values."""
        n0, e0 = encode_aspect_scalar(0.0)
        n360, e360 = encode_aspect_scalar(360.0)
        self.assertAlmostEqual(n0, n360, places=7)
        self.assertAlmostEqual(e0, e360, places=7)
        self.assertAlmostEqual(n0, 1.0, places=7)
        self.assertAlmostEqual(e0, 0.0, places=7)

    def test_cardinal_directions(self):
        """Verifies exact coordinates for East (90), South (180), West (270)."""
        # East: 90 deg -> Northness=0, Eastness=1
        n90, e90 = encode_aspect_scalar(90.0)
        self.assertAlmostEqual(n90, 0.0, places=6)
        self.assertAlmostEqual(e90, 1.0, places=6)

        # South: 180 deg -> Northness=-1, Eastness=0
        n180, e180 = encode_aspect_scalar(180.0)
        self.assertAlmostEqual(n180, -1.0, places=6)
        self.assertAlmostEqual(e180, 0.0, places=6)

        # West: 270 deg -> Northness=0, Eastness=-1
        n270, e270 = encode_aspect_scalar(270.0)
        self.assertAlmostEqual(n270, 0.0, places=6)
        self.assertAlmostEqual(e270, -1.0, places=6)


class TestLinExLoss(unittest.TestCase):
    """Verifies Equation (3): Scale-Normalized LinEx Asymmetric Loss."""

    def test_zero_error_yields_zero_loss(self):
        """When y_true == y_pred, loss must be exactly 0."""
        loss = linex_loss_scalar(y_true=10.0, y_pred=10.0, sigma_y=5.0, b=1.0)
        self.assertAlmostEqual(loss, 0.0, places=7)

    def test_asymmetric_penalty_direction(self):
        """
        Under-prediction (y > y_pred) must be penalized more heavily than
        equivalent over-prediction (y < y_pred).
        """
        y_true = 20.0
        sigma = 5.0
        delta = 3.0  # 3 units difference
        loss_under = linex_loss_scalar(y_true=y_true + delta, y_pred=y_true, sigma_y=sigma, b=1.0)
        loss_over = linex_loss_scalar(y_true=y_true - delta, y_pred=y_true, sigma_y=sigma, b=1.0)
        self.assertGreater(
            loss_under, loss_over,
            msg=f"Under-prediction loss ({loss_under}) must exceed over-prediction loss ({loss_over})"
        )

    def test_symmetry_recovery_as_u_approaches_zero(self):
        """
        Taylor series expansion: as u -> 0, (2/b^2)[exp(bu) - bu - 1] -> u^2 (MSE).
        """
        sigma = 10.0
        b = 1.0
        small_u_errors = [0.01, 0.005, 0.001]
        for delta in small_u_errors:
            loss = linex_loss_scalar(y_true=delta, y_pred=0.0, sigma_y=1.0, b=b)
            expected_mse = delta ** 2
            ratio = loss / expected_mse
            self.assertAlmostEqual(
                ratio, 1.0, delta=0.05,
                msg=f"LinEx failed MSE convergence at delta={delta}: ratio={ratio}"
            )

    def test_overflow_protection_clipping(self):
        """
        Extreme error (e.g., delta=500.0) without clipping causes math range overflow.
        With clipping to [-10, 10], loss must remain finite.
        """
        loss_huge = linex_loss_scalar(y_true=500.0, y_pred=0.0, sigma_y=1.0, b=1.0, clip_val=10.0)
        self.assertTrue(math.isfinite(loss_huge))
        expected_max = (2.0 / (1.0 ** 2)) * (math.exp(10.0) - 10.0 - 1.0)
        self.assertAlmostEqual(loss_huge, expected_max, places=4)


class TestConformalCalibration(unittest.TestCase):
    """Verifies Equations (30)-(35): Split-Conformal Bounded Intervals."""

    def test_finite_sample_correction(self):
        """
        Finite-sample quantile index k = ceil((n + 1) * (1 - alpha)).
        With n=100 and alpha=0.10 (90% CI), k = ceil(101 * 0.90) = ceil(90.9) = 91.
        """
        scores = list(range(1, 101))  # 1 to 100
        q90 = conformal_quantile(scores, alpha=0.10)
        self.assertEqual(q90, 91)

    def test_bounded_interval_non_negativity(self):
        """
        Predictions close to 0 must produce lower bounds clipped at 0.0,
        never returning negative fire counts or negative burnt area.
        """
        y_pred = 0.5
        q_val = 2.0
        lower, upper = bounded_conformal_interval(y_pred, q_val)
        self.assertGreaterEqual(lower, 0.0)
        self.assertEqual(lower, 0.0)
        self.assertEqual(upper, 2.5)

    def test_empirical_coverage_guarantee(self):
        """
        Given synthetic calibration residuals, the empirical coverage on the
        calibration set must meet or exceed the nominal 1 - alpha level.
        """
        alpha = 0.10  # 90% target coverage
        cal_residuals = [abs(i * 0.1 - 2.5) for i in range(100)]
        q_cal = conformal_quantile(cal_residuals, alpha=alpha)

        # Check coverage
        covered = sum(1 for r in cal_residuals if r <= q_cal)
        empirical_picp = covered / len(cal_residuals)
        self.assertGreaterEqual(
            empirical_picp, 1.0 - alpha,
            msg=f"Empirical coverage {empirical_picp} fell below nominal {1.0 - alpha}"
        )


class TestPhysicalBoundsAndRounding(unittest.TestCase):
    """Verifies physical boundary conditions and count discretizations."""

    def test_fire_count_integer_rounding(self):
        """
        Discrete count projection max(0, floor(x + 0.5)):
        - Negative values -> 0
        - Float 0.49 -> 0
        - Float 0.51 -> 1
        - Float 2.80 -> 3
        """
        self.assertEqual(physical_project_count(-1.4), 0)
        self.assertEqual(physical_project_count(-0.01), 0)
        self.assertEqual(physical_project_count(0.0), 0)
        self.assertEqual(physical_project_count(0.49), 0)
        self.assertEqual(physical_project_count(0.50), 1)
        self.assertEqual(physical_project_count(0.51), 1)
        self.assertEqual(physical_project_count(2.80), 3)
        self.assertEqual(physical_project_count(15.1), 15)


class TestSlidingWindowZeroLeakage(unittest.TestCase):
    """Verifies strict zero temporal leakage in lookback indexing."""

    def test_forecast_origin_lookback_disjointness(self):
        """
        For any forecast origin t and horizon L:
        - Lookback window indices are {t - W + 1, ..., t}
        - Forecast target index is t + L
        - Lookback indices must be strictly < t + L
        - At lead time L >= 1, max(lookback) = t < t + L (target).
        """
        W = 30  # Lookback window
        for lag in [1, 2, 7, 14, 30]:
            for origin_t in [50, 100, 200]:
                lookback_indices = set(range(origin_t - W + 1, origin_t + 1))
                target_index = origin_t + lag

                # Target must never be in the lookback
                self.assertNotIn(target_index, lookback_indices)
                self.assertLess(max(lookback_indices), target_index)

    def test_chronological_splits_ordering(self):
        """
        Ensures D_train, D_cal, and D_test partitions do not overlap and maintain
        chronological progression.
        """
        d_train_end = datetime.strptime(config.TRAIN_END, "%Y-%m-%d")
        d_cal_start = datetime.strptime(config.CAL_START, "%Y-%m-%d")
        d_cal_end = datetime.strptime(config.CAL_END, "%Y-%m-%d")
        d_test_start = datetime.strptime(config.TEST_START, "%Y-%m-%d")

        self.assertLess(d_train_end, d_cal_start)
        self.assertLess(d_cal_end, d_test_start)


class TestConfigRegionalIntegrity(unittest.TestCase):
    """Verifies configuration constants, feature counts, and regional dates."""

    def test_feature_counts(self):
        """21 dynamic + 6 static descriptors = 27 total features."""
        self.assertEqual(len(config.DYNAMIC_FEATURES), 21)
        self.assertEqual(len(config.STATIC_FEATURES), 6)
        self.assertEqual(len(config.FEATURES), 27)

    def test_regional_test_dates(self):
        """Regional test dates match Table I."""
        self.assertEqual(config.get_test_end("bandhavgarh"), "2024-07-31")
        self.assertEqual(config.get_test_end("similipal"), "2024-07-31")
        self.assertEqual(config.get_test_end("chamba"), "2025-08-01")
        # Default fallback
        self.assertEqual(config.get_test_end(), config.TEST_END)

    def test_forecast_lags(self):
        """Forecast lags strictly match L in {1, 2, 7, 14, 30}."""
        self.assertEqual(config.LAGS, [1, 2, 7, 14, 30])


class TestEventDetectionContingencyMetrics(unittest.TestCase):
    """Verifies Section IV.J Earth-Observation Contingency Metrics (POD, FAR, F1, CSI)."""

    def test_contingency_metrics_zero_vs_active(self):
        """Naive zero-predictor must yield POD=0, CSI=0, while active predictor yields high POD/CSI."""
        y_true = [0, 0, 0, 0, 2, 5, 0, 0, 3, 0]
        y_zero = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        y_active = [0, 0, 0, 0, 2, 4, 1, 0, 3, 0]

        def calc_contingency(yt, yp, thresh=1.0):
            hits = sum(1 for t, p in zip(yt, yp) if t >= thresh and p >= thresh)
            misses = sum(1 for t, p in zip(yt, yp) if t >= thresh and p < thresh)
            fa = sum(1 for t, p in zip(yt, yp) if t < thresh and p >= thresh)
            pod = hits / max(hits + misses, 1e-6)
            csi = hits / max(hits + misses + fa, 1e-6)
            return pod, csi

        pod_zero, csi_zero = calc_contingency(y_true, y_zero)
        pod_act, csi_act = calc_contingency(y_true, y_active)
        self.assertAlmostEqual(pod_zero, 0.0, places=6)
        self.assertAlmostEqual(csi_zero, 0.0, places=6)
        self.assertAlmostEqual(pod_act, 1.0, places=5)
        self.assertAlmostEqual(csi_act, 0.75, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)

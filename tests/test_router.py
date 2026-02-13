"""Tests for the AESC profiler, catalog, and Router."""

from __future__ import annotations

import numpy as np
import pytest

from framework.aesc_profiler import (
    FINANCIAL_ENERGY_SYSTEM,
    EPIDEMIC_NETWORK,
    SUPPLY_CHAIN,
    create_profile,
    infer_roles,
    profile_summary,
    profile_to_dict,
)
from framework.catalog import (
    epistemic_distance,
    get_catalog,
    get_methods_by_family,
    get_methods_by_role,
)
from framework.config import RouterConfig
from framework.router import (
    compute_diversity_matrix,
    compute_fit,
    generate_decision_cards,
    select_kit,
)
from framework.types import (
    EpistemicFamily,
    Role,
    Scale,
)


# ===================================================================
# Profiler tests
# ===================================================================

class TestProfiler:
    def test_create_profile_defaults(self):
        p = create_profile("test", "a test system")
        assert p.name == "test"
        assert len(p.primary_roles) + len(p.secondary_roles) > 0

    def test_infer_roles_returns_at_least_one_primary(self):
        p = create_profile("minimal", "minimal profile")
        assert len(p.primary_roles) >= 1

    def test_financial_energy_has_expected_roles(self):
        p = FINANCIAL_ENERGY_SYSTEM
        all_roles = set(p.primary_roles + p.secondary_roles)
        # Should include BEHAVIOR and FORECAST given fast dynamics + streaming
        assert Role.BEHAVIOR in all_roles
        assert Role.FORECAST in all_roles

    def test_profile_to_dict_roundtrip(self):
        d = profile_to_dict(FINANCIAL_ENERGY_SYSTEM)
        assert isinstance(d, dict)
        assert d["name"] == FINANCIAL_ENERGY_SYSTEM.name
        assert isinstance(d["primary_roles"], list)

    def test_profile_summary_is_string(self):
        s = profile_summary(SUPPLY_CHAIN)
        assert isinstance(s, str)
        assert SUPPLY_CHAIN.name in s


# ===================================================================
# Catalog tests
# ===================================================================

class TestCatalog:
    def test_catalog_has_12_methods(self):
        assert len(get_catalog()) == 12

    def test_get_methods_by_role(self):
        forecasters = get_methods_by_role(Role.FORECAST)
        assert len(forecasters) >= 4
        for m in forecasters:
            assert Role.FORECAST in m.roles_supported

    def test_get_methods_by_family(self):
        ml = get_methods_by_family(EpistemicFamily.MACHINE_LEARNING)
        assert len(ml) == 3  # GBM, DL, GNN
        for m in ml:
            assert m.family == EpistemicFamily.MACHINE_LEARNING

    def test_epistemic_distance_range(self):
        catalog = get_catalog()
        for i, m1 in enumerate(catalog):
            for m2 in catalog[i + 1:]:
                d = epistemic_distance(m1, m2)
                assert 0.0 <= d <= 1.0

    def test_epistemic_distance_self_is_zero(self):
        for m in get_catalog():
            assert epistemic_distance(m, m) == 0.0

    def test_epistemic_distance_symmetry(self):
        catalog = get_catalog()
        for i in range(len(catalog)):
            for j in range(i + 1, len(catalog)):
                d1 = epistemic_distance(catalog[i], catalog[j])
                d2 = epistemic_distance(catalog[j], catalog[i])
                assert abs(d1 - d2) < 1e-12


# ===================================================================
# Router tests
# ===================================================================

class TestRouter:
    def test_fit_scores_in_unit_interval(self):
        profile = FINANCIAL_ENERGY_SYSTEM
        for m in get_catalog():
            score = compute_fit(profile, m)
            assert 0.0 <= score <= 1.0, f"{m.name}: fit={score}"

    def test_diversity_matrix_symmetric(self):
        catalog = get_catalog()
        mat = compute_diversity_matrix(catalog)
        assert mat.shape == (12, 12)
        np.testing.assert_array_almost_equal(mat, mat.T)
        np.testing.assert_array_almost_equal(np.diag(mat), 0.0)

    def test_select_kit_financial_energy(self):
        config = RouterConfig(min_methods=3, max_methods=7)
        selection = select_kit(FINANCIAL_ENERGY_SYSTEM, config=config)
        assert config.min_methods <= len(selection.selected_methods) <= config.max_methods
        names = [m.name for m in selection.selected_methods]
        assert len(names) == len(set(names)), "duplicates found"

    def test_select_kit_epidemic(self):
        selection = select_kit(EPIDEMIC_NETWORK)
        assert len(selection.selected_methods) >= 3
        # Epidemiological models should rank well
        families = {m.family for m in selection.selected_methods}
        assert len(families) >= 2, "kit should be diverse"

    def test_select_kit_supply_chain(self):
        selection = select_kit(SUPPLY_CHAIN)
        assert len(selection.selected_methods) >= 3

    def test_at_least_one_interpretable(self):
        config = RouterConfig(require_interpretable=True)
        selection = select_kit(FINANCIAL_ENERGY_SYSTEM, config=config)
        interpretable = [
            m for m in selection.selected_methods if m.interpretability == "high"
        ]
        assert len(interpretable) >= 1

    def test_justifications_present(self):
        selection = select_kit(FINANCIAL_ENERGY_SYSTEM)
        for m in selection.selected_methods:
            assert m.name in selection.justifications
            assert len(selection.justifications[m.name]) > 0

    def test_decision_cards(self):
        selection = select_kit(FINANCIAL_ENERGY_SYSTEM)
        cards = generate_decision_cards(selection, FINANCIAL_ENERGY_SYSTEM)
        assert len(cards) == len(selection.selected_methods)
        for card in cards:
            assert "method" in card
            assert "roles_covered" in card
            assert "key_assumptions" in card
            assert "failure_modes" in card
            assert "contribution" in card

    def test_avg_diversity_positive(self):
        selection = select_kit(FINANCIAL_ENERGY_SYSTEM)
        assert selection.avg_diversity > 0.0

    def test_total_fit_positive(self):
        selection = select_kit(FINANCIAL_ENERGY_SYSTEM)
        assert selection.total_fit > 0.0

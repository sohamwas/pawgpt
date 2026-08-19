"""
Tests for the structured filter. No API keys, no network, no LLM - the whole
retrieval core is deterministic, so it can be verified offline and in CI.

The expected counts are not invented: they were measured against the shipped
dataset before the filter existed, which is what makes them a real check on the
implementation rather than a restatement of it.
"""
import pytest

from retrieval import traits
from retrieval.filters import UnknownTrait, filter_breeds


# --- exact matches -----------------------------------------------------------

def test_readme_flagship_query():
    """The example query from the README: quiet, low-energy, apartment-suitable."""
    result = filter_breeds(min_apartment=4, max_barking=2, max_energy=3)
    assert result.exact
    assert len(result) == 22
    assert "Cavalier King Charles Spaniel" in result.breeds


def test_small_dog_left_alone_in_a_flat():
    result = filter_breeds(max_weight_kg=15, min_alone=4, min_apartment=4)
    assert result.exact
    assert len(result) == 7
    assert "Lhasa Apso" in result.breeds


def test_quiet_apartment_dog_tolerating_solitude():
    """Three constraints narrow 391 breeds to the one famously barkless breed."""
    result = filter_breeds(min_apartment=4, max_barking=2, min_alone=4)
    assert result.exact
    assert result.breeds == ["Basenji"]


def test_novice_owner_low_shedding_good_with_kids():
    result = filter_breeds(min_novice=4, max_shedding=2, min_kid_friendly=4)
    assert result.exact
    assert len(result) == 41


def test_single_constraint_selectivity():
    """Selectivity is reported even when the filter succeeds."""
    result = filter_breeds(min_alone=4)
    assert len(result) == 22
    assert result.pass_rates["min_alone"] == pytest.approx(22 / 391, abs=1e-3)


# --- relaxation --------------------------------------------------------------

def test_over_constrained_request_relaxes_and_says_so():
    result = filter_breeds(min_apartment=5, max_barking=1, max_shedding=1,
                           min_novice=5, min_kid_friendly=5)
    assert not result.exact
    assert result.breeds, "relaxation should recover some near-matches"
    assert result.relaxations, "the loosened constraint must be recorded"
    assert "loosened" in result.summary()


def test_relaxation_targets_the_most_selective_constraint():
    """min_alone passes only 6% of breeds, so it is the one that should give."""
    result = filter_breeds(min_apartment=5, max_barking=1, min_alone=5)
    assert not result.exact
    assert [r.trait for r in result.relaxations] == ["alone"]


def test_impossible_request_returns_nothing_rather_than_everything():
    """Relaxation is capped, so a contradiction does not degrade into 391 breeds."""
    result = filter_breeds(min_weight_kg=80, max_weight_kg=3)
    assert result.breeds == [] or len(result) < 391
    assert not result.exact


# --- ordinal handling --------------------------------------------------------

def test_ordinal_accepts_label_or_number():
    by_label = filter_breeds(min_kid_friendly="Very High")
    by_number = filter_breeds(min_kid_friendly=5)
    assert by_label.breeds == by_number.breeds
    assert len(by_label) == 149


def test_ordinal_label_is_case_insensitive():
    assert filter_breeds(min_energy="very high").breeds == \
           filter_breeds(min_energy=5).breeds


def test_unknown_ordinal_label_is_rejected():
    with pytest.raises(ValueError, match="expects one of"):
        filter_breeds(min_kid_friendly="Extremely High")


# --- ranking -----------------------------------------------------------------

def test_results_are_ranked_not_arbitrary():
    """Best match first: the top breed should beat the last on the asked traits."""
    result = filter_breeds(min_apartment=4, max_shedding=2)
    top, bottom = result.breeds[0], result.breeds[-1]
    scores = result.table.set_index("Breed Name")["_score"]
    assert scores[top] >= scores[bottom]


def test_ranking_is_deterministic():
    assert filter_breeds(min_apartment=4, max_shedding=2).breeds == \
           filter_breeds(min_apartment=4, max_shedding=2).breeds


# --- categorical -------------------------------------------------------------

def test_breed_group_filters_by_equality():
    result = filter_breeds(breed_group="Hound Dogs", min_apartment=4)
    groups = result.table["Dog Breed Group"].unique()
    assert list(groups) == ["Hound Dogs"]


# --- input validation --------------------------------------------------------

def test_unknown_trait_names_the_alternatives():
    with pytest.raises(UnknownTrait, match="apartment"):
        filter_breeds(min_fluffiness=3)


def test_malformed_parameter_is_rejected():
    with pytest.raises(UnknownTrait):
        filter_breeds(apartment=4)  # missing min_/max_ prefix


def test_excluded_columns_stay_excluded():
    """Size and Dog Size contradict each other; they must not be filterable."""
    for name in ("min_size", "max_size", "min_dog_size"):
        with pytest.raises(UnknownTrait):
            filter_breeds(**{name: 3})


# --- registry invariants -----------------------------------------------------

def test_every_trait_column_exists_and_is_complete():
    df = traits.load()
    for trait in traits.TRAITS:
        col = traits.value_column(trait)
        assert col in df.columns, f"{trait} did not normalize"
        assert not df[col].isna().any(), f"{trait} has nulls after normalize"


def test_filterable_traits_are_all_registered():
    assert set(traits.FILTERABLE) <= set(traits.TRAITS)


def test_registry_does_not_expose_excluded_columns():
    used = {traits.column(t) for t in traits.TRAITS}
    assert not (used & traits.EXCLUDED_COLUMNS)

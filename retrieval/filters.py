"""
Structured filtering over the 391 breeds - the retrieval mode cosine similarity
cannot provide.

Semantic search answers "what text resembles this question". It cannot answer
"which breeds satisfy all of these constraints", rank by a measured trait, or
report that nothing qualifies. Those are the questions PawGPT is actually asked
("a quiet dog for a flat that I can leave while I work"), and every one of them is
a handful of columns and a comparison.

The public surface is `filter_breeds(**constraints)`, taking `min_<trait>` /
`max_<trait>` keyword arguments plus `breed_group`, and returning a `FilterResult`
that carries not just the matches but *why* - which constraint excluded what, and
what had to be loosened if nothing qualified.

    >>> filter_breeds(min_apartment=4, max_barking=2, min_alone=4).breeds
    ['Basenji']
"""
from dataclasses import dataclass, field

import pandas as pd

from . import traits

# How far a single relaxation step moves a constraint. Ratings and ordinals live on
# a 1-5 integer scale so one point is the natural increment; continuous traits have
# no such unit, so they move by a fraction of their observed range.
RATING_STEP = 1
CONTINUOUS_STEP_FRACTION = 0.10

# Cap on total loosening. Without it, an impossible request relaxes until the
# constraints mean nothing and the result is "here are 391 breeds", which is a
# worse answer than "nothing matches".
MAX_RELAXATION_STEPS = 6

# A filter can be correct and still unhelpful. "quiet, apartment-suitable, tolerates
# being alone" matches exactly one breed of 391 - a true answer that leaves the person
# with no choice at all. So when the exact set is smaller than this, near-misses are
# gathered as well and returned separately, clearly labelled with what they miss.
# Exact matches are never diluted: `breeds` still holds only those, and near-misses
# live in `near`.
PREFERRED_RESULTS = 3


class UnknownTrait(ValueError):
    pass


@dataclass
class Constraint:
    trait: str
    op: str        # "min" or "max"
    value: float

    @property
    def name(self):
        return f"{self.op}_{self.trait}"

    def mask(self, df):
        col = df[traits.value_column(self.trait)]
        return col >= self.value if self.op == "min" else col <= self.value

    def describe(self):
        symbol = ">=" if self.op == "min" else "<="
        return f"{self.trait} {symbol} {self.value:g}"


@dataclass
class Relaxation:
    trait: str
    op: str
    original: float
    relaxed: float

    def describe(self):
        symbol = ">=" if self.op == "min" else "<="
        return (f"{self.trait} loosened from {symbol} {self.original:g} "
                f"to {symbol} {self.relaxed:g}")


@dataclass
class FilterResult:
    breeds: list                       # breed names, best match first
    exact: bool                        # True if every constraint was satisfied as asked
    requested: list = field(default_factory=list)    # Constraint, as the caller gave them
    applied: list = field(default_factory=list)      # Constraint, after any relaxation
    relaxations: list = field(default_factory=list)  # Relaxation
    pass_rates: dict = field(default_factory=dict)   # constraint name -> fraction passing
    table: pd.DataFrame = None         # per-breed values for the constrained traits
    near: list = field(default_factory=list)         # (breed, [missed constraint...])
    near_table: pd.DataFrame = None

    def __len__(self):
        return len(self.breeds)

    @property
    def all_breeds(self):
        """Exact matches first, then near-misses - what the caller can offer."""
        return self.breeds + [name for name, _ in self.near]

    def summary(self, limit=10):
        """Plain-text rendering for a prompt or a log line."""
        if not self.requested:
            return f"No constraints given; {len(self.breeds)} breeds."

        asked = ", ".join(c.describe() for c in self.requested)
        lines = [f"Constraints: {asked}"]

        if not self.breeds:
            lines.append("No breed satisfies these, even after relaxation.")
            return "\n".join(lines)

        if self.exact:
            lines.append(f"{len(self.breeds)} breed(s) match all constraints exactly.")
        else:
            lines.append(
                f"No breed matches all constraints. {len(self.breeds)} near-match(es) "
                "after relaxing: " + "; ".join(r.describe() for r in self.relaxations)
            )

        shown = self.breeds[:limit]
        cols = ["Breed Name"] + [traits.value_column(c.trait) for c in self.applied]
        view = self.table.loc[self.table["Breed Name"].isin(shown), cols]
        view = view.set_index("Breed Name").loc[shown]
        view.columns = [c.lstrip("_") for c in view.columns]
        lines.append(view.to_string())
        if len(self.breeds) > limit:
            lines.append(f"... and {len(self.breeds) - limit} more")

        if self.near:
            lines.append(
                f"\nOnly {len(self.breeds)} breed(s) match exactly, so here are "
                f"{len(self.near)} near match(es). Each meets every requirement "
                "except the one named:"
            )
            near_names = [name for name, _ in self.near]
            nview = self.near_table.loc[
                self.near_table["Breed Name"].isin(near_names), cols
            ].set_index("Breed Name").loc[near_names]
            nview.columns = [c.lstrip("_") for c in nview.columns]
            lines.append(nview.to_string())
            for name, missed in self.near:
                lines.append(f"  {name}: misses {', '.join(missed)}")
        return "\n".join(lines)


def parse_constraints(kwargs):
    """Turn min_x=4 / max_y=2 / breed_group=<value> into Constraint objects.

    Returns (numeric_constraints, categorical_equalities).
    """
    numeric, categorical = [], {}
    for key, value in kwargs.items():
        if value is None:
            continue

        if key in traits.TRAITS and traits.kind(key) == traits.CATEGORICAL:
            categorical[key] = value
            continue

        op, _, trait = key.partition("_")
        if op not in ("min", "max") or not trait:
            raise UnknownTrait(
                f"{key!r} is not a valid constraint. Use min_<trait> or max_<trait>, "
                f"e.g. min_apartment=4. Available: {', '.join(traits.FILTERABLE)}"
            )
        if trait not in traits.TRAITS:
            raise UnknownTrait(
                f"unknown trait {trait!r}. Available: {', '.join(traits.FILTERABLE)}"
            )
        if traits.kind(trait) == traits.CATEGORICAL:
            raise UnknownTrait(
                f"{trait!r} is categorical - pass {trait}=<value>, not {key}"
            )

        numeric.append(Constraint(trait, op, float(traits.to_number(trait, value))))
    return numeric, categorical


def _step(constraint):
    """How much to loosen this constraint by, in one move."""
    if traits.kind(constraint.trait) == traits.CONTINUOUS:
        lo, hi = traits.scale(constraint.trait)
        return (hi - lo) * CONTINUOUS_STEP_FRACTION
    return RATING_STEP


def _loosen(constraint):
    """A copy of the constraint, one step weaker. None if already past the limit."""
    lo, hi = traits.scale(constraint.trait)
    step = _step(constraint)
    value = constraint.value - step if constraint.op == "min" else constraint.value + step
    if constraint.op == "min" and value < lo:
        return None
    if constraint.op == "max" and value > hi:
        return None
    return Constraint(constraint.trait, constraint.op, value)


def _rank(df, constraints):
    """Score each row 0-1 by how well it satisfies the constrained traits.

    Each constrained trait is normalized to its observed range and oriented so that
    1.0 is the direction the caller asked for, then averaged. This is what decides
    which 5 of 22 matches to show, and it is deterministic - the same request always
    produces the same order.
    """
    if not constraints:
        return pd.Series(0.0, index=df.index)

    scores = []
    for c in constraints:
        lo, hi = traits.scale(c.trait)
        span = hi - lo or 1.0
        norm = (df[traits.value_column(c.trait)] - lo) / span
        scores.append(norm if c.op == "min" else 1.0 - norm)
    return sum(scores) / len(scores)


def filter_breeds(df=None, preferred=PREFERRED_RESULTS, **kwargs):
    """Find breeds matching the given constraints, relaxing them if none do.

    Hard-filters first: if anything satisfies every constraint, those are exact
    matches and are returned as such. Only when nothing qualifies does it loosen -
    one step at a time, always taking the most selective constraint first, since
    that is the one excluding the most - and it records exactly what it changed, so
    the caller can say so rather than quietly presenting near-misses as matches.
    """
    df = traits.load() if df is None else df
    numeric, categorical = parse_constraints(kwargs)

    base = df
    for trait, value in categorical.items():
        base = base[base[traits.value_column(trait)] == value]

    # Selectivity of each constraint on its own. This drives relaxation order, and
    # is worth surfacing regardless: it is the answer to "why did I get nothing".
    pass_rates = {c.name: float(c.mask(base).mean()) if len(base) else 0.0
                  for c in numeric}

    def matches(constraints):
        if not constraints:
            return base
        mask = constraints[0].mask(base)
        for c in constraints[1:]:
            mask &= c.mask(base)
        return base[mask]

    applied = list(numeric)
    hits = matches(applied)
    exact = True
    original_values = {}

    steps = 0
    while hits.empty and applied and steps < MAX_RELAXATION_STEPS:
        # Loosen whichever surviving constraint excludes the most on its own.
        candidates = [c for c in applied if _loosen(c) is not None]
        if not candidates:
            break
        target = min(candidates, key=lambda c: pass_rates.get(c.name, 0.0))
        weaker = _loosen(target)
        applied = [weaker if c is target else c for c in applied]
        original_values.setdefault(target.trait, target.value)
        exact = False
        steps += 1
        hits = matches(applied)

    relaxations = [
        Relaxation(c.trait, c.op, original_values[c.trait], c.value)
        for c in applied if c.trait in original_values
    ]

    if hits.empty:
        return FilterResult(breeds=[], exact=False, requested=numeric,
                            applied=applied, relaxations=relaxations,
                            pass_rates=pass_rates, table=hits)

    hits = hits.copy()
    hits["_score"] = _rank(hits, applied)
    hits = hits.sort_values(["_score", "Breed Name"], ascending=[False, True])
    matched = hits["Breed Name"].tolist()

    # Too few exact matches to be a useful recommendation? Gather the breeds that miss
    # the fewest constraints, so the answer can offer alternatives rather than a single
    # take-it-or-leave-it result. Exact matches stay in `breeds`; these go in `near`.
    near, near_table = [], None
    if exact and len(matched) < preferred and applied:
        satisfied = sum(c.mask(base).astype(int) for c in applied)
        shortfall = len(applied) - satisfied
        candidates = base[(shortfall == 1) & ~base["Breed Name"].isin(matched)].copy()
        if not candidates.empty:
            candidates["_score"] = _rank(candidates, applied)
            candidates = candidates.sort_values(
                ["_score", "Breed Name"], ascending=[False, True]
            ).head(preferred - len(matched))
            for _, row in candidates.iterrows():
                missed = [c.describe() for c in applied
                          if not c.mask(row.to_frame().T).iloc[0]]
                near.append((row["Breed Name"], missed))
            near_table = candidates

    return FilterResult(
        breeds=matched,
        exact=exact,
        requested=numeric,
        applied=applied,
        relaxations=relaxations,
        pass_rates=pass_rates,
        table=hits,
        near=near,
        near_table=near_table,
    )

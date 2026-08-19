"""
Single source of truth mapping usable trait names onto the dataset's columns.

The CSV's column names are unusable as function parameters ("Adapts Well To
Apartment Living", "Avg. Weight, kg"), and the same conceptual trait is stored in
two different encodings depending on the column: most are integers 1-5, eight
exist *only* as text labels. Everything downstream - the filter, the tool schema
handed to the LLM, the UI labels - is generated from the table below, so adding a
trait is a one-line change in one place.

All 391 rows are complete: every column referenced here has zero nulls.
"""
import functools
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "data" / "dogs_final_for_rag.csv"

# The eight *_Text columns use exactly this five-value vocabulary, so mapping to
# integers is a bijection - "Very High" -> 5 -> "Very High" loses nothing. It just
# puts them in the same space as the 1-5 rating columns so one comparison works
# for both.
ORDINAL = {"Very Low": 1, "Low": 2, "Average": 3, "High": 4, "Very High": 5}

RATING = "rating"          # integer 1-5, already numeric in the CSV
ORDINAL_TEXT = "ordinal"   # "Very Low".."Very High", needs ORDINAL applied
CONTINUOUS = "continuous"  # real-valued, no fixed scale
CATEGORICAL = "categorical"

# name -> (column, kind)
TRAITS = {
    # --- 1-5 ratings -------------------------------------------------------
    "apartment":          ("Adapts Well To Apartment Living", RATING),
    "novice":             ("Good For Novice Owners",          RATING),
    "sensitivity":        ("Sensitivity Level",               RATING),
    "alone":              ("Tolerates Being Alone",           RATING),
    "cold":               ("Tolerates Cold Weather",          RATING),
    "hot":                ("Tolerates Hot Weather",           RATING),
    "stranger_friendly":  ("Friendly Toward Strangers",       RATING),
    "shedding":           ("Amount Of Shedding",              RATING),
    "drooling":           ("Drooling Potential",              RATING),
    "grooming_ease":      ("Easy To Groom",                   RATING),
    "weight_gain":        ("Potential For Weight Gain",       RATING),
    "mouthiness":         ("Potential For Mouthiness",        RATING),
    "prey_drive":         ("Prey Drive",                      RATING),
    "barking":            ("Tendency To Bark Or Howl",        RATING),
    "wanderlust":         ("Wanderlust Potential",            RATING),
    "intensity":          ("Intensity",                       RATING),
    "playfulness":        ("Potential For Playfulness",       RATING),

    # --- traits that exist ONLY as text labels ------------------------------
    # These are not worded copies of any rating column. Crosstabbing
    # Easy To Train_Text against the numeric Trainability shows "Average"
    # spanning 1.8 to 4.4, so they are independent signals. Three of the most
    # commonly asked criteria - kid_friendly, energy, exercise - live only here.
    "kid_friendly":       ("Kid-Friendly_Text",               ORDINAL_TEXT),
    "dog_friendly":       ("Dog Friendly_Text",               ORDINAL_TEXT),
    "family_affection":   ("Affectionate With Family_Text",   ORDINAL_TEXT),
    "trainability":       ("Easy To Train_Text",              ORDINAL_TEXT),
    "intelligence":       ("Intelligence_Text",               ORDINAL_TEXT),
    "energy":             ("Energy Level_Text",               ORDINAL_TEXT),
    "exercise":           ("Exercise Needs_Text",             ORDINAL_TEXT),
    "health":             ("General Health_Text",             ORDINAL_TEXT),

    # --- continuous ---------------------------------------------------------
    "weight_kg":          ("Avg. Weight, kg",                 CONTINUOUS),
    "height_cm":          ("Avg. Height, cm",                 CONTINUOUS),
    "lifespan":           ("Avg. Life Span, years",           CONTINUOUS),

    # --- categorical --------------------------------------------------------
    "breed_group":        ("Dog Breed Group",                 CATEGORICAL),
}

# Columns deliberately left out, with the reason - so nobody "fixes" this later by
# adding them back:
#
#   Size, Dog Size            The two contradict each other and neither tracks
#                             actual mass. Numeric Size bucket 3 spans 5.6kg to
#                             48.4kg, and "Very Large" contains Size 2, 3, 4 and 5.
#                             weight_kg is clean and continuous - use that.
#
#   Adaptability_Text         313 of 391 rows are "Unknown".
#
#   All Around Friendliness   Derived means of other columns (they hold fractional
#   Health And Grooming Needs values like 3.67 and 1.8). Exposing both a composite
#   Trainability              and its components gives the LLM two ways to say one
#   Physical Needs            thing, which is a reliable source of malformed calls.
#                             The components are all exposed individually above.
EXCLUDED_COLUMNS = {
    "Size", "Dog Size", "Adaptability_Text", "All Around Friendliness",
    "Health And Grooming Needs", "Trainability", "Physical Needs",
}

# Traits worth exposing to the LLM as filter parameters. The full table above stays
# available to code, but a tool schema listing every trait in both directions runs
# to ~54 parameters, which is where smaller models start emitting malformed calls.
# This is the subset users actually ask about; widen it if evaluation shows gaps.
FILTERABLE = [
    "apartment", "novice", "alone", "cold", "hot", "shedding", "drooling",
    "grooming_ease", "barking", "prey_drive", "energy", "exercise",
    "kid_friendly", "dog_friendly", "trainability", "weight_kg", "lifespan",
    "breed_group",
]

NUMERIC_KINDS = {RATING, ORDINAL_TEXT, CONTINUOUS}


def column(trait):
    return TRAITS[trait][0]


def kind(trait):
    return TRAITS[trait][1]


def value_column(trait):
    """Name of the normalized column added by `normalize`."""
    return f"_{trait}"


def normalize(df):
    """Add a `_<trait>` column for every trait, all numerically comparable.

    Ordinal text is mapped through ORDINAL; ratings and continuous values are
    copied as-is. After this, downstream code never touches a raw column name and
    never has to care which encoding a trait originally used.
    """
    df = df.copy()
    for trait, (col, k) in TRAITS.items():
        df[value_column(trait)] = df[col].map(ORDINAL) if k == ORDINAL_TEXT else df[col]
    return df


def to_number(trait, value):
    """Accept either 4 or "High" for an ordinal trait; pass anything else through."""
    if kind(trait) == ORDINAL_TEXT and isinstance(value, str):
        try:
            return ORDINAL[value.strip().title()]
        except KeyError:
            raise ValueError(
                f"{trait!r} expects one of {list(ORDINAL)} or 1-5, got {value!r}"
            ) from None
    return value


def scale(trait):
    """(min, max) observed for a trait - used to normalize ranking scores."""
    col = load()[value_column(trait)]
    return float(col.min()), float(col.max())


@functools.lru_cache(maxsize=1)
def load():
    """Load and normalize the structured columns once.

    Only the 30 columns named in TRAITS are read. The CSV's `Combined_Info` column
    holds the full breed write-ups - 23 MB of prose that the filter never looks at,
    and that lives in data/breed_prose.json in deduplicated form for the code that
    does. Reading it here cost 47.6 MB of resident memory against 0.3 MB for the
    columns actually used.
    """
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {CSV_PATH}")
    needed = ["Breed Name"] + [col for col, _ in TRAITS.values()]
    return normalize(pd.read_csv(CSV_PATH, usecols=sorted(set(needed))))

"""Closed vocabulary of the synthetic scene-graph world."""

SHAPES = ("cube", "sphere", "cylinder", "cone")
COLORS = ("red", "blue", "green", "yellow", "purple")
MATERIALS = ("metal", "rubber")

# Discrete horizontal positions. Relations are derived from slot order only.
N_SLOTS = 3

# Attribute types that can be swapped / replaced. Shape is the object category.
ATTR_VALUES = {"color": COLORS, "material": MATERIALS}
ATTRS = tuple(ATTR_VALUES)

# Default (shape, color) pairs never seen in any non-held-out split.
# Chosen as a partial Latin square so every shape and every color that is
# held out in one pair still appears in training with other partners.
DEFAULT_HELDOUT_PAIRS = (
    ("cube", "red"),
    ("sphere", "blue"),
    ("cylinder", "green"),
    ("cone", "yellow"),
)

FUNCTION_WORDS = ("a", "the", "that", "is", "and", "of", "left", "right", ",", ";")

WORDS = FUNCTION_WORDS + SHAPES + COLORS + MATERIALS

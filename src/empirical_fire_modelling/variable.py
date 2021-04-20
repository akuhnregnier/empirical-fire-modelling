# -*- coding: utf-8 -*-
from abc import ABC
from dataclasses import dataclass, field
from enum import Enum

from immutabledict import immutabledict


@dataclass(frozen=True)
class VariableFactory:
    """Creation of related Variables based on given shifts (in months)."""

    rank: int
    name: str
    units: str

    def __getitem__(self, months):
        if months not in lags:
            raise ValueError(f"Unsupported months '{months}'. Expected one of {lags}.")
        return StandardVariable(
            rank=self.rank, name=self.name, shift=months, units=self.units, parent=self
        )

    def __str__(self):
        return self.name


@dataclass(frozen=True, order=True, init=False)
class Variable(ABC):
    """A variable with its associated name, shift (in months), and units."""

    rank: int
    name: str = field(compare=False)
    shift: int
    units: str = field(compare=False)
    parent: VariableFactory = field(compare=False)

    def get_offset(self):
        """Return a transformed Variable if there is a large (>12) shift."""
        if self.shift >= 12 and not isinstance(self, OffsetVariable):
            return OffsetVariable(
                rank=self.rank,
                name=self.name,
                shift=self.shift,
                units=self.units,
                parent=self.parent,
            )
        return self

    def get_standard(self):
        """The inverse of `get_offset()`."""
        if self.shift >= 12 and isinstance(self, OffsetVariable):
            return StandardVariable(
                rank=self.rank,
                name=self.name,
                shift=self.shift,
                units=self.units,
                parent=self.parent,
            )
        return self


@dataclass(frozen=True, order=True)
class StandardVariable(Variable):
    @property
    def _fill_root(self):
        """Add the fill params if needed."""
        if self.parent in filled_variables:
            return f"{self.name} {st_persistent_perc}P {st_k}k"
        return self.name

    @property
    def _nn_fill_root(self):
        """Add the fill params if needed."""
        if self.parent in filled_variables:
            return f"{self.name} {nn_n_months}NN"
        return self.name

    @property
    def filled(self):
        """Filled name (if applicable)."""
        if self.parent in filled_variables:
            return f"{self._fill_root} {self.shift}M"
        return self._fill_root

    @property
    def nn_filled(self):
        """Filled name (if applicable)."""
        if self.parent in filled_variables:
            return f"{self._nn_fill_root} {self.shift}M"
        return self._nn_fill_root

    def __str__(self):
        if self.shift != 0:
            return f"{self.name} {self.shift}M"
        return self.name

    @property
    def raw(self):
        if self.shift != 0:
            return f"{self.name} -{self.shift} Month"
        return self.name

    @property
    def raw_filled(self):
        if self.shift != 0:
            return f"{self._fill_root} -{self.shift} Month"
        return self._fill_root

    @property
    def raw_nn_filled(self):
        if self.shift != 0:
            return f"{self._nn_fill_root} -{self.shift} Month"
        return self._nn_fill_root


@dataclass(frozen=True, order=True)
class OffsetVariable(Variable):
    """A variable with its associated name, shift (in months), and units.

    This variable represents an anomaly from its own values shift % 12 months ago.

    """

    comp_shift: int = field(init=False, compare=False)

    def __post_init__(self):
        if self.shift < 12:
            raise ValueError(f"Expected a shift >= 12, got '{self.shift}'.")
        object.__setattr__(self, "comp_shift", self.shift % 12)

    def __str__(self):
        return f"{self.name} Δ{self.shift}M"


def sort_variables(variables):
    """Sort variables based on their rank and shift.

    Note that this relies on all variables having a unique rank.

    """
    return tuple(sorted(variables, key=lambda v: (v.rank, v.shift)))


def get_matching(variables, strict=True, single=True, **criteria):
    """Given a set of criteria, find the matching variables(s).

    Args:
        variables (iterable of Variable): Variables to match against.
        strict (bool): If True, require that at least one match is found (see
            `single`).
        single (bool): If True, require that exactly one variable is found.
        **criteria: Criteria to match against, e.g. {'name': 'FAPAR'}.

    Returns:
        Variable: If `single` is True.
        tuple of Variable: Otherwise.

    Raises:
        RuntimeError: If no matching variable was found.
        RuntimeError: If `single` is True and more than a single matching variable was
            found.

    """
    matching = []
    for var in variables:
        for crit_name, crit_info in criteria.items():
            if getattr(var, crit_name) == crit_info:
                continue
            else:
                break
        else:
            matching.append(var)

    if not matching and strict:
        raise RuntimeError("No matching variables were found.")
    if single:
        if len(matching) > 1:
            raise RuntimeError(
                f"Expected to find 1 matching variable. Found '{matching}'."
            )
        if not matching:
            return ()
        return matching[0]
    return tuple(matching)


def match_factory(variable, factories):
    """Match variable to VariableFactory using rank, name, and units.

    Args:
        variable (Variable): Variable to match.
        factories (VariableFactory or tuple of VariableFactory): VariableFactory to
            check against.

    Returns:
        bool: True if a match was found against one of the given VariableFactory.

    """
    if not isinstance(factories, tuple):
        factories = (factories,)

    for factory in factories:
        if (
            variable.rank == factory.rank
            and variable.name == factory.name
            and variable.units == factory.units
        ):
            return True
    return False


def get_variable_lags(var_factory):
    """Get the lags for a given VariableFactory.

    Args:
        var_factory (VariableFactory): VariableFactory to retrieve lags for.

    Returns:
        tuple of int: All possible lags corresponding to the given variable.


    """
    if var_factory in shifted_variables:
        return lags
    return (0,)


def get_shifted_variables(var_factory):
    """Get all possible shifted variables given a VariableFactory.

    Args:
        var_factory (VariableFactory): Basis for shifted Variable copies.

    Returns:
        tuple of Variable: All possible shifted variables.

    """
    shifted = []
    for lag in get_variable_lags(var_factory):
        shifted.append(var_factory[lag])
    return tuple(shifted)


DRY_DAY_PERIOD = VariableFactory(
    rank=1,
    name="Dry Day Period",
    units="days",
)
SWI = VariableFactory(
    rank=2,
    name="SWI(1)",
    units="$\mathrm{m}^3 \mathrm{m}^{-3}$",
)
MAX_TEMP = VariableFactory(
    rank=3,
    name="Max Temp",
    units="K",
)
DIURNAL_TEMP_RANGE = VariableFactory(
    rank=4,
    name="Diurnal Temp Range",
    units="K",
)
LIGHTNING = VariableFactory(
    rank=5,
    name="lightning",
    units="$\mathrm{strokes}\ \mathrm{km}^{-2}$",
)
PFT_CROP = VariableFactory(
    rank=6,
    name="pftCrop",
    units="1",
)
POPD = VariableFactory(
    rank=7,
    name="popd",
    units="$\mathrm{inh}\ \mathrm{km}^{-2}$",
)
PFT_HERB = VariableFactory(
    rank=8,
    name="pftHerb",
    units="1",
)
SHRUB_ALL = VariableFactory(
    rank=9,
    name="ShrubAll",
    units="1",
)
TREE_ALL = VariableFactory(
    rank=10,
    name="TreeAll",
    units="1",
)
AGB_TREE = VariableFactory(
    rank=11,
    name="AGB Tree",
    units="r$\mathrm{kg}\ \mathrm{m}^{-2}$",
)
VOD = VariableFactory(
    rank=12,
    name="VOD Ku-band",
    units="1",
)
FAPAR = VariableFactory(
    rank=13,
    name="FAPAR",
    units="1",
)
LAI = VariableFactory(
    rank=14,
    name="LAI",
    units="$\mathrm{m}^2\ \mathrm{m}^{-2}$",
)
SIF = VariableFactory(
    rank=15,
    name="SIF",
    units="r$\mathrm{mW}\ \mathrm{m}^{-2}\ \mathrm{sr}^{-1}\ \mathrm{nm}^{-1}$",
)

GFED4_BA = StandardVariable(rank=0, name="GFED4 BA", shift=0, units="1", parent=None)
MCD64CMQ_BA = StandardVariable(
    rank=-1, name="MCD64CMQ BA", shift=0, units="1", parent=None
)

# Investigated lags.
lags = (0, 1, 3, 6, 9, 12, 18, 24)
shifted_variables = (DRY_DAY_PERIOD, LAI, FAPAR, VOD, SIF)


# Data filling params.
filled_variables = (SWI, FAPAR, LAI, VOD, SIF)

# Season-trend & minima.
st_persistent_perc = 50
st_k = 4

# NN.
nn_n_months = 3


Category = Enum(
    "Category",
    [
        "METEOROLOGY",
        "HUMAN",
        "LANDCOVER",
        "VEGETATION",
    ],
)

feature_categories = immutabledict(
    {
        Category.METEOROLOGY: (
            DRY_DAY_PERIOD,
            SWI,
            MAX_TEMP,
            DIURNAL_TEMP_RANGE,
            LIGHTNING,
        ),
        Category.HUMAN: (PFT_CROP, POPD),
        Category.LANDCOVER: (PFT_HERB, SHRUB_ALL, TREE_ALL, AGB_TREE),
        Category.VEGETATION: (VOD, FAPAR, LAI, SIF),
    }
)

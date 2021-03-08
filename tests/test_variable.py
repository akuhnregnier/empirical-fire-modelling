# -*- coding: utf-8 -*-
from operator import methodcaller

import pytest

from empirical_fire_modelling import variable


@pytest.fixture
def dummy_variables():
    variables = []
    for rank, name, units in [(0, "foo", "1"), (1, "bar", "%")]:
        for shift in [0, 1, 3, 6, 9]:
            variables.append(
                variable.VariableFactory(rank=rank, name=name, units=units)[shift]
            )
    return tuple(variables)


def test_get_offset(dummy_variables):
    transformed = tuple(map(methodcaller("get_offset"), dummy_variables))
    assert transformed == dummy_variables


def test_get_matching(dummy_variables):
    assert (
        variable.get_matching(dummy_variables, single=False, name="foo")
        == variable.get_matching(dummy_variables, single=False, units="1")
        == variable.get_matching(dummy_variables, single=False, name="foo", units="1")
        == variable.get_matching(dummy_variables, single=False, units="1", name="foo")
        == dummy_variables[:5]
    )


def test_get_matching_single_exc(dummy_variables):
    with pytest.raises(RuntimeError):
        variable.get_matching(dummy_variables, name="foo")


def test_get_matching_strict_exc(dummy_variables):
    assert variable.get_matching(dummy_variables, strict=False, name="foobar") == ()
    with pytest.raises(RuntimeError):
        variable.get_matching(dummy_variables, name="foobar")


def test_instantiate_variable():
    kwargs = dict(rank=0, name="test", shift=0, units="1", parent=None)
    with pytest.raises(TypeError):
        variable.Variable(**kwargs)
    assert variable.StandardVariable(**kwargs)
    assert variable.OffsetVariable(**{**kwargs, **dict(shift=12)})


@pytest.mark.parametrize("shift", [12, 18, 24])
def test_get_offset(shift):
    kwargs = dict(rank=0, name="test", shift=shift, units="1", parent=None)
    assert variable.StandardVariable(**kwargs).get_offset() == variable.OffsetVariable(
        **kwargs
    )
    assert variable.StandardVariable(
        **kwargs
    ).get_offset().get_standard() == variable.StandardVariable(**kwargs)


def test_offset_variable_exc():
    with pytest.raises(ValueError):
        variable.OffsetVariable(rank=0, name="a", shift=0, units="1", parent=None)


def test_order_rank():
    assert variable.StandardVariable(
        rank=0, name="test", shift=0, units="1", parent=None
    ) < variable.StandardVariable(rank=1, name="test", shift=0, units="1", parent=None)


def test_order_name():
    assert variable.StandardVariable(
        rank=0, name="a", shift=0, units="1", parent=None
    ) == variable.StandardVariable(rank=0, name="b", shift=0, units="1", parent=None)


def test_order_shift():
    assert variable.StandardVariable(
        rank=0, name="a", shift=0, units="1", parent=None
    ) < variable.StandardVariable(rank=0, name="a", shift=1, units="1", parent=None)


def test_order_shift_offset():
    assert variable.OffsetVariable(
        rank=0, name="a", shift=12, units="1", parent=None
    ) < variable.OffsetVariable(rank=0, name="a", shift=18, units="1", parent=None)


def test_order_units():
    assert variable.StandardVariable(
        rank=0, name="a", shift=0, units="1", parent=None
    ) == variable.StandardVariable(rank=0, name="a", shift=0, units="2", parent=None)


def test_order_rank_shift():
    common_kwargs = dict(name="a", units="1", parent=None)
    assert variable.StandardVariable(
        rank=0, shift=0, **common_kwargs
    ) == variable.StandardVariable(rank=0, shift=0, **common_kwargs)
    assert variable.StandardVariable(
        rank=0, shift=0, **common_kwargs
    ) < variable.StandardVariable(rank=1, shift=0, **common_kwargs)
    assert variable.StandardVariable(
        rank=0, shift=0, **common_kwargs
    ) < variable.StandardVariable(rank=0, shift=1, **common_kwargs)
    assert variable.StandardVariable(
        rank=0, shift=0, **common_kwargs
    ) < variable.StandardVariable(rank=1, shift=1, **common_kwargs)

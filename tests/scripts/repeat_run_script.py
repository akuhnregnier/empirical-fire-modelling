# -*- coding: utf-8 -*-
from argparse import ArgumentParser

from joblib import Memory

from empirical_fire_modelling.cache import (
    DepMACache,
    cache,
    check_in_store,
    custom_get_hash,
    mark_dependency,
)

parser = ArgumentParser()
parser.add_argument("--tmp-dir", required=True)
parser.add_argument("--first-run", action="store_true")
args = parser.parse_args()

dummy_memory = Memory(args.tmp_dir, backend="custom")
ma_cache_inst = DepMACache(memory=dummy_memory, hash_func=custom_get_hash)
test_cache = cache(ma_cache_inst=ma_cache_inst)


@test_cache
@mark_dependency
def f0(x):
    return x + 1


@test_cache(dependencies=(f0,))
@mark_dependency
def f1(x):
    return f0(x) - 1


if args.first_run:
    assert f0(0) == 1
    assert f1(0) == 0
else:
    check_in_store(f0, 0)
    check_in_store(f1, 0)

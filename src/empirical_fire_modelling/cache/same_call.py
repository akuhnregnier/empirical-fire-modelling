# -*- coding: utf-8 -*-
"""Decorator guaranteeing uniform function calls."""
from inspect import Parameter, signature


def extract_uniform_args_kwargs(f, *args, **kwargs):
    """Extract uniform arguments given a function."""
    sig = signature(f)
    name_kind = {p.name: p.kind for p in sig.parameters.values()}

    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()

    # Possible argument types:
    #
    # KEYWORD_ONLY
    # POSITIONAL_ONLY
    # POSITIONAL_OR_KEYWORD
    # VAR_KEYWORD
    # VAR_POSITIONAL
    #
    # Accumulate POSITIONAL_ONLY, POSITIONAL_OR_KEYWORD, and VAR_POSITIONAL in the
    # order given in `arguments`.
    new_args = []
    pos_kind = (
        Parameter.POSITIONAL_ONLY,
        Parameter.POSITIONAL_OR_KEYWORD,
        Parameter.VAR_POSITIONAL,
    )
    for name, value in bound_args.arguments.items():
        if name_kind[name] not in pos_kind:
            break
        if name_kind[name] == Parameter.VAR_POSITIONAL:
            new_args.extend(value)
        else:
            new_args.append(value)

    # Accumulate KEYWORD_ONLY and VAR_KEYWORD in the
    # order given in `arguments`.
    new_kwargs = {}
    kw_kind = (Parameter.KEYWORD_ONLY, Parameter.VAR_KEYWORD)
    for name, value in bound_args.arguments.items():
        if name_kind[name] in pos_kind:
            continue
        assert name_kind[name] in kw_kind
        if name_kind[name] == Parameter.VAR_KEYWORD:
            new_kwargs.update(value)
        else:
            new_kwargs[name] = value

    return new_args, new_kwargs

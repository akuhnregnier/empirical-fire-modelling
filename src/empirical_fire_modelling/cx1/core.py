# -*- coding: utf-8 -*-
"""Run code on the cx1 cluster."""
import logging
import os
import random
import string
import sys
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from pprint import pformat, pprint
from subprocess import check_output

import cloudpickle
from jinja2 import Environment, FileSystemLoader
from wildfires.exceptions import NotCachedError

from ..configuration import Experiment
from ..exceptions import NoCX1Error
from ..utils import tqdm

logger = logging.getLogger(__name__)

__all__ = ("get_parsers", "run")

template_dir = Path(__file__).resolve().parent / "templates"


def get_parsers():
    """Parse command line arguments to determine where to run a function."""
    parser = ArgumentParser(description="Run a function either locally or on CX1")

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="output progress bar"
    )
    parser.add_argument(
        "-s", "--single", action="store_true", help="only run a single iteration"
    )
    parser.add_argument("--nargs", type=int, help="how many iterations to run")
    parser.add_argument(
        "--uncached", action="store_true", help="only run uncached calls"
    )
    parser.add_argument(
        "--experiment", help="select a specific experiment to run, e.g. 'ALL'"
    )
    parser.add_argument(
        "--list-experiments", action="store_true", help="list available experiments"
    )

    subparsers = parser.add_subparsers(
        help="execution target", dest="dest", required=True
    )
    local_parser = subparsers.add_parser("local", help="run functions locally")
    local_parser.add_argument(
        "-n",
        "--n-cores",
        default=1,
        type=int,
        help="number of cores to use in parallel (default: single threaded)",
    )
    mode_group = local_parser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument(
        "--threads", action="store_true", help="use threads (default)"
    )
    mode_group.add_argument("--processes", action="store_true", help="use processes")

    cx1_parser = subparsers.add_parser("cx1", help="run functions on CX1 using PBS")

    check_parser = subparsers.add_parser(
        "check", help="locally check which calls are cached"
    )

    return dict(
        parser=parser,
        subparsers=subparsers,
        local_parser=local_parser,
        mode_group=mode_group,
        cx1_parser=cx1_parser,
        check_parser=check_parser,
    )


def run_local(func, args, kwargs, backend="threads", n_cores=1, verbose=False):
    chosen_executor = {"threads": ThreadPoolExecutor, "processes": ProcessPoolExecutor}[
        backend
    ]
    out = []
    futures = []
    with chosen_executor(max_workers=n_cores) as executor:
        for single_args in zip(*args):
            futures.append(executor.submit(func, *single_args, **kwargs))

        # Progress bar (out of order).
        for future in tqdm(
            as_completed(futures),
            desc="Processing",
            total=len(futures),
            disable=not verbose,
        ):
            # Ensure exceptions are caught here already.
            future.result()

        # Collect results in order.
        for future in futures:
            out.append(future.result())
    return tuple(out)


def run_cx1(func, args, kwargs, cx1_kwargs, verbose):
    if cx1_kwargs is False:
        raise NoCX1Error("`cx1_kwargs` is `False`, but running on CX1 was requested.")
    if cx1_kwargs is None:
        cx1_kwargs = {}

    n_args = len(args[0])
    # Submit either a single job (if there is only one set of arguments, or an
    # array job for multiple arguments.
    job_template = (
        "array_job_script.sh.jinja2" if n_args > 1 else "job_script.sh.jinja2"
    )

    # Store temporary files, e.g. different input arguments, in the EPHEMERAL
    # directory so the jobs can access them later.
    ephemeral = Path(os.environ["EPHEMERAL"])
    if not ephemeral.is_dir():
        raise RuntimeError(f"Ephemeral directory {ephemeral} was not found.")

    job_id = "".join(random.sample(string.ascii_lowercase, 8))
    job_name = f"{func.__name__}_{job_id}"

    job_dir = ephemeral / job_name
    log_dir = job_dir / "pbs_output"

    # Create the necessary directories.
    log_dir.mkdir(parents=True, exist_ok=False)

    # Store the function with arguments for later retrieval by the job.
    bound_functions = [
        partial(func, *single_args, **kwargs) for single_args in zip(*args)
    ]

    bound_functions_file = job_dir / "bound_functions.pkl"
    with bound_functions_file.open("wb") as f:
        cloudpickle.dump(bound_functions, f, -1)

    # Render the Python script template.
    python_template_kwargs = dict(
        bound_functions_file=bound_functions_file,
        pythonpath=repr(list(map(os.path.abspath, sys.path))),
    )

    python_script = job_dir / "python_script.py"
    with python_script.open("w") as f:
        f.write(
            Environment(loader=FileSystemLoader(template_dir))
            .get_template("python_script.py.jinja2")
            .render(**python_template_kwargs)
        )

    # Render the job script template.
    if "walltime" not in cx1_kwargs:
        cx1_kwargs["walltime"] = "10:00:00"
    if "ncpus" not in cx1_kwargs:
        cx1_kwargs["ncpus"] = 1
    if "mem" not in cx1_kwargs:
        cx1_kwargs["mem"] = "5GB"

    job_template_kwargs = dict(
        job_name=job_name,
        executable=sys.executable,
        python_script=str(python_script),
        job_log_dir=log_dir,
        step=1,
        min_index=0,
        max_index=n_args - 1,  # This is inclusive (PBS).
        **cx1_kwargs,
    )

    job_script = job_dir / "job_script.sh"
    with job_script.open("w") as f:
        f.write(
            Environment(loader=FileSystemLoader(template_dir))
            .get_template(job_template)
            .render(**job_template_kwargs)
        )

    job_str = check_output(["qsub", str(job_script)]).decode().strip()
    print(f"Submitted job '{job_str}' with job name '{job_name}'.")


def run(
    func,
    *args,
    cx1_kwargs=None,
    get_parsers=get_parsers,
    return_local_args=False,
    **kwargs,
):
    """Run a function depending on given (including command line) arguments.

    Command line arguments will dictate if this function is run locally or as an
    (array) job on the CX1 cluster.

    The function should cache/save its output internally.

    Note that `args` and `kwargs` will be pickled along with `func` itself in order to
    facilitate running as PBS jobs on CX1. It may be more efficient to defer loading
    of input data such that this is only carried out within `func` itself, based on
    the input arguments given here.

    For checking the presence of cached data, the given function should accept a
    `cache_check` keyword argument that will be set to True when checking is desired.

    Note also that functions running on CX1 will not try to connect to a distributed
    Dask scheduler, instead spawning a local Client with $NCPUS threads.

    Args:
        func (callable): Function to be run.
        *args (length-N iterables): Function arguments. These will be zipped before
            being passed to the function.
        cx1_kwargs: Further specification of the type of job that will be used to
            run the function on the cluster. The following arguments are supported:
            'walltime', 'ncpus', and 'mem'. If `False` is given, trying to run on cx1
            will raise an error.
        get_parsers (callable): Callable that returns a dictionary containing (at
            least) a 'parser' key, which references the parser object used to parse
            command line arguments. This return value should support the following
            call (i.e. `get_parsers()['parser'].parse_args()`).
        return_local_args (bool):
            If True, return the arguments and kwargs along with the results (only
            applies for 'local').
        **kwargs: Function keyword arguments. These will be given identically to each
            function call, as opposed to `args`.

    Returns:
        tuple or None: The output results are returned if running locally. Otherwise,
            None is returned.

    Raises:
        NoCX1Error: If cx1_kwargs is `False` but running on CX1 was requested.

    """
    cmd_args = get_parsers()["parser"].parse_args()
    verbose = cmd_args.verbose
    single = cmd_args.single
    nargs = cmd_args.nargs
    kwargs = {**dict(single=single, nargs=nargs, verbose=verbose), **kwargs}

    if len(args) == 0 or len(args[0]) == 0:
        print("No args given.")
        return

    if cmd_args.experiment is not None:
        # Select all args matching the given experiment.
        selected_experiment = Experiment[cmd_args.experiment]
        # One of the args entries should contain Experiments.
        exp_arg_indices = [
            i for i in range(len(args)) if isinstance(args[i][0], Experiment)
        ]
        if len(exp_arg_indices) == 0:
            raise ValueError(
                "'--experiment' was given, but none of the args contains an "
                "Experiment."
            )
        if len(exp_arg_indices) > 1:
            raise ValueError(
                "'--experiment' was given, but more than one of the args contains "
                "an Experiment."
            )
        exp_arg_index = exp_arg_indices[0]
        if not all(isinstance(arg, Experiment) for arg in args[exp_arg_index]):
            raise ValueError(
                "'--experiment' was given, but the args starting with an Experiment "
                "contained other types too."
            )
        args = tuple(
            zip(
                *(
                    single_args
                    for single_args in zip(*args)
                    if single_args[exp_arg_index] == selected_experiment
                )
            )
        )

    if cmd_args.list_experiments:
        print("Available experiments:")
        for exp in Experiment:
            print(" -", exp.name)
        sys.exit(0)

    if single:
        # Only run a single iteration.
        args = list(args)
        args[0] = args[0][:1]  # Because zip() is used later on, this is sufficient.
        args = tuple(args)
    elif nargs:
        # Only run a limited number of iterations.
        args = list(args)
        args[0] = args[0][:nargs]  # Because zip() is used later on, this is sufficient.
        args = tuple(args)

    if cmd_args.dest == "check" or cmd_args.uncached:

        def check_in_store(func, *args, **kwargs):
            if hasattr(func, "check_in_store"):
                return func.check_in_store(*args, **kwargs)
            else:
                return func(*args, cache_check=True, **kwargs)

        # Check which calls are not yet cached. This relies on functions implementing
        # the `cache_check` keyword argument.
        checked = dict(present=[], uncached=[])
        uncached_args = []
        for single_args in tqdm(
            zip(*args), desc="Checking", total=len(args[0]), disable=not verbose
        ):
            try:
                check_in_store(func, *single_args, **kwargs)
                checked["present"].append((single_args, kwargs))
            except NotCachedError:
                checked["uncached"].append((single_args, kwargs))
                uncached_args.append(single_args)

        pprint({key: len(val) for key, val in checked.items()})

        if cmd_args.dest == "check":
            # Only checking was requested.
            logger.info(f"Cache status:\n{pformat(checked)}")
            logger.info(
                f"Number of uncached calls: {len(uncached_args)}/{len(args[0])}"
            )
            return checked
        # Otherwise, we want to run only the uncached calls.
        args = tuple(zip(*uncached_args))

    if cmd_args.dest == "local":
        out = run_local(
            func=func,
            args=args,
            kwargs=kwargs,
            backend=("processes" if cmd_args.processes else "threads"),
            n_cores=cmd_args.n_cores,
            verbose=verbose,
        )
        if return_local_args:
            return args, kwargs, out
        return out
    elif cmd_args.dest == "cx1":
        run_cx1(
            func=func, args=args, kwargs=kwargs, cx1_kwargs=cx1_kwargs, verbose=verbose
        )

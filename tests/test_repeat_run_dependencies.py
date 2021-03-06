# -*- coding: utf-8 -*-

import sys
from pathlib import Path
from subprocess import check_call

from .utils import *  # noqa


def test_repeat_run_dependencies(tmp_dir):
    common_args = [
        sys.executable,
        str(Path(__file__).resolve().parent / "scripts" / "repeat_run_script.py"),
        "--tmp-dir",
        str(tmp_dir),
    ]
    # Upon the first call, the cache entries are created.
    check_call(
        [
            *common_args,
            "--first-run",
        ]
    )
    # The second call should then be able to access these.
    check_call(common_args)

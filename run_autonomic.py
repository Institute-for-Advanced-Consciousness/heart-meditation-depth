# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""Autonomic variables as a function of reported depth.

Mean inter-beat interval, RMSSD, respiratory rate and respiration amplitude are computed on
the 60 s analysis windows of the silent blocks, regressed on depth within recording, and the
participant slopes are tested by sign-flipping permutation. The within-recording correlation
between depth and each variable is written alongside.

    python run_autonomic.py

Writes autonomic_depth_group.csv, autonomic_depth_within.csv and autonomic_depth_segments.csv
to <results>/autonomic/.
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'functions'))

import config                                                     # noqa: E402
import autonomic_depth                                        # noqa: E402

if __name__ == '__main__':
    sys.exit(autonomic_depth.main(['--data', config.DATA_DIR, '--out', config.AUTONOMIC_DIR]))

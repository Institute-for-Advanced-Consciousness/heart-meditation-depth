# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""session_length: the recording-length check shared by the heartbeat and theta analyses.

The segment files carry no recording duration, so the minimum-length gate (MIN_SESSION_MIN)
is applied to a lower bound: the last sample the extraction referred to, the end of the last
meditation block or of the last segment, whichever is later. The real recording is at least
this long, usually longer (post-meditation, cool-down and trailing rest come after it).
"""
import numpy as np


def session_duration_lowerbound_min(block_spans, seg_span, fs):
    """Lower bound on the recording length, in minutes (NaN when there is nothing to go on)."""
    ends = []
    bs = np.atleast_2d(np.asarray(block_spans))
    if bs.size:
        ends.append(float(bs[:, 1].max()))
    ss = np.atleast_2d(np.asarray(seg_span))
    if ss.size:
        ends.append(float(ss[:, 1].max()))
    if not ends:
        return np.nan
    return max(ends) / float(fs) / 60.0

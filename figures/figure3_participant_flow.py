# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""Figure 3: flow of participants, recordings and analysis segments.

Counts on the left are those retained at each stage, counts on the right the exclusions.
Silent-block depth reports are counted from the events of the raw BIDS dataset
(config.BIDS_DIR), each paired with the next confidence rating within nine events, as the
segment extraction pairs them; depth 0 is not a report. Recordings and segments come from the
cohort table of the primary run (perbeat_wise/main/cohort_sessions.csv). The numbers of
individuals screened and of pilot files excluded come from the study's screening record, which
is not part of the dataset, and are set below (N_SCREENED, N_PILOT). The script checks that
every recording of the dataset is accounted for as in the paper, and stops otherwise.

    python figures/figure3_participant_flow.py
"""
import csv
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'functions'))

import config                                                      # noqa: E402
import figure_style as F                                          # noqa: E402
from run_component_group_tests import run_root                     # noqa: E402

STEM = 'Figure_3_participant_flow'
N_SCREENED = 41
N_PILOT = 2                   # pilot files: corrupted, or recorded with 32 channels
MIN_CONF, MAX_PAIR_LAG = 4, 9
INK = '#1A1A1A'
RULE = 0.7


def participants():
    """Participant folders of the segment data (BIDS labels), in order."""
    return sorted(d for d in os.listdir(config.DATA_DIR)
                  if d.startswith('sub-') and os.path.isdir(os.path.join(config.DATA_DIR, d)))


def silent_reports(sub, ses):
    """[(depth, confidence or None, onset s)] of the silent-block depth reports of one
    recording, and the time of its last event in minutes; (None, None) without events."""
    p = os.path.join(config.BIDS_DIR, sub, 'ses-%d' % ses, 'eeg',
                     '%s_ses-%d_task-meditation_events.tsv' % (sub, ses))
    if not os.path.isfile(p):
        return None, None
    rows = list(csv.DictReader(io.open(p, encoding='utf-8'), delimiter='\t'))
    out = []
    for i, r in enumerate(rows):
        if r['trial_type'] != 'depth_rating' or not r['block'].startswith('silent'):
            continue
        d = int(float(r['depth']))
        if d < 1:
            continue
        c = None
        for q in rows[i + 1:i + 1 + MAX_PAIR_LAG]:
            if q['trial_type'] == 'confidence_rating':
                c = int(float(q['confidence']))
                break
        out.append((min(d, 5), c, float(r['onset'])))
    return out, max(float(r['onset']) for r in rows) / 60.0


def flow_table():
    """One row per recording: reports, and the recording's fate in the primary run."""
    if not os.path.isdir(config.BIDS_DIR):
        raise SystemExit('raw BIDS dataset not found at %s (set HMD_BIDS)' % config.BIDS_DIR)
    main =os.path.join(run_root(config.COMPONENT_ROOT, config.COMPONENT_SETTINGS[config.PRIMARY]),
                        'perbeat_wise', 'main')
    run = {(r['subject'], int(r['session'])): r for r in csv.DictReader(
        io.open(os.path.join(main, 'cohort_sessions.csv'), encoding='utf-8-sig'))}
    ids = participants()
    rows = []
    for sid in ids:
        for ses in (1, 2):
            rep, last = silent_reports(sid, ses)
            if rep is None:
                continue
            r = run.get((sid, ses), {})
            rows.append(dict(
                subject=sid, session=ses, last_event_min=last, reports_silent=len(rep),
                unpaired=sum(1 for _d, c, _t in rep if c is None),
                reports_conf4=sum(1 for _d, c, _t in rep if c is not None and c >= MIN_CONF),
                run_status=r.get('status', 'not in run'), run_reason=r.get('reason', ''),
                run_n_seg_cond=r.get('n_seg_cond', ''), run_n_seg_used=r.get('n_seg_used', '')))
    return rows


def num(r, k):
    try:
        return int(float(r[k]))
    except (TypeError, ValueError):
        return 0


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle

    rows = flow_table()
    kept = [r for r in rows if r['run_status'] == 'kept']
    # under the length threshold: absent from the run (never preprocessed), or dropped by its
    # MIN_SESSION_MIN gate
    min_len = float(config.COMPONENT_SETTINGS[config.PRIMARY].get('MIN_SESSION_MIN') or 0)
    trim = config.COMPONENT_SETTINGS[config.PRIMARY].get('OVERLAP_RULE', 'drop') == 'trim'
    min_keep = float(config.COMPONENT_SETTINGS[config.PRIMARY].get('OVERLAP_MIN_S', 30))
    short = [r for r in rows if r['run_status'] == 'not in run'
             or 'MIN_SESSION_MIN' in r['run_reason']]
    depths = [r for r in rows if 'MIN_DEPTHS' in r['run_reason']]

    n_preproc = N_SCREENED - N_PILOT
    n_subj = len({r['subject'] for r in rows})
    n_rep = sum(num(r, 'reports_silent') for r in rows)
    n_conf = sum(num(r, 'reports_conf4') for r in rows)
    n_below = n_rep - n_conf
    n_conf_kept = sum(num(r, 'reports_conf4') for r in kept)
    n_win = sum(num(r, 'run_n_seg_cond') for r in kept)
    n_nowin = n_conf_kept - n_win
    n_used = sum(num(r, 'run_n_seg_used') for r in kept)
    n_overlap = n_win - n_used
    per_subj = {}
    for r in kept:
        per_subj[r['subject']] = per_subj.get(r['subject'], 0) + 1
    n_two = sum(1 for v in per_subj.values() if v == 2)
    n_one = sum(1 for v in per_subj.values() if v == 1)

    # the chain must close
    assert n_subj == n_preproc and len(rows) == 2 * n_subj
    assert len(kept) + len(short) + len(depths) == len(rows), 'unaccounted recording'
    assert all(float(r['last_event_min']) < max(min_len, 45) for r in short)
    assert all(num(r, 'unpaired') == 0 for r in rows)
    assert n_conf_kept + sum(num(r, 'reports_conf4') for r in short + depths) == n_conf
    assert len(per_subj) == n_subj and 2 * n_two + n_one == len(kept)
    assert n_below >= 0 and n_nowin >= 0 and n_overlap >= 0

    # (retained, excluded on the way into this stage)
    steps = [
        ('Screened\nn = %d' % N_SCREENED, None),
        ('EEG preprocessed\n%d participants, %d recordings\n'
         '%d depth reports, Silent blocks' % (n_preproc, len(rows), n_rep),
         'Pilot files not carried forward  (n = %d)\n'
         'corrupted, or recorded with 32 channels' % N_PILOT),
        ('Reports at confidence ≥ 4\nn = %d  (%d participants)' % (n_conf, n_subj),
         'Reports with confidence below 4  (n = %d)' % n_below),
        ('Recordings entering analysis\n%d recordings, %d segments (one per report)\n'
         '%d participants with two, %d with one' % (len(kept), n_win, n_two, n_one),
         'Recordings excluded  (n = %d; %d reports)\n'
         '%d with < 2 distinct depth levels\n'
         % (len(short) + len(depths), sum(num(r, 'reports_conf4') for r in short + depths),
            len(depths))
         + ('%d shorter than %g min\n' % (len(short), min_len) if short else '')
         + 'Reports with no usable segment in the cleaned data  (n = %d)' % n_nowin),
        ('Segments analysed\nn = %d  (%d participants)' % (n_used, n_subj),
         ('Segments removed because they overlapped the next\nwith less than %g s of their '
          'own left  (n = %d)' % (min_keep, n_overlap)) if trim else
         ('Segments removed because they overlapped\nthe next segment  (n = %d segments)'
          % n_overlap)),
    ]

    with plt.rc_context(F.PAPER_RC):
        fig, ax = plt.subplots(figsize=(6.85, 5.4), dpi=600)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.axis('off')
        n = len(steps)
        top, bot = 94.0, 6.0
        pitch = (top - bot) / (n - 1)
        xm, wm = 30.0, 40.0        # retained column: centre and width
        xd, wd = 76.0, 40.0        # exclusion column

        def hmain(txt):
            return 6.0 + 3.2 * txt.count('\n')

        def box(cx, cy, w, txt, h):
            ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fill=False, edgecolor=INK,
                                   linewidth=RULE, zorder=2))
            ax.text(cx, cy, txt, ha='center', va='center', fontsize=6.0, color=INK,
                    linespacing=1.45, zorder=3)

        def arrow(x0, y0, x1, y1):
            ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=6,
                                         linewidth=RULE, color=INK, shrinkA=0, shrinkB=0,
                                         zorder=1))

        for i, (main_txt, drop_txt) in enumerate(steps):
            y = top - i * pitch
            h = hmain(main_txt)
            box(xm, y, wm, main_txt, h)
            if i < n - 1:
                y_next = top - (i + 1) * pitch
                arrow(xm, y - h / 2, xm, y_next + hmain(steps[i + 1][0]) / 2)
            # the exclusion sits on the arrow into this stage
            if drop_txt and i > 0:
                y_prev = top - (i - 1) * pitch
                y_mid = (y + h / 2 + y_prev - hmain(steps[i - 1][0]) / 2) / 2
                hd = 5.4 + 3.2 * drop_txt.count('\n')
                box(xd, y_mid, wd, drop_txt, hd)
                arrow(xm, y_mid, xd - wd / 2, y_mid)

        fig.tight_layout(pad=0.2)
        os.makedirs(config.FIGURES_DIR, exist_ok=True)
        for ext in ('pdf', 'png'):
            p = os.path.join(config.FIGURES_DIR, '%s.%s' % (STEM, ext))
            fig.savefig(p, dpi=600, bbox_inches='tight')
            print('wrote %s' % p)
        plt.close(fig)

    print('screened %d, preprocessed %d, analysed %d participants'
          % (N_SCREENED, n_preproc, n_subj))
    print('reports %d, at confidence >= %d %d, in kept recordings %d, windows %d, segments %d'
          % (n_rep, MIN_CONF, n_conf, n_conf_kept, n_win, n_used))
    print('recordings %d, analysed %d (%d participants with two, %d with one)'
          % (len(rows), len(kept), n_two, n_one))


if __name__ == '__main__':
    main()

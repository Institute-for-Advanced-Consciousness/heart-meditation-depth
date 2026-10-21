# Copyright (C) 2026 Institute for Advanced Consciousness Studies
# Author: Cedric Cannard
# SPDX-License-Identifier: GPL-3.0-or-later
"""The descriptive numbers of the paper's Participants, Methods and cohort paragraphs.

Prints, for the participants of the segment data (config.DATA_DIR):

  demographics    age, sex, years of practice, retreat days, meditation frequency and typical
                  session length (participants.tsv of the BIDS dataset)
  preprocessing   interpolated electrodes, components per decomposition, ASR, ECG artifact
                  share and recording duration (config.PREPROCESSING_SUMMARY)
  cohort          recordings kept and dropped, inter-beat-interval and cardiac-field
                  diagnostics, segments, heartbeats and depth-level coverage (the primary run)
  reports         silent-block depth reports, the confidence floor, the depth distribution and
                  the gap between consecutive confident reports (raw BIDS events,
                  config.BIDS_DIR)
  autonomic       autonomic variables per segment and their within-recording correlation with
                  depth (run_autonomic.py)

A section whose input is absent is skipped. Writes nothing; the heartbeat counts come from the
cached model-adjusted level means of the primary run (built on first use).

    python analysis/cohort_numbers.py
"""
import csv
import io
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'functions'))
sys.path.insert(0, os.path.join(ROOT, 'figures'))

import config                                                      # noqa: E402
import adjusted_depth_means as AD                                  # noqa: E402
import outcomes as OC                                          # noqa: E402
from figure3_participant_flow import participants, silent_reports  # noqa: E402
from heart_depth_run import DEFAULT_RUN_PARAMS                 # noqa: E402
from heart_depth_pipeline import window_indices                    # noqa: E402
from run_component_group_tests import run_root                     # noqa: E402

SILENT, DEPTHS = 0, (1, 2, 3, 4, 5)


def ms(v, nd=2):
    v = np.asarray(v, float)
    return '%.*f +/- %.*f (range %.*f to %.*f, n %d)' % (nd, v.mean(), nd, v.std(ddof=1),
                                                         nd, v.min(), nd, v.max(), v.size)


def read_csv(fp):
    return list(csv.DictReader(io.open(fp, encoding='utf-8-sig')))


def demographics(ids):
    fp = OC.participants_file(config.BEHAVIOUR_DIR)
    if not os.path.isfile(fp):
        print('\nDEMOGRAPHICS skipped: %s not found' % fp)
        return
    scr = {r['participant_id'].strip(): r
           for r in csv.DictReader(io.open(fp, encoding='utf-8-sig'), delimiter='\t')}
    d = [scr[i] for i in ids if i in scr]
    print('\nDEMOGRAPHICS: %d of %d participants in participants.tsv' % (len(d), len(ids)))

    def col(k):
        out = []
        for r in d:
            try:
                out.append(float(r[k]))
            except (ValueError, KeyError):
                out.append(np.nan)
        return np.array(out)
    for k in ('age', 'meditation_years_experience', 'retreat_days', 'meditation_frequency'):
        v = col(k)
        v = v[np.isfinite(v)]
        print('  %-28s %s, median %.1f' % (k, ms(v, 1), np.median(v)))
    sex = [r['sex'] for r in d]
    print('  %-28s %s' % ('sex', {k: sex.count(k) for k in sorted(set(sex))}))
    v = col('meditation_length')
    print('  %-28s counts by code %s' % ('meditation_length', {
        int(x): int((v == x).sum()) for x in np.unique(v[np.isfinite(v)])}))


def preprocessing(ids):
    import scipy.io as sio
    files = [f for f in config.PREPROCESSING_SUMMARY.split(os.pathsep) if f]
    if not files or not all(os.path.isfile(f) for f in files):
        print('\nPREPROCESSING skipped: %s not found' % config.PREPROCESSING_SUMMARY)
        return
    ses = []
    for f in files:
        m = sio.loadmat(f, squeeze_me=True, struct_as_record=False)
        for e in np.atleast_1d(m['sInfo']):
            for k, s in enumerate(np.atleast_1d(e.session), 1):
                ses.append((str(e.subject), k, s))
    ses = [x for x in ses if x[0] in ids]
    ok = [x for x in ses if str(getattr(x[2], 'status', '')) == 'ok']
    print('\nPREPROCESSING: %d recordings of the analysed participants, %d with status ok'
          % (len(ses), len(ok)))

    def g(name, rows=ok):
        return np.array([float(np.size(getattr(s, name)) if name == 'badChan'
                               else getattr(s, name)) for _a, _b, s in rows])
    nbad, ngood = g('nBadComp'), g('nGoodComp')
    ncomp = nbad + ngood
    bc = g('badChan')
    print('  electrodes interpolated     %s; recordings with none: %d' % (ms(bc), (bc == 0).sum()))
    for k_ in (1, 2):
        print('     session %d:               %s' % (k_, ms(g('badChan', [x for x in ok
                                                                      if x[1] == k_]))))
    print('  components per decomposition %s; full rank 64 in %d of %d'
          % (ms(ncomp), (ncomp == 64).sum(), ncomp.size))
    print('  components removed          ', ms(nbad))
    print('  ASR on the ICA copy, %%      ', ms(g('ASR1_pct')))
    print('  ASR on the broadband copy, %%', ms(g('ASR2_pct')))
    print('  ECG flagged, %%              ', ms(g('ECG_pct_bad')))
    print('  duration after cleaning, min', ms(g('dur_clean_min'), 1))


def cohort(root):
    main = os.path.join(root, 'perbeat_wise', 'main')
    if not os.path.isfile(os.path.join(main, 'cohort_sessions.csv')):
        print('\nCOHORT skipped: %s not found (run run_component_analysis.py)'
              % os.path.join(main, 'cohort_sessions.csv'))
        return
    run = read_csv(os.path.join(main, 'cohort_sessions.csv'))
    kept = [r for r in run if r['status'] == 'kept']
    print('\nCOHORT: %d recordings, %d analysed, %d participants'
          % (len(run), len(kept), len({r['subject'] for r in kept})))
    for r in run:
        if r['status'] != 'kept':
            print('  dropped: %s' % r['reason'])
    ib = np.array([float(r['ibi_short_prop']) for r in kept]) * 100
    cr = np.array([float(r['ecg_clean_ratio']) for r in kept]) * 100
    cf = np.array([float(r['cfa_topo_r2']) for r in kept])
    nseg = np.array([int(r['n_seg_used']) for r in kept])
    print('  inter-beat intervals < 600 ms, %%: median %.2f, mean %.2f, SD %.2f; recordings > 0: '
          '%d; > 10%%: %d; largest %.1f' % (np.median(ib), ib.mean(), ib.std(ddof=1),
                                           (ib > 0).sum(), (ib > 10).sum(), ib.max()))
    by = {}
    for r in kept:
        by.setdefault(r['subject'], []).append(float(r['ecg_clean_ratio']) * 100)
    print('  heartbeats rejected by inter-beat-interval cleaning: none in %d recordings; largest '
          '%.1f%%; largest participant mean %.1f%%'
          % ((cr == 0).sum(), cr.max(), max(np.mean(v) for v in by.values())))
    print('  R2 between component and R-peak scalp field: median %.2f, IQR %.2f to %.2f; >= 0.5 '
          'in %d (%.0f%%)' % (np.median(cf), np.percentile(cf, 25), np.percentile(cf, 75),
                              (cf >= .5).sum(), 100 * (cf >= .5).mean()))
    ps, nrec = {}, {}
    for r in kept:
        ps[r['subject']] = ps.get(r['subject'], 0) + int(r['n_seg_used'])
        nrec[r['subject']] = nrec.get(r['subject'], 0) + 1
    v = np.array(list(ps.values()))
    print('  segments %d: per recording mean %.1f, median %.0f, range %d to %d; per participant '
          'mean %.1f, median %.0f, range %d to %d' % (nseg.sum(), nseg.mean(), np.median(nseg),
                                                      nseg.min(), nseg.max(), v.mean(),
                                                      np.median(v), v.min(), v.max()))
    print('  participants with two analysed recordings %d, with one %d'
          % (sum(1 for x in nrec.values() if x == 2), sum(1 for x in nrec.values() if x == 1)))
    cnt = np.asarray(AD.load(main, config.DATA_DIR)['cnt'])
    lv = (cnt > 0).sum(1)
    print('  heartbeats %d (%.1f per segment); participants per depth level %s; participants '
          'reporting 2, 3, 4, 5 levels: %s' % (cnt.sum(), cnt.sum() / nseg.sum(),
                                               (cnt > 0).sum(0).tolist(),
                                               [int((lv == k).sum()) for k in (2, 3, 4, 5)]))

    # segments per depth level, recounted from the segment files with the overlap rule
    prm = dict(DEFAULT_RUN_PARAMS, **config.COMPONENT_SETTINGS[config.PRIMARY])
    tot, differ = np.zeros(len(DEPTHS), int), 0
    for r in kept:
        with np.load(os.path.join(config.DATA_DIR, r['subject'], 'ses%s.npz' % r['session']),
                     allow_pickle=True) as z:
            fs = float(z['fs'])
            dep = np.asarray(z['seg_depth']).astype(int)
            cnd = np.asarray(z['seg_cond']).astype(int)
            span = np.atleast_2d(np.asarray(z['seg_span'])).astype(np.int64)
            seg_len = np.asarray(z['seg_len'])
        # the pipeline's own window and overlap rule (heart_depth_pipeline)
        w0, w1, w_ok = window_indices(seg_len, fs, prm['EPOCH_DUR'], prm['PRE_END'])
        keep = (cnd == SILENT) & np.isin(dep, DEPTHS) & w_ok
        ix = np.where(keep)[0]
        ix = ix[np.argsort(span[ix, 0])]
        st = span[ix, 0] + w0
        en = span[ix, 0] + w1
        trim = prm.get('OVERLAP_RULE', 'drop') == 'trim'
        drop = [ix[a] for a in range(len(ix) - 1) if en[a] > st[a + 1]
                and not (trim and st[a + 1] - st[a] >= float(prm.get('OVERLAP_MIN_S', 30)) * fs)]
        keep[np.asarray(drop, int)] = False
        differ += int(keep.sum()) != int(r['n_seg_used'])
        for k, dv in enumerate(DEPTHS):
            tot[k] += int((dep[keep] == dv).sum())
    print('  segments at depths 1 to 5: %s (sum %d); recordings where the recount differs from '
          'the run: %d' % (tot.tolist(), tot.sum(), differ))


def reports(ids):
    if not os.path.isdir(config.BIDS_DIR):
        print('\nREPORTS skipped: %s not found' % config.BIDS_DIR)
        return
    allr, gaps, n_files = [], [], 0
    for sid in ids:
        for ses in (1, 2):
            rep, _last = silent_reports(sid, ses)
            if rep is None:
                continue
            n_files += 1
            allr += [(d, c) for d, c, _t in rep]
            gaps += np.diff(sorted(t for _d, c, t in rep if c is not None and c >= 4)).tolist()
    dall = np.array([a for a, _c in allr])
    c = np.array([-1 if b is None else b for _a, b in allr])
    print('\nREPORTS (silent blocks, %d recordings): %d; confidence >= 4: %d (%.1f%%); below 4: '
          '%d; unpaired %d' % (n_files, dall.size, (c >= 4).sum(), 100 * (c >= 4).mean(),
                               ((c >= 0) & (c < 4)).sum(), (c < 0).sum()))
    print('  depth distribution, all reports:       %s'
          % ', '.join('%.1f%%' % (100 * (dall == k).mean()) for k in DEPTHS))
    d4 = dall[c >= 4]
    print('  depth distribution, confidence >= 4:   %s'
          % ', '.join('%.1f%%' % (100 * (d4 == k).mean()) for k in DEPTHS))
    gp = np.array(gaps)
    print('  gap between consecutive confident reports of a recording: median %.0f s, IQR %.0f '
          'to %.0f; >= 63 s in %.0f%%' % (np.median(gp), np.percentile(gp, 25),
                                          np.percentile(gp, 75), 100 * (gp >= 63).mean()))


def autonomic():
    seg = os.path.join(config.AUTONOMIC_DIR, 'autonomic_depth_segments.csv')
    if not os.path.isfile(seg):
        print('\nAUTONOMIC skipped: %s not found' % seg)
        return
    a = read_csv(seg)
    print('\nAUTONOMIC: %d segments, %d recordings, %d participants'
          % (len(a), len({(r['Subject'], r['Session']) for r in a}),
             len({r['Subject'] for r in a})))
    for k in a[0]:
        if k in ('Subject', 'Session', 'Depth'):
            continue
        v = np.array([float(r[k]) for r in a])
        print('  %-10s median %.2f, IQR %.2f to %.2f'
              % (k, np.nanmedian(v), np.nanpercentile(v, 25), np.nanpercentile(v, 75)))
    w = read_csv(os.path.join(config.AUTONOMIC_DIR, 'autonomic_depth_within.csv'))
    for var in dict.fromkeys(r['Variable'] for r in w):
        rr = np.array([float(r['r']) for r in w
                       if r['Variable'] == var and r['r'] not in ('', 'nan')])
        rr = rr[np.isfinite(rr)]
        print('  within-recording r(depth, %s): median |r| %.2f, mean r %+.2f (%d recordings)'
              % (var, np.median(np.abs(rr)), rr.mean(), rr.size))


def main():
    ids = participants()
    root = run_root(config.COMPONENT_ROOT, config.COMPONENT_SETTINGS[config.PRIMARY])
    print('%d participants in %s' % (len(ids), config.DATA_DIR))
    demographics(ids)
    preprocessing(ids)
    cohort(root)
    reports(ids)
    autonomic()


if __name__ == '__main__':
    main()

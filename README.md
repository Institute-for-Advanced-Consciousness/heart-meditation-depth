# heart-meditation-depth

Code for the paper

> Cannard, C., Nath, M., Becattini, V., Laukkonen, R. E., Ganesan, S., Hanley, A., Holecek, A.,
> Simonian, N., Lifshitz, M., Schoeller, F., Cahn, B. R., Khalsa, S., Sacchet, M. D., and
> Reggente, N. Heartbeat-related spectral power, but not evoked amplitude or phase consistency,
> tracks self-reported meditative depth.

It reproduces the analyses, figures and tables of the paper from the public dataset: the
preprocessing (MATLAB, EEGLAB), the segment extraction and all statistics (Python). The steps
are, in order:

1. download the dataset;
2. install MATLAB, EEGLAB, BrainBeats and the Python packages;
3. preprocess the recordings (MATLAB);
4. extract the analysis segments (Python);
5. run the analyses: participant-level depth slopes (L1), then the group tests (L2);
6. draw the figures;
7. print the numbers quoted in the text.

The phase measure is called HRPC (heartbeat-related phase consistency) in the paper and `hepc` in
the code and its outputs.

## 1. Data

The recordings, events, questionnaires and participant information are on OpenNeuro:
[ds008836, version 1.0.2](https://doi.org/10.18112/openneuro.ds008836.v1.0.2) (about 70 GB).
Download the whole dataset into `data/bids/` inside this repository, for example with
[openneuro-py](https://github.com/hoechenberger/openneuro-py):

    pip install openneuro-py
    openneuro-py download --dataset ds008836 --tag 1.0.2 --target-dir data/bids

or with DataLad (`datalad install https://github.com/OpenNeuroDatasets/ds008836.git`, then
`datalad get .`).

One table is not in the dataset and ships with this repository:
`metadata/eeg_viability_ratings.tsv`, the experimenters' rating of each participant's EEG from
the session notes, which defines the participant subsets of Supplementary Table 2.

Disk space: about 70 GB for the dataset, 65 GB for the preprocessed recordings, 12 GB for the
segment files and 10 GB for the results.

## 2. Requirements

Preprocessing:

- MATLAB with the Signal Processing Toolbox and the Statistics and Machine Learning Toolbox
  (checked with R2026a)
- [EEGLAB](https://eeglab.org) with the plugins clean_rawdata (2.11), ICLabel (1.7), firfilt
  (2.8) and dipfit (5.6)
- [BrainBeats](https://github.com/amisepa/BrainBeats) from GitHub, main branch at commit
  `15b0d96` or later (`git clone https://github.com/amisepa/BrainBeats`), on the MATLAB path.
  The release in the EEGLAB plugin manager (1.5) does not contain `apply_car` and
  `detect_ecg_artifacts`, which the preprocessing calls.

Analyses: Python 3.12 and the packages in `requirements.txt`:

    pip install -r requirements.txt

The pinned versions are those under which the paper's numbers were reproduced; other versions can
change the last digits. The source localization (Figure 6) also needs:

- the AAL atlas of FieldTrip (`template/atlas/aal`, found inside EEGLAB's Fieldtrip-lite plugin
  under `~/Documents/MATLAB/eeglab*/plugins/Fieldtrip-lite*`, or point `HEP_AAL_DIR` at it);
- MNE's `fsaverage` subject, which MNE downloads on first use (internet access; stored in
  `~/mne_data`);
- for the anatomical background of Figure 6, the MNI152 T1 of EEGLAB's dipfit plugin
  (`standard_BEM/standard_mri.mat`, found through `EEGLAB_PATH` or `HEP_MNI_MRI`); without it the
  figure is drawn without anatomy.

## Paths

All Python scripts are run from the repository folder. By default every input is read from
`data/` and every output written to `results/`, both inside the repository; these environment
variables change that:

| Variable | Default | What it is |
|---|---|---|
| `HMD_BIDS` | `data/bids` | the BIDS dataset (read by the Python steps) |
| `HMD_BEHAVIOUR` | `<HMD_BIDS>/phenotype` | the questionnaire tables; `participants.tsv` is read from the folder above it |
| `HMD_PREPROCESSED` | `data/preprocessed` | the preprocessed recordings, read by the segment extraction |
| `HMD_PREPROCESSING_SUMMARY` | `data/preprocessed/sInfo.mat` | the preprocessing summary, read by `analysis/cohort_numbers.py` |
| `HMD_DATA` | `data/segments` | the segment files |
| `HMD_ICLABELS` | `<HMD_DATA>/ic_labels.npz` | the ICLabel probabilities |
| `HMD_RESULTS` | `results` | every output |
| `HMD_VIABILITY` | `metadata/eeg_viability_ratings.tsv` | the EEG viability ratings |
| `HEP_AAL_DIR`, `EEGLAB_PATH`, `HEP_MNI_MRI` | see above | atlas and MRI for the source localization |

The MATLAB step does not read these variables; it takes `bids_path` and `out_path` (below).

## 3. Preprocessing (MATLAB)

Start MATLAB in the repository folder, put EEGLAB and BrainBeats on the path, and run:

    bids_path = fullfile(pwd, 'data', 'bids');
    out_path  = fullfile(pwd, 'data', 'preprocessed');
    run('preprocessing/process_raw_data.m')

(`bids_path` and `out_path` default to these two folders when they are not set.) For each of the
78 recordings this resamples to 250 Hz, filters, rejects and interpolates bad channels, runs
artifact subspace reconstruction and extended Infomax ICA, removes the components ICLabel assigns
to muscle or eye, and removes the samples where the ECG is contaminated. It writes one EEGLAB file
per recording (`<participant>/<session>/<participant>_ses<session>.set`) and a summary,
`sInfo.mat`, saved after every recording. A recording whose ECG is contaminated in more than 20%
of its samples gets no file (the summary marks it `ecg_bad`). The ICA starts from random weights,
so a rerun matches the paper's preprocessed files closely but not exactly; the other steps are
deterministic.

## 4. Segment extraction (Python)

    python functions/extract_npz.py --data data/preprocessed --out data/segments

(or set `HMD_PREPROCESSED` and `HMD_DATA` and omit the two options). This writes one file per
recording, `data/segments/<participant>/ses<session>.npz`, with the 90 s before each depth
report, the four recording periods, the ECG and respiration and the ICA decomposition, and one
file with the ICLabel probabilities of every component, `data/segments/ic_labels.npz`.

## 5. Analyses

Run the scripts in this order; each script's header lists its steps and outputs. "L1" is the
participant level (each participant's depth slope at every time and frequency point), "L2" the
group level (weighted threshold-free cluster enhancement, TFCE).

| Order | Script | What it computes | Needs |
|---|---|---|---|
| 1 | `run_component_analysis.py` | Analysis 2, L1: one heartbeat-related component per recording, the three measures, the depth slope maps (eight settings and two confound runs) | step 4 |
| 2 | `run_component_group_tests.py` | Analysis 2, L2, in steps: `tfce`, `confounds`, `loso` (leave one participant out), `surrogate` (shifted heartbeats), `subsets` (EEG viability), `sources` (source localization), `ongoing` (ongoing beta power) | 1 |
| 3 | `run_scalp_analysis.py` | Analysis 1, L1 and L2: HEP, HRSP and HRPC at all 64 electrodes | step 4 |
| 4 | `run_fmt_depth.py` | Analysis 3: frontal-midline theta against depth, and its associations with the self-reports | step 4 |
| 5 | `run_period_contrasts.py` | Analysis 4: differences between baseline, pre-meditation, post-meditation and the oddball task | step 4 |
| 6 | `run_autonomic.py` | inter-beat interval, RMSSD and respiration against depth | step 4 |
| 7 | `run_associations.py` | associations of the primary depth effect with the self-reports and demographics | 1 and 2 (`tfce`) |

Scripts 3 to 6 do not depend on one another or on 1 and 2. Most of the time goes into the
permutation tests. `run_component_analysis.py`, `run_scalp_analysis.py` and
`run_period_contrasts.py` take `--limit N` to run on the first N participants as a quick check
(for `run_component_group_tests.py` it limits only the `surrogate` step). A limited run writes
into the same result folders as a full one, so set `HMD_RESULTS` to another folder for it.

Where the paper's tables come from:

| In the paper | File |
|---|---|
| primary result (Figure 5) | `results/component/HEART_DEPTH/<primary setting>/perbeat_wise/main/l2_tfce_summary_session.csv` |
| Supplementary Table 1 | `results/figures/Supplementary_Table_1_confounds.csv`, written by `figures/supplementary_figure2_confounds.py` |
| Supplementary Table 2 | `results/component/HEART_DEPTH/<primary setting>/viability_subsets.csv` |
| Supplementary Table 3 | `results/associations/outcome_skipped_correlations.csv` |
| Supplementary Table 4 | `results/component/HEART_DEPTH/<primary setting>/perbeat_wise/outcomes_demographics_tfce/*_demographics.csv` (the other tables in that folder are not reported) |
| Supplementary Table 5 | `results/fmt/fmt_outcome_skipped_correlations.csv` |
| Supplementary Table 6 | `results/periods/HEART_PERIOD/<run>/period_tfce_all.csv` |

The primary setting's folder is `lee-hrsp-raw_1IC_dur60_bl_nosapo` (selection by the method of
Lee et al., 2024, on raw HRSP; one component; 60 s windows; baseline correction; silent blocks).

## 6. Figures

`python figures/<name>.py` writes into `results/figures/`, as PDF and PNG (Supplementary Figure 3
as PNG only):

| Script | Draws | Needs |
|---|---|---|
| `figure3_participant_flow.py` | Figure 3 | the BIDS events and analysis 1 |
| `figure4_scalp.py` | Figure 4 and Supplementary Figure 1 | analysis 3, and analysis 2 `tfce` (the span outlined on the maps) |
| `figure5_primary.py` | Figure 5 | analyses 1 and 2 `tfce` |
| `figure6_sources.py` | Figure 6 | analysis 2 `sources` |
| `figure7_theta.py` | Figure 7 | analysis 4 |
| `supplementary_figure2_confounds.py` | Supplementary Figure 2 and Table 1 | analysis 2 `confounds` |
| `supplementary_figure3_questionnaires.py` | Supplementary Figure 3 | analysis 7 |
| `supplementary_figure4_periods.py` | Supplementary Figure 4 | analysis 5 |

Figures 1 and 2 (the design and the setup) are not data figures. The numbers of individuals
screened (41) and of pilot recordings excluded (2) in Figure 3 come from the study's screening
record, which is not part of the dataset, and are written in the script.

## 7. Numbers quoted in the text

`analysis/` prints them: `cohort_numbers.py` (participants, recordings, reports and segments at
each step), `ecg_polarity.py` (recordings with an inverted ECG) and `per_trial_normalisation.py`
(the comparison with the per-trial normalisation of Lee et al., 2024, in the Methods).

## Layout

    preprocessing/   MATLAB preprocessing and its helper functions
    metadata/        the EEG viability ratings (Supplementary Table 2)
    functions/       the Python modules the scripts use
    analysis/        scripts that print numbers quoted in the paper
    figures/         one script per figure
    run_*.py         the analyses
    config.py        paths and the settings of every analysis

## Licence

GNU General Public License v3.0 or later (`LICENSE`). Copyright (C) 2026 Institute for Advanced
Consciousness Studies.

## Contact

Cedric Cannard, ccannard@protonmail.com

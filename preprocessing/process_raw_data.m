% Copyright (C) 2026 Institute for Advanced Consciousness Studies
% Author: Cedric Cannard
% SPDX-License-Identifier: GPL-3.0-or-later
%
% PROCESS_RAW_DATA  Preprocess every recording of the BIDS dataset (OpenNeuro ds008836).
%
% For each sub-*/ses-*/eeg/*_eeg.set:
%   1. keep the 64 EEG channels, the ECG and the respiration belt
%   2. resample to 250 Hz; ECG and respiration are set aside unfiltered
%   3. EEG: high-pass 0.5 Hz, notch 58-62 Hz, low-pass 30 Hz, standard 10-05 positions,
%      average reference computed with an added zero channel
%   4. events: depth reports of 0 removed, reports above 5 set to 5 (clean_depth_reports)
%   5. on a copy high-passed at 1 Hz: bad channels (clean_channels), artifact subspace
%      reconstruction (cutoff 80), spherical interpolation of the bad channels on both copies,
%      extended Infomax ICA at the rank of the data
%   6. the ICA weights go to the 0.5 Hz data; components that ICLabel assigns to muscle or eye
%      with probability 0.9 or more are removed
%   7. artifact subspace reconstruction of the 0.5 Hz data (cutoff 15)
%   8. ECG high-passed at 1 Hz, respiration low-passed at 1 Hz
%   9. samples where the ECG is contaminated are removed from all signals together; a
%      recording with more than 20% contaminated samples is skipped
% The ECG and the respiration are saved inside each file as EEG.ECG and EEG.RESP.
%
% The recordings of the paper were loaded from the original XDF files with the same stream
% synchronisation and cropping that produced the BIDS files; the BIDS files hold the same
% samples and markers. Gap markers of the BIDS files (a few recordings) are dropped here
% because the original recordings were processed as continuous signals.
%
% Requires the Signal Processing and the Statistics and Machine Learning Toolboxes, EEGLAB
% (with the clean_rawdata, ICLabel, firfilt and dipfit plugins), the BrainBeats toolbox
% (apply_car, detect_ecg_artifacts; see the README for the version) and the functions in
% preprocessing/functions. Extended Infomax ICA starts from random weights, so a rerun matches
% the published preprocessed files closely, not exactly; the other steps are deterministic.
%
% Output: <out_path>/<participant>/<session>/<participant>_ses<session>.set, and sInfo.mat (a
% per-recording summary, saved after every recording) in <out_path>. A recording whose ECG is
% contaminated in more than 20% of its samples gets no .set file; sInfo records it as ecg_bad.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PATHS (edit, or set before running) %%%%%%%%
repo_root = fileparts(fileparts(mfilename('fullpath')));
if ~exist('bids_path', 'var'), bids_path = fullfile(repo_root, 'data', 'bids'); end
if ~exist('out_path', 'var'),  out_path  = fullfile(repo_root, 'data', 'preprocessed'); end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath(fullfile(fileparts(mfilename('fullpath')), 'functions'))
eeglab nogui;
elc_file = fullfile(fileparts(which('pop_dipfit_settings')), 'standard_BEM', 'elec', ...
    'standard_1005.elc');
req = {'apply_car', 'clean_channels', 'clean_asr', 'runica', 'pop_iclabel', 'pop_interp', ...
       'detect_ecg_artifacts', 'clean_depth_reports', 'reorder_channels', ...
       'remove_bad_segments', 'merge_segments', ...
       'fir1', 'filtfilt', 'resample', 'mad'};        % Signal Processing / Statistics toolboxes
missing = req(cellfun(@(f) isempty(which(f)), req));
if ~isempty(missing)
    error('Not on the MATLAB path: %s', strjoin(missing, ', '));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dwnsamp     = 250;
hp_final    = 0.5;
hp_ica      = 1.0;
lp_cutoff   = 30;
minCorr     = .65;
maxTol      = .4;
nSamp       = 50;
asr_thresh1 = 80;
asr_thresh2 = 15;

% ICLabel: [min max] probability per class (Brain, Muscle, Eye, Heart, Line, Channel, Other)
ica_flags = [NaN NaN; .9 1; .9 1; NaN NaN; NaN NaN; NaN NaN; NaN NaN];

% ECG contamination
ECG_WIN_SEC    = 0.5;   % window of the high-frequency power estimate (s)
ECG_HF_THRESH  = 15;    % high-frequency threshold (robust z)
ECG_AMP_THRESH = 6;     % amplitude threshold (robust z)
MAX_ECG_BAD    = 20;    % skip a recording with more than this percentage flagged
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

files = dir(fullfile(bids_path, 'sub-*', 'ses-*', 'eeg', '*_task-meditation_eeg.set'));
if isempty(files)
    error('No BIDS EEG files under %s', bids_path);
end
if ~exist(out_path, 'dir'), mkdir(out_path); end
sInfo = struct('subject', {}, 'session', {});   % sInfo(i).session(k): one entry per recording

for iF = 1:numel(files)
    tok = regexp(files(iF).name, '^(sub-[^_]+)_ses-(\d+)_', 'tokens', 'once');
    subject  = tok{1};
    iSession = str2double(tok{2});
    outFolder = fullfile(out_path, subject, num2str(iSession));
    if ~exist(outFolder, 'dir'), mkdir(outFolder); end
    info = struct('subject', subject, 'session', iSession, 'status', 'running');
    fprintf('\n%s session %d\n', subject, iSession);

    % ── 1. LOAD ───────────────────────────────────────────────────────
    EEG_all = pop_loadset('filename', files(iF).name, 'filepath', files(iF).folder);
    labels  = {EEG_all.chanlocs.labels};
    EEG_all = pop_select(EEG_all, 'channel', [labels(1:64) {'ECG' 'RESP'}]);
    EEG_all.event(strcmpi({EEG_all.event.type}, 'boundary')) = [];
    EEG_all = eeg_checkset(EEG_all, 'eventconsistency');

    % ── 2. MINIMAL PREPROCESSING ──────────────────────────────────────
    EEG_all = pop_resample(EEG_all, dwnsamp);

    % physiological channels are set aside before the EEG is filtered
    ECG_full  = pop_select(EEG_all, 'channel', {'ECG'});
    RESP_full = pop_select(EEG_all, 'channel', {'RESP'});
    EEG_all   = pop_select(EEG_all, 'nochannel', {'ECG' 'RESP'});

    EEG_all = pop_eegfiltnew(EEG_all, 'locutoff', hp_final);
    EEG_all = pop_eegfiltnew(EEG_all, 'locutoff', 58, 'hicutoff', 62, 'revfilt', 1);
    EEG_all = pop_eegfiltnew(EEG_all, 'hicutoff', lp_cutoff);
    EEG_all = pop_chanedit(EEG_all, {'lookup', elc_file});
    EEG_all = apply_car(EEG_all);

    % reattached so that the event cleaning applies to all signals together
    EEG_all.data(end+1,:)          = ECG_full.data;
    EEG_all.chanlocs(end+1).labels = 'ECG';
    EEG_all.data(end+1,:)          = RESP_full.data;
    EEG_all.chanlocs(end+1).labels = 'RESP';
    EEG_all.nbchan = size(EEG_all.data, 1);
    EEG_all = eeg_checkset(EEG_all);

    % ── 3a. Events ────────────────────────────────────────────────────
    EEG = clean_depth_reports(EEG_all);
    info.dur_min   = round(EEG.pnts / EEG.srate / 60, 2);

    % ── 3b. Separate ECG/RESP from the EEG ────────────────────────────
    ECG  = pop_select(EEG, 'channel', {'ECG'});
    RESP = pop_select(EEG, 'channel', {'RESP'});
    EEG  = pop_select(EEG, 'nochannel', {'ECG' 'RESP'});
    EEG  = reorder_channels(EEG);
    oriChanlocs = EEG.chanlocs;

    % 1 Hz high-pass copy for ICA
    EEG_ica = pop_eegfiltnew(EEG, 'locutoff', hp_ica);

    % ── 3c. Bad channels ──────────────────────────────────────────────
    [EEG_ica, badChan] = clean_channels(EEG_ica, minCorr, 15, [], maxTol, nSamp);
    info.badChan = {oriChanlocs(badChan).labels};
    if any(badChan)
        EEG = pop_select(EEG, 'nochannel', find(badChan));
    end

    % ── 3d. ASR on the ICA copy (cutoff 80) ───────────────────────────
    cleanEEG_asr1 = clean_asr(EEG_ica, asr_thresh1, [], [], [], [], [], [], [], false, []);
    mask1 = sum(abs(EEG_ica.data - cleanEEG_asr1.data), 1) > 1e-10;
    info.ASR1_pct = round(sum(mask1) / numel(mask1) * 100, 2);
    EEG_ica = cleanEEG_asr1;

    % ── 3e. Interpolate the bad channels ──────────────────────────────
    if any(badChan)
        EEG_ica = pop_interp(EEG_ica, oriChanlocs, 'spherical');
        EEG     = pop_interp(EEG, oriChanlocs, 'spherical');
    end

    % ── 3f. ICA at the rank of the data ───────────────────────────────
    dataRank = sum(eig(cov(double(EEG.data'))) > 1e-7);
    EEG_ica  = pop_runica(EEG_ica, 'icatype', 'runica', 'options', {'extended', 1, 'pca', dataRank});
    EEG.icaweights  = EEG_ica.icaweights;
    EEG.icasphere   = EEG_ica.icasphere;
    EEG.icachansind = EEG_ica.icachansind;
    EEG = eeg_checkset(EEG);

    % ── 3g. ICLabel: muscle and eye components removed ────────────────
    EEG = pop_iclabel(EEG, 'default');
    EEG = pop_icflag(EEG, ica_flags);
    badComp = find(EEG.reject.gcompreject);
    info.nBadComp  = numel(badComp);
    info.nGoodComp = size(EEG.icaweights, 1) - numel(badComp);
    EEG = pop_subcomp(EEG, badComp, 0);
    EEG.icaact = [];
    EEG = eeg_checkset(EEG);

    % ── 3h. ASR on the cleaned data (cutoff 15) ───────────────────────
    cleanEEG_asr2 = clean_asr(EEG, asr_thresh2, [], [], [], [], [], [], [], false, []);
    mask2 = sum(abs(EEG.data - cleanEEG_asr2.data), 1) > 1e-10;
    info.ASR2_pct = round(sum(mask2) / numel(mask2) * 100, 2);
    EEG = cleanEEG_asr2;

    % ── 3i. Filter ECG and respiration ────────────────────────────────
    ECG  = pop_eegfiltnew(ECG,  'locutoff', 1.0);
    RESP = pop_eegfiltnew(RESP, 'hicutoff', 1.0);

    % ── 3j. Remove the samples where the ECG is contaminated ──────────
    mask_hf = detect_ecg_artifacts(ECG, 'WinSec', ECG_WIN_SEC, 'HFthresh', ECG_HF_THRESH, ...
        'AmpThresh', ECG_AMP_THRESH, 'Plot', false);
    info.ECG_pct_bad = round(sum(mask_hf) / numel(mask_hf) * 100, 2);
    if info.ECG_pct_bad > MAX_ECG_BAD
        warning('%s session %d: %.1f%% of the ECG flagged, recording skipped.', ...
            subject, iSession, info.ECG_pct_bad);
        info.status = 'ecg_bad';
        sInfo = store(sInfo, info);
        save(fullfile(out_path, 'sInfo.mat'), 'sInfo');
        continue
    end
    physio             = ECG;
    physio.data        = [ECG.data; RESP.data];
    physio.chanlocs(2) = RESP.chanlocs(1);
    physio.nbchan      = 2;
    [EEG, physio] = remove_bad_segments(EEG, physio, mask_hf);
    nan_mask = isnan(physio.data(1,:)) | isnan(physio.times);
    if any(nan_mask)
        [EEG, physio] = remove_bad_segments(EEG, physio, nan_mask);
    end
    assert(EEG.pnts == physio.pnts, 'EEG/physio length mismatch (%s session %d)', subject, iSession);
    EEG.ECG  = pop_select(physio, 'channel', 1);
    EEG.RESP = pop_select(physio, 'channel', 2);
    info.dur_clean_min = round(EEG.pnts / EEG.srate / 60, 2);

    % ── 3k. Save ──────────────────────────────────────────────────────
    info.status = 'ok';
    pop_saveset(EEG, 'filepath', outFolder, 'filename', sprintf('%s_ses%d.set', subject, iSession));
    sInfo = store(sInfo, info);
    save(fullfile(out_path, 'sInfo.mat'), 'sInfo');
end
fprintf('\nDone.\n');


function sInfo = store(sInfo, info)
% Add one recording's summary as sInfo(participant).session(session).
k = find(strcmp({sInfo.subject}, info.subject), 1);
if isempty(k)
    k = numel(sInfo) + 1;
    sInfo(k).subject = info.subject;
end
for f = fieldnames(info)'
    if ~ismember(f{1}, {'subject', 'session'})
        sInfo(k).session(info.session).(f{1}) = info.(f{1});
    end
end
end

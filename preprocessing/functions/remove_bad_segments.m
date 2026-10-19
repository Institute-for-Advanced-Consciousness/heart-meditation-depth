function [EEG, ECG] = remove_bad_segments(EEG, ECG, mask, minGapSec)
% REMOVE_BAD_SEGMENTS  Remove the flagged samples from the EEG and from the physiological
% signals together. Runs of flagged samples closer than minGapSec (default 0.1 s) are merged
% first. The mask of the samples kept is stored in ECG.etc.clean_sample_mask.
%
% Copyright (C) 2026 Institute for Advanced Consciousness Studies
% Author: Cedric Cannard
% SPDX-License-Identifier: GPL-3.0-or-later

if nargin < 4, minGapSec = 0.1; end

mask = logical(mask(:)');
fprintf('  Removing %.2f%% of data\n', 100*mean(mask));
ECG.etc.clean_sample_mask = ~mask;

badData = reshape(find(diff([false mask false])), 2, [])';
badData(:,2) = badData(:,2) - 1;
if isempty(badData), fprintf('  No bad segments detected.\n'); return; end

bad_times_sec = [(badData(:,1)-1), badData(:,2)] / ECG.srate;
badData_ECG   = [round(bad_times_sec(:,1)*ECG.srate)+1, round(bad_times_sec(:,2)*ECG.srate)];
badData_EEG   = [round(bad_times_sec(:,1)*EEG.srate)+1, round(bad_times_sec(:,2)*EEG.srate)];

badData_ECG(:,2) = min(badData_ECG(:,2), ECG.pnts);
badData_ECG(badData_ECG(:,1) > ECG.pnts, :) = [];
badData_EEG(:,2) = min(badData_EEG(:,2), EEG.pnts);
badData_EEG(badData_EEG(:,1) > EEG.pnts, :) = [];

badData_ECG = merge_segments(badData_ECG, round(minGapSec*ECG.srate));
badData_EEG = merge_segments(badData_EEG, round(minGapSec*EEG.srate));
fprintf('  Bad segments after merging: ECG=%d  EEG=%d\n', size(badData_ECG,1), size(badData_EEG,1));

mask_updated = false(1, ECG.pnts);
for i = 1:size(badData_ECG,1)
    mask_updated(badData_ECG(i,1):badData_ECG(i,2)) = true;
end
ECG.etc.clean_sample_mask = ~mask_updated;

ECG = pop_select(ECG, 'nopoint', badData_ECG);
EEG = pop_select(EEG, 'nopoint', badData_EEG);

fprintf('  EEG duration after cleaning: %.1f min\n', EEG.pnts/EEG.srate/60);
fprintf('  ECG duration after cleaning: %.1f min\n', ECG.pnts/ECG.srate/60);
end

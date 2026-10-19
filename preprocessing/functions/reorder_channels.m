function EEG = reorder_channels(EEG)
% Reorder channels to follow front-to-back direction.
% Channels not found in desired_order are appended at the end with a warning.
% If ICA is present, update EEG.icachansind so ICA remains aligned with the
% reordered channel layout.
%
% Copyright (C) 2026 Institute for Advanced Consciousness Studies
% Author: Cedric Cannard
% SPDX-License-Identifier: GPL-3.0-or-later

desired_order = { ...
    'Fp1','Fpz','Fp2', ...
    'AF7','AF3','AFz','AF4','AF8', ...
    'F7','F5','F3','F1','Fz','F2','F4','F6','F8', ...
    'FT7','FT9','FT8','FT10', ...
    'FC5','FC3','FC1','FCz','FC2','FC4','FC6', ...
    'T7','T8', ...
    'C5','C3','C1','Cz','C2','C4','C6', ...
    'TP7','TP9','TP8','TP10', ...
    'CP5','CP3','CP1','CPz','CP2','CP4','CP6', ...
    'P9','P7','P5','P3','P1','Pz','P2','P4','P6','P8','P10', ...
    'PO9','PO7','PO3','POz','PO4','PO8','PO10', ...
    'O1','Oz','O2','Iz' ...
    };

current_labels = {EEG.chanlocs.labels};
[found, loc_in_desired] = ismember(lower(current_labels), lower(desired_order));

nRecorded  = sum(found);
nAvailable = numel(desired_order);
fprintf('reorder_channels: %d / %d known channels present\n', nRecorded, nAvailable);

if ~all(found)
    missing = current_labels(~found);
    warning('reorder_channels: %d channel(s) not in desired_order, appending at end: %s', ...
        numel(missing), strjoin(missing, ', '));
    loc_in_desired(~found) = numel(desired_order) + (1:sum(~found));
end

[~, order_idx] = sort(loc_in_desired, 'ascend');

% Map old channel index -> new channel index
old_to_new = zeros(1, numel(order_idx));
old_to_new(order_idx) = 1:numel(order_idx);

% Reorder chanlocs
EEG.chanlocs = EEG.chanlocs(order_idx);

% Reorder data (works for 2D continuous and 3D epoched data)
if ndims(EEG.data) == 2
    EEG.data = EEG.data(order_idx, :);
elseif ndims(EEG.data) == 3
    EEG.data = EEG.data(order_idx, :, :);
else
    error('reorder_channels: unsupported EEG.data dimensionality.');
end

% If ICA is present, remap icachansind so ICA still points to the correct channels
if isfield(EEG, 'icachansind') && ~isempty(EEG.icachansind)
    EEG.icachansind = old_to_new(EEG.icachansind);
end

end
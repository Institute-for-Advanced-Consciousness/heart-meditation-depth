function EEG = clean_depth_reports(EEG)
% CLEAN_DEPTH_REPORTS  Remove depth reports of 0 and set depth and confidence reports above 5
% to 5. Other events are left as they are.
%
% Copyright (C) 2026 Institute for Advanced Consciousness Studies
% Author: Cedric Cannard
% SPDX-License-Identifier: GPL-3.0-or-later

labels = lower(strtrim(string({EEG.event.type})));
depth0 = contains(labels, "meditation_depth-0") | contains(labels, "rating-depth-0");
if any(depth0)
    EEG.event(depth0) = [];
    EEG = eeg_checkset(EEG);
end
for i = 1:numel(EEG.event)
    for stem = ["meditation_depth-", "confidence_depth-"]
        tok = regexp(EEG.event(i).type, ['^(.*' char(stem) ')(\d+)$'], 'tokens', 'once');
        if ~isempty(tok)
            if str2double(tok{2}) > 5
                EEG.event(i).type = [tok{1} '5'];
            end
            break
        end
    end
end
EEG = eeg_checkset(EEG);
end

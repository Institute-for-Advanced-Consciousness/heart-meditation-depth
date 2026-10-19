function segs = merge_segments(segs, minGap)
% MERGE_SEGMENTS  Merge [start end] sample ranges separated by minGap samples or fewer.
%
% Copyright (C) 2026 Institute for Advanced Consciousness Studies
% Author: Cedric Cannard
% SPDX-License-Identifier: GPL-3.0-or-later
if isempty(segs), return; end
i = 1;
while i < size(segs, 1)
    if segs(i+1,1) - segs(i,2) <= minGap
        segs(i,2)   = segs(i+1,2);
        segs(i+1,:) = [];
    else
        i = i + 1;
    end
end
end
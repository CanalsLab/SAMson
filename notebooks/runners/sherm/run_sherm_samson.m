function run_sherm_samson()
%RUN_SHERM_SAMSON  SHERM (Liu et al., 2020) on the SAMson publication datasets D1-D7.
%
%   1. Edit the SETTINGS block below.
%   2. Type  run_sherm_samson  in the MATLAB command window, first with MODE = 'pilot' (one scan per
%      dataset-contrast, 10 scans, written to <OUTPUTS>/pilot/) and send the OUTPUTS folder back for
%      checking; then with MODE = 'all' (409 scans).
%   A run can be stopped and restarted: scans whose mask already exists are skipped.
%
%   Input   <DATASETS>/<Dk>/<anat|dwi|func>[/<age>]/<stem>.nii.gz        (scans listed by dataset_settings.json)
%   Output  <OUTPUTS>/masks/sherm/<Dk>/<anat|dwi|func>[/<age>]/<stem>_mask.nii.gz
%               uint8 0/1 mask on the grid and with the header of the dataset image (layout of every other tool)
%           <OUTPUTS>/sherm_runtimes.csv      wall-clock time, CPU time, peak memory and workers: one row per scan,
%                                             one per dataset-contrast and one per run
%           <OUTPUTS>/sherm_environment.json  computer (operating system, CPU model, cores, RAM, GPUs), MATLAB,
%                                             toolboxes, VLFeat, SHERM commit, workers, settings, elapsed time
%           <OUTPUTS>/logs/sherm/             one JSON log per scan and one environment file per execution
%           <OUTPUTS>/pilot/                  MODE = 'pilot': the same files, and sherm_inputs/ with the exact image
%                                             given to SHERM, for checking
%
%   Image given to SHERM (same rule as for the other tools of the benchmark; docs/tools_and_conventions.md):
%     a. the dataset image re-oriented losslessly to the canonical SAMson orientation (axis permutation and
%        reversals of dataset_settings.json; 3rd axis = coronal slices, caudal to rostral);
%     b. 2nd axis reversed, so that the dorsal side is on the first row: the orientation of SHERM's example images
%        (also the input orientation of BEN and RBM in the benchmark);
%     c. header voxel size divided by 10 (real mm). The datasets store
%        10x voxel sizes, and SHERM's parameters are fixed in mm (kernel radii 0.2-0.7 mm; brain volume
%        300-550 mm3 for 'mouse', 1500-1900 mm3 for 'rat').
%   No intensity or other change. The mask is mapped back with the inverse of a and b.
%
%   Requirements: MATLAB R2017a or later with the Image Processing and the Statistics and Machine Learning
%   toolboxes (Parallel Computing Toolbox optional); SHERM, VLFeat 0.9.21 and NIfTI_tools (see README.md).

%% =============================== SETTINGS ===============================
ROOT       = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));   % release folder (<release>/notebooks/runners/sherm/this file)
DATASETS   = fullfile(ROOT, 'datasets');           % folder that contains dataset_settings.json
OUTPUTS    = fullfile(ROOT, 'work', 'sherm');      % created if needed (the published SHERM records are in outputs/sherm and outputs/masks/sherm)
SHERM_DIR  = '/path/to/SHERM-rodentSkullStrip';    % github.com/liu-yikang/SHERM-rodentSkullStrip, commit c44afe8
VLFEAT_DIR = '/path/to/vlfeat-0.9.21';             % folder that contains toolbox/vl_setup.m
NIFTI_DIR  = '/path/to/NIfTI_tools';               % folder with load_untouch_nii.m and save_untouch_nii.m
MODE       = 'pilot';                              % 'pilot' or 'all'
N_WORKERS  = 4;                                    % parallel MATLAB workers, 1 thread each; 0 = one scan at a time
MEMORY_GB  = 64;                                   % RAM the run may use: fewer workers on the large D7 images
ONLY       = {};                                   % groups to run, in this order (empty = all), e.g.
                                                   % {'D7_T2w'}

% SHERM settings (one per dataset; do not change between scans of a dataset)
ANIMAL = struct('D1', 'mouse', 'D2', 'mouse', 'D3', 'mouse', 'D4', 'mouse', 'D5', 'mouse', 'D6', 'mouse', 'D7', 'mouse');
ISOTROPIC = 0;   % all datasets have slices thicker than the in-plane voxel size: SHERM's anisotropic (2D) mode

% Brain volumes of the reference masks: 320-540 mm3; D4 (6 slices) 256-320 mm3.
%% =========================================================================

DATASETS = strip_separator(DATASETS);
OUTPUTS = strip_separator(OUTPUTS);
if strcmp(MODE, 'pilot')
    OUTPUTS = fullfile(OUTPUTS, 'pilot');   % kept apart: the full run then times all scans of a dataset together
elseif ~strcmp(MODE, 'all')
    error('run_sherm_samson:mode', 'MODE must be ''pilot'' or ''all''.');
end
addpath(SHERM_DIR);
addpath(NIFTI_DIR);
run(fullfile(VLFEAT_DIR, 'toolbox', 'vl_setup.m'));
check_setup(DATASETS, SHERM_DIR);

cfg.descriptor = fullfile(SHERM_DIR, 'rodent_brain_polar_descriptor.mat');
cfg.isotropic = ISOTROPIC;
cfg.outputs = OUTPUTS;
cfg.keep_inputs = strcmp(MODE, 'pilot');
cfg.run_id = char(datetime('now', 'Format', 'yyyyMMdd''T''HHmmss'));   % identifies this execution in the logs

jobs = list_jobs(DATASETS, OUTPUTS, ANIMAL);
fprintf('%d runs (%d scans) found in %s\n', numel(jobs), sum(strcmp({jobs.run}, 'main')), DATASETS);
if strcmp(MODE, 'pilot')
    [~, first] = unique({jobs.group}, 'stable');
    jobs = jobs(first);
end

% computer, software and settings of this execution; written again at the end with the elapsed time
make_folder(fullfile(OUTPUTS, 'logs', 'sherm'));
t_execution = tic;
env = describe_execution(cfg.run_id, MODE, SHERM_DIR, N_WORKERS, MEMORY_GB, ANIMAL, ISOTROPIC, numel(jobs));
write_environment(OUTPUTS, env);

% one thread per scan: N_WORKERS single-thread workers, or the client limited to one thread
use_pool = N_WORKERS > 0 && ~isempty(ver('parallel')) && license('test', 'Distrib_Computing_Toolbox');
if use_pool
    t_pool = tic;
    pool = gcp('nocreate');
    if isempty(pool) || pool.NumWorkers ~= N_WORKERS
        if ~isempty(pool)
            delete(pool);
        end
        pool = parpool(N_WORKERS);
    end
    env.resources.pool_workers = pool.NumWorkers;
    env.resources.pool_start_seconds = toc(t_pool);
    max_workers = N_WORKERS;
else
    try   % SHERM's internal parfor loops must run in this one process
        delete(gcp('nocreate'));
        settings = parallel.Settings;
        settings.Pool.AutoCreate = false;
    catch
    end
    env.resources.pool_workers = 0;
    max_workers = 1;
end
threads_before = maxNumCompThreads(1);
restore_threads = onCleanup(@() maxNumCompThreads(threads_before));

groups = unique({jobs.group}, 'stable');
if ~isempty(ONLY)   % run only some groups, in the order given
    unknown = setdiff(ONLY, groups);
    if ~isempty(unknown)
        error('ONLY: unknown group(s): %s. Groups: %s', strjoin(unknown, ', '), strjoin(groups, ', '));
    end
    groups = ONLY(:)';
end
for g = 1:numel(groups)
    group = jobs(strcmp({jobs.group}, groups{g}));
    group = group(arrayfun(@(j) ~exist(j.output, 'file'), group));
    if isempty(group)
        fprintf('%s: all masks exist\n', groups{g});
        continue
    end
    need_gb = memory_estimate(group(1));
    workers = max(1, min(max_workers, floor(MEMORY_GB / need_gb)));
    concurrent = min(workers, numel(group));   % scans of this dataset that run at the same time
    fprintf('%s: %d scans, about %.0f GB each, %d at a time\n', groups{g}, numel(group), need_gb, concurrent);
    if use_pool
        parfor (j = 1:numel(group), concurrent)
            process_scan(group(j), cfg, concurrent);
        end
    else
        for j = 1:numel(group)
            process_scan(group(j), cfg, 1);
        end
    end
end

env.finished = time_now();
env.elapsed_seconds = toc(t_execution);
write_environment(OUTPUTS, env);
merge_logs(OUTPUTS, jobs);
end


%% ------------------------------------------------------------------ one scan

function process_scan(job, cfg, workers)
threads_before = maxNumCompThreads(1);              % one thread per scan
restore_threads = onCleanup(@() maxNumCompThreads(threads_before));
peak_reset = reset_peak_memory();
[t_start, start_posix] = time_now();
cpu_start = cputime;
t0 = tic;
status = 'ok';
message = '';
tmp = tempname;
mkdir(tmp);
cleaner = onCleanup(@() rmdir(tmp, 's'));
try
    raw = read_nifti(job.image, tmp);                       % stored orientation, x10 header
    img = double(raw.img(:, :, :, 1));
    slope = double(raw.hdr.dime.scl_slope);
    if isfinite(slope) && slope ~= 0
        img = img * slope + double(raw.hdr.dime.scl_inter);
    end

    % image given to SHERM: canonical orientation, dorsal side first, real voxel size (mm)
    img = to_sherm(img, job.perm, job.flips);
    voxel = abs(double(raw.hdr.dime.pixdim(2:4))) / 10;
    voxel = voxel(job.perm + 1);
    if ~isempty(job.voxel_mm)   % voxel size given explicitly (canonical axes); empty: header / 10
        voxel = job.voxel_mm;
    end
    in = raw;
    in.img = single(img);
    in.hdr.dime.dim(1:8) = [3, size(img, 1), size(img, 2), size(img, 3), 1, 1, 1, 1];
    in.hdr.dime.pixdim(2:4) = voxel;
    in.hdr.dime.datatype = 16;
    in.hdr.dime.bitpix = 32;
    in.hdr.dime.scl_slope = 1;
    in.hdr.dime.scl_inter = 0;
    in.hdr.hist.qform_code = 0;                            % SHERM reads the voxel size only
    in.hdr.hist.sform_code = 0;
    in_file = fullfile(tmp, 'sherm_input.nii');
    out_file = fullfile(tmp, 'sherm_output.nii');
    save_untouch_nii(in, in_file);
    if cfg.keep_inputs
        keep_input(in_file, job, cfg, tmp);
    end

    try
        sherm(in_file, out_file, cfg.descriptor, job.animal, cfg.isotropic);
        result = load_untouch_nii(out_file);
        mask = result.img > 0.5;
        if ~any(mask(:))
            status = 'empty';
        end
    catch err
        if ~is_no_candidate(err)
            rethrow(err);
        end
        status = 'no_candidate';   % no region met SHERM's volume and shape criteria: empty mask
        message = 'no region met the volume and shape criteria of SHERM';
        mask = false(size(img));
    end

    write_mask(raw, from_sherm(mask, job.perm, job.flips), job.output, tmp);
catch err
    if is_setup_error(err)
        rethrow(err);   % stops the run: fix the installation and restart
    end
    status = classify_error(err);   % no mask is written; the scan is retried when the run is restarted
    message = sprintf('%s: %s', err.identifier, err.message);
end
seconds = toc(t0);
cpu_seconds = cputime - cpu_start;                  % CPU time of this MATLAB process during the scan
[t_end, end_posix] = time_now();
peak_gb = NaN;
if peak_reset
    peak_gb = peak_memory_gb();
end
[load_1min, available_gb] = system_load();
row = struct('tool', 'sherm', 'level', 'scan', 'dataset', job.dataset, 'contrast', job.contrast, 'run', job.run, ...
    'subject', job.subject, 'session', job.session, 'stem', job.stem, 'status', status, 'n_scans', 1, ...
    'seconds', seconds, 'cpu_seconds', cpu_seconds, 'peak_memory_gb', peak_gb, 'start_time', t_start, ...
    'end_time', t_end, 'workers', workers, 'threads_per_worker', maxNumCompThreads, 'gpu', 'none', ...
    'load_1min', load_1min, 'available_memory_gb', available_gb, 'voxel_mm', voxel_text(job), ...
    'animal', job.animal, 'isotropic', cfg.isotropic, 'matlab', version('-release'), 'run_id', cfg.run_id, ...
    'message', message, 'start_posix', start_posix, 'end_posix', end_posix);
write_json(fullfile(cfg.outputs, 'logs', 'sherm', [job.log_name '.json']), row);
fprintf('[%s] %s: %s (%.1f s)\n', job.group, job.stem, status, seconds);
end


function a = to_sherm(a, perm, flips)
% stored -> canonical (canonical axis i = stored axis perm(i) + 1, then reversals) -> dorsal side first
a = permute(a, perm(:)' + 1);
for axis = 1:3
    if flips(axis)
        a = flip(a, axis);
    end
end
a = flip(a, 2);
end


function a = from_sherm(a, perm, flips)
% inverse of to_sherm
a = flip(a, 2);
for axis = 1:3
    if flips(axis)
        a = flip(a, axis);
    end
end
a = ipermute(a, perm(:)' + 1);
end


function write_mask(raw, mask, output, tmp)
% uint8 0/1 mask with the header of the dataset image (x10 voxel size, same affine)
assert(isequal(size(mask), double(raw.hdr.dime.dim(2:4))), 'mask size differs from the image size');
nii = raw;
nii.img = uint8(mask);
nii.hdr.dime.dim(1:8) = [3, size(mask, 1), size(mask, 2), size(mask, 3), 1, 1, 1, 1];
nii.hdr.dime.datatype = 2;
nii.hdr.dime.bitpix = 8;
nii.hdr.dime.scl_slope = 1;
nii.hdr.dime.scl_inter = 0;
nii.hdr.dime.cal_max = 0;
nii.hdr.dime.cal_min = 0;
nii.hdr.dime.glmax = 1;
nii.hdr.dime.glmin = 0;
[folder, name] = fileparts(output);   % name = <stem>_mask.nii
plain = fullfile(tmp, name);
save_untouch_nii(nii, plain);
packed = gzip(plain, tmp);
make_folder(folder);
movefile(packed{1}, output, 'f');
end


function keep_input(in_file, job, cfg, tmp)
copy = fullfile(tmp, [job.stem '_sherm_input.nii']);
copyfile(in_file, copy);
folder = fullfile(cfg.outputs, 'sherm_inputs', job.run, job.dataset, job.rel_dir);
make_folder(folder);
gzip(copy, folder);
end


%% ------------------------------------------------------------------ errors

function tf = is_no_candidate(err)
% sherm.m fails at desc_dist_min(1) when no candidate region has a positive shape distance
tf = ~isempty(err.stack) && strcmp(err.stack(1).name, 'sherm') && ...
     (strcmp(err.identifier, 'MATLAB:badsubscript') || contains(err.message, 'Index exceeds'));
end


function tf = is_setup_error(err)
tf = any(strcmp(err.identifier, {'MATLAB:UndefinedFunction', 'MATLAB:undefinedVarOrFunction'})) || ...
     startsWith(err.identifier, 'MATLAB:license');
end


function status = classify_error(err)
if any(strcmp(err.identifier, {'MATLAB:nomem', 'MATLAB:array:SizeLimitExceeded', 'MATLAB:pmaxsize'})) || ...
        contains(lower(err.message), 'out of memory')
    status = 'out_of_memory';
else
    status = 'error';
end
end


%% ------------------------------------------------------------------ scans

function jobs = list_jobs(datasets, outputs, animal)
settings = jsondecode(fileread(fullfile(datasets, 'dataset_settings.json')));
keys = setdiff(fieldnames(settings), {'conventions'}, 'stable');
jobs = struct('dc', {}, 'dataset', {}, 'contrast', {}, 'subject', {}, 'session', {}, 'stem', {}, 'image', {}, ...
              'rel_dir', {}, 'output', {}, 'perm', {}, 'flips', {}, 'animal', {}, 'run', {}, 'group', {}, ...
              'voxel_mm', {}, 'log_name', {});
for k = 1:numel(keys)
    s = settings.(keys{k});
    files = glob_pattern(datasets, s.image_pattern);
    if numel(files) ~= s.n_scans
        warning('run_sherm_samson:count', '%s: %d files found, %d expected', keys{k}, numel(files), s.n_scans);
    end
    base = fullfile(datasets, s.dataset);
    for i = 1:numel(files)
        [folder, name] = fileparts(files{i});            % name = <stem>.nii
        stem = regexprep(name, '\.nii$', '');
        rel_dir = folder(numel(base) + 2:end);           % e.g. anat, anat/3_month, func
        jobs(end + 1) = struct('dc', keys{k}, 'dataset', s.dataset, 'contrast', s.contrast, ...
            'subject', first_token(stem, '^sub-([^_]+)'), 'session', first_token(stem, '_ses-([^_]+)'), ...
            'stem', stem, 'image', files{i}, 'rel_dir', rel_dir, ...
            'output', fullfile(outputs, 'masks', 'sherm', s.dataset, rel_dir, [stem '_mask.nii.gz']), ...
            'perm', s.canonical.perm(:)', 'flips', s.canonical.flips(:)', 'animal', animal.(s.dataset), ...
            'run', 'main', 'group', keys{k}, 'voxel_mm', [], 'log_name', [s.dataset '_' stem]); %#ok<AGROW>
    end
end
end


function files = glob_pattern(root, pattern)
% files matching a dataset_settings.json pattern such as D1/anat/*_month/sub-*_T2w.nii.gz
parts = strsplit(pattern, '/');
paths = {root};
for p = 1:numel(parts)
    found = {};
    for q = 1:numel(paths)
        if contains(parts{p}, '*')
            entries = dir(fullfile(paths{q}, parts{p}));
            for e = 1:numel(entries)
                if ~startsWith(entries(e).name, '.')
                    found{end + 1} = fullfile(paths{q}, entries(e).name); %#ok<AGROW>
                end
            end
        elseif exist(fullfile(paths{q}, parts{p}), 'file')
            found{end + 1} = fullfile(paths{q}, parts{p}); %#ok<AGROW>
        end
    end
    paths = found;
end
files = sort(paths);
end


function token = first_token(text, expression)
token = regexp(text, expression, 'tokens', 'once');
if isempty(token)
    token = '';
else
    token = token{1};
end
end


function gb = memory_estimate(job)
% peak memory of SHERM's 2D mode: double image pyramid (opening and closing channels), twice
tmp = tempname;
mkdir(tmp);
cleaner = onCleanup(@() rmdir(tmp, 's'));
nii = read_nifti(job.image, tmp);
dims = double(nii.hdr.dime.dim(2:4));
voxel = abs(double(nii.hdr.dime.pixdim(2:4))) / 10;
dims = dims(job.perm + 1);
voxel = voxel(job.perm + 1);
if ~isempty(job.voxel_mm)
    voxel = job.voxel_mm;
end
drs = min(voxel);
channels = numel(0.2:drs:0.7) * (1 + numel(0.2:drs:0.5));
voxels = dims(1) * dims(2) * (round(dims(3) * voxel(3) / drs) + 10);
gb = 2.5 * channels * voxels * 8 / 1e9 + 1;
end


%% ------------------------------------------------------------------ files and logs

function nii = read_nifti(file, tmp)
gunzip(file, tmp);
[~, name] = fileparts(file);   % <stem>.nii
nii = load_untouch_nii(fullfile(tmp, name));
end


function make_folder(folder)
if ~exist(folder, 'dir')
    [~, ~] = mkdir(folder);   % no error if another worker created it meanwhile
end
end


function path = strip_separator(path)
path = char(path);
while numel(path) > 1 && any(path(end) == '/\')
    path = path(1:end - 1);
end
end


function write_json(file, value)
% a log that cannot be written gives a warning and never stops the run
try
    try
        text = jsonencode(value, 'PrettyPrint', true);   % R2021a or later
    catch
        text = jsonencode(value);
    end
    f = fopen(file, 'w');
    fprintf(f, '%s\n', text);
    fclose(f);
catch err
    warning('run_sherm_samson:log', 'Could not write %s: %s', file, err.message);
end
end


function text = read_text(file)
f = fopen(file, 'r');
if f < 0
    error('run_sherm_samson:read', 'Cannot read %s', file);
end
text = fread(f, Inf, '*char')';
fclose(f);
end


function text = read_text_safe(file)
try
    text = read_text(file);
catch
    text = '';
end
end


function text = system_text(command)
% output of a shell command, or '' if it fails
text = '';
try
    [status, out] = system(command);
    if status == 0
        text = strtrim(out);
    end
catch
end
end


function [text, posix] = time_now()
t = datetime('now', 'Format', 'yyyy-MM-dd''T''HH:mm:ss.SSS');
text = char(t);
posix = posixtime(t);
end


function value = first_number(text, expression)
token = regexp(text, expression, 'tokens', 'once');
if isempty(token)
    value = NaN;
else
    value = str2double(token{1});
end
end


function value = as_number(x)
if isempty(x) || ~isnumeric(x)
    value = NaN;   % null in the JSON log
else
    value = double(x(1));
end
end


function text = voxel_text(job)
if isempty(job.voxel_mm)
    text = 'header/10';
else
    text = strjoin(arrayfun(@(v) sprintf('%g', v), job.voxel_mm, 'UniformOutput', false), 'x');
end
end


function merge_logs(outputs, jobs)
% sherm_runtimes.csv from the JSON logs of all scans done so far (any execution): one row per scan, one per
% dataset-contrast and one per run. Wall-clock time of a dataset or run = time while at least one of its
% scans was running (scans that run at the same time are counted once; interruptions are not counted).
fields = {'tool', 'level', 'dataset', 'contrast', 'run', 'subject', 'session', 'stem', 'status', 'n_scans', ...
          'seconds', 'cpu_seconds', 'peak_memory_gb', 'start_time', 'end_time', 'workers', 'threads_per_worker', ...
          'gpu', 'load_1min', 'available_memory_gb', 'voxel_mm', 'animal', 'isotropic', 'matlab', 'run_id', 'message'};
header = strrep(strrep(strjoin(fields, ','), 'start_time', 'start'), 'end_time', 'end');
scans = {};
groups = {};
for i = 1:numel(jobs)
    file = fullfile(outputs, 'logs', 'sherm', [jobs(i).log_name '.json']);
    if ~exist(file, 'file')
        continue
    end
    try
        scans{end + 1} = jsondecode(read_text(file)); %#ok<AGROW>
        groups{end + 1} = jobs(i).group; %#ok<AGROW>
    catch
        warning('run_sherm_samson:log', 'Log not readable: %s', file);
    end
end
if isempty(scans)
    fprintf('No scan logs yet.\n');
    return
end

summaries = {};
labels = {};
names = unique(groups, 'stable');
for g = 1:numel(names)
    summaries{end + 1} = summary_row(scans(strcmp(groups, names{g})), 'dataset'); %#ok<AGROW>
    labels{end + 1} = names{g}; %#ok<AGROW>
end
runs = cellfun(@(r) r.run, scans, 'UniformOutput', false);
names = unique(runs, 'stable');
for r = 1:numel(names)
    summaries{end + 1} = summary_row(scans(strcmp(runs, names{r})), 'run'); %#ok<AGROW>
    labels{end + 1} = ['all, run ' names{r}]; %#ok<AGROW>
end

rows = [scans, summaries];
lines = cell(1, numel(rows) + 1);
lines{1} = header;
for i = 1:numel(rows)
    lines{i + 1} = csv_line(rows{i}, fields);
end
f = fopen(fullfile(outputs, 'sherm_runtimes.csv'), 'w');
fprintf(f, '%s\n', lines{:});
fclose(f);

fprintf('\nTime per dataset-contrast and run (sherm_runtimes.csv):\n');
fprintf('  %-46s %6s %11s %9s %8s  %s\n', '', 'scans', 'wall (min)', 'CPU (h)', 'at once', 'status');
for i = 1:numel(summaries)
    s = summaries{i};
    fprintf('  %-46s %6d %11.1f %9.2f %8s  %s\n', labels{i}, s.n_scans, s.seconds / 60, s.cpu_seconds / 3600, ...
            s.workers, s.status);
end
end


function row = summary_row(rs, level)
starts = cellfun(@(r) as_number(r.start_posix), rs);
ends = cellfun(@(r) as_number(r.end_posix), rs);
[~, a] = min(starts);
[~, b] = max(ends);
executions = unique(cellfun(@(r) r.run_id, rs, 'UniformOutput', false));
executions = executions(:)';
row = struct('tool', 'sherm', 'level', level, 'dataset', joined(rs, 'dataset'), 'contrast', joined(rs, 'contrast'), ...
    'run', joined(rs, 'run'), 'subject', '', 'session', '', 'stem', '', 'status', status_counts(rs), ...
    'n_scans', numel(rs), 'seconds', busy_seconds(starts, ends), ...
    'cpu_seconds', sum(cellfun(@(r) as_number(r.cpu_seconds), rs), 'omitnan'), ...
    'peak_memory_gb', max(cellfun(@(r) as_number(r.peak_memory_gb), rs)), ...
    'start_time', rs{a}.start_time, 'end_time', rs{b}.end_time, 'workers', joined(rs, 'workers'), ...
    'threads_per_worker', joined(rs, 'threads_per_worker'), 'gpu', 'none', ...
    'load_1min', mean(cellfun(@(r) as_number(r.load_1min), rs), 'omitnan'), ...
    'available_memory_gb', min(cellfun(@(r) as_number(r.available_memory_gb), rs)), ...
    'voxel_mm', joined(rs, 'voxel_mm'), 'animal', joined(rs, 'animal'), 'isotropic', joined(rs, 'isotropic'), ...
    'matlab', joined(rs, 'matlab'), 'run_id', strjoin(executions, ';'), ...
    'message', sprintf(['seconds: wall-clock time while at least one of the %d scans was running; ' ...
                        'cpu_seconds: sum over scans; peak_memory_gb: largest scan; %d execution(s)'], ...
                       numel(rs), numel(executions)));
end


function seconds = busy_seconds(starts, ends)
% length of the union of the intervals [start, end]
keep = isfinite(starts) & isfinite(ends);
[starts, order] = sort(starts(keep));
ends = ends(keep);
ends = ends(order);
seconds = 0;
if isempty(starts)
    return
end
s = starts(1);
e = ends(1);
for i = 2:numel(starts)
    if starts(i) > e
        seconds = seconds + (e - s);
        s = starts(i);
        e = ends(i);
    else
        e = max(e, ends(i));
    end
end
seconds = seconds + (e - s);
end


function text = status_counts(rs)
statuses = cellfun(@(r) r.status, rs, 'UniformOutput', false);
names = unique(statuses, 'stable');
names = names(:)';
text = strjoin(cellfun(@(s) sprintf('%s=%d', s, sum(strcmp(statuses, s))), names, 'UniformOutput', false), ';');
end


function text = joined(rs, field)
values = cellfun(@(r) value_text(r.(field)), rs, 'UniformOutput', false);
values = unique(values, 'stable');
text = strjoin(values(:)', ';');
end


function text = value_text(v)
if ischar(v)
    text = v;
elseif isempty(v)
    text = '';
else
    text = sprintf('%.10g', v);
end
end


function line = csv_line(row, fields)
cells = cell(1, numel(fields));
for k = 1:numel(fields)
    if isfield(row, fields{k})
        cells{k} = csv_cell(row.(fields{k}));
    else
        cells{k} = '';
    end
end
line = strjoin(cells, ',');
end


function text = csv_cell(v)
if isempty(v)
    text = '';
elseif ischar(v)
    text = regexprep(v, '\s+', ' ');
    if any(text == ',' | text == '"')
        text = ['"' strrep(text, '"', '""') '"'];
    end
elseif ~isfinite(v)
    text = '';
else
    text = sprintf('%.10g', v);
end
end


%% ------------------------------------------------------------------ computer and resources

function env = describe_execution(run_id, mode, sherm_dir, n_workers, memory_gb, animal, isotropic, n_runs)
env.tool = 'SHERM';
env.repository = 'https://github.com/liu-yikang/SHERM-rodentSkullStrip';
env.commit = system_text(sprintf('git -C "%s" rev-parse --short HEAD', sherm_dir));   % c44afe8 expected
if isempty(env.commit)
    env.commit = 'unknown (git not available; c44afe8 expected)';
end
env.run_id = run_id;
env.mode = mode;
env.started = time_now();
env.finished = '';            % written at the end of the execution
env.elapsed_seconds = NaN;
env.machine = describe_machine();
env.resources = struct('workers_requested', n_workers, 'pool_workers', NaN, 'pool_start_seconds', NaN, ...
    'threads_per_worker', 1, 'memory_limit_gb', memory_gb, ...
    'gpu', 'none: SHERM has no GPU code and runs on the CPU only');
toolboxes = ver;
env.software = struct('matlab', version, 'architecture', computer('arch'), 'vlfeat', '', ...
    'toolboxes', {arrayfun(@(t) [t.Name ' ' t.Version], toolboxes, 'UniformOutput', false)});
try
    env.software.vlfeat = vl_version;
catch
end
env.settings = struct('animal', animal, 'isotropic', isotropic, 'runs', n_runs);
end


function m = describe_machine()
% operating system, CPU model, physical cores and logical CPUs, RAM and GPUs of this computer ('' or NaN if not found)
m = struct('os', '', 'cpu_model', '', 'physical_cores', NaN, 'logical_cpus', NaN, 'ram_gb', NaN, 'gpus', {{}});
try
    text = evalc('feature(''numcores'')');   % "MATLAB detected: N physical cores. ... M logical cores."
    m.physical_cores = first_number(text, 'detected:\s*(\d+)\s*physical');
    m.logical_cpus = first_number(text, 'detected:\s*(\d+)\s*logical');
    if isnan(m.physical_cores)
        m.physical_cores = feature('numcores');
    end
catch
end
gpus = '';
try
    if ispc
        m.os = system_text('powershell -NoProfile -Command "(Get-CimInstance Win32_OperatingSystem).Caption"');
        m.cpu_model = system_text('powershell -NoProfile -Command "(Get-CimInstance Win32_Processor).Name"');
        if isnan(m.logical_cpus)
            m.logical_cpus = str2double(getenv('NUMBER_OF_PROCESSORS'));
        end
        [~, sys] = memory;
        m.ram_gb = sys.PhysicalMemory.Total / 1e9;
        gpus = system_text('powershell -NoProfile -Command "(Get-CimInstance Win32_VideoController).Name"');
    elseif ismac
        m.os = ['macOS ' system_text('sw_vers -productVersion')];
        m.cpu_model = system_text('sysctl -n machdep.cpu.brand_string');
        if isnan(m.logical_cpus)
            m.logical_cpus = str2double(system_text('sysctl -n hw.logicalcpu'));
        end
        m.ram_gb = str2double(system_text('sysctl -n hw.memsize')) / 1e9;
        gpus = system_text('system_profiler SPDisplaysDataType | grep "Chipset Model"');
    else
        m.os = strtrim([first_token(read_text_safe('/etc/os-release'), 'PRETTY_NAME="?([^"\n]*)') ' ' ...
                        system_text('uname -r')]);
        cpuinfo = read_text_safe('/proc/cpuinfo');
        m.cpu_model = strtrim(first_token(cpuinfo, 'model name\s*:\s*([^\n]*)'));
        if isempty(m.cpu_model)
            m.cpu_model = strtrim(first_token(system_text('lscpu'), 'Model name:\s*([^\n]*)'));
        end
        if isnan(m.logical_cpus)
            m.logical_cpus = numel(regexp(cpuinfo, '^processor\s*:', 'lineanchors'));
        end
        m.ram_gb = first_number(read_text_safe('/proc/meminfo'), 'MemTotal:\s*(\d+)') * 1024 / 1e9;
        gpus = system_text('lspci | grep -iE "vga|3d controller|display"');
    end
catch
end
nvidia = system_text('nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader');
if ~isempty(nvidia)
    gpus = nvidia;
end
items = strtrim(regexp(gpus, '\r?\n', 'split'));
m.gpus = items(~cellfun(@isempty, items));
end


function write_environment(outputs, env)
write_json(fullfile(outputs, 'logs', 'sherm', ['environment_' env.run_id '.json']), env);
write_json(fullfile(outputs, 'sherm_environment.json'), env);   % latest execution
end


function ok = reset_peak_memory()
% Linux: reset the peak resident memory (VmHWM) of this MATLAB process, so that it measures one scan
ok = false;
if ~isunix || ismac
    return
end
try
    f = fopen('/proc/self/clear_refs', 'w');
    if f >= 0
        fprintf(f, '5');
        ok = fclose(f) == 0;
    end
catch
end
end


function gb = peak_memory_gb()
% Linux: peak resident memory of this MATLAB process since reset_peak_memory (MATLAB itself included)
gb = first_number(read_text_safe('/proc/self/status'), 'VmHWM:\s*(\d+)') * 1024 / 1e9;
end


function [load_1min, available_gb] = system_load()
% load of the whole computer at the end of a scan: 1-minute load average (Linux, macOS) and available RAM
% (Linux, Windows); shows whether other jobs were running
load_1min = NaN;
available_gb = NaN;
try
    if ispc
        [~, sys] = memory;
        available_gb = sys.PhysicalMemory.Available / 1e9;
    elseif ismac
        values = sscanf(regexprep(system_text('sysctl -n vm.loadavg'), '[{}]', ''), '%f');
        load_1min = values(1);
    else
        values = sscanf(read_text('/proc/loadavg'), '%f');
        load_1min = values(1);
        available_gb = first_number(read_text('/proc/meminfo'), 'MemAvailable:\s*(\d+)') * 1024 / 1e9;
    end
catch
end
end


function check_setup(datasets, sherm_dir)
if verLessThan('matlab', '9.2')
    error('run_sherm_samson:setup', 'MATLAB R2017a or later is needed (imresize3).');
end
needed = {'load_untouch_nii', 'save_untouch_nii', 'vl_mser', 'vl_erfill', 'vl_version', 'sherm', ...
          'sherm_parallel_2d', 'imresize3', 'imopen', 'bwlabeln', 'imwarp', 'affine3d', 'imref3d', 'pca'};
missing = needed(cellfun(@(name) exist(name) == 0, needed)); %#ok<EXIST>
if ~isempty(missing)
    error('run_sherm_samson:setup', 'Not found on the MATLAB path: %s (see README.md).', strjoin(missing, ', '));
end
try
    vl_version;
catch err
    error('run_sherm_samson:setup', 'VLFeat MEX files do not run on this computer (%s); see README.md.', err.message);
end
if ~exist(fullfile(datasets, 'dataset_settings.json'), 'file')
    error('run_sherm_samson:setup', 'dataset_settings.json not found in %s.', datasets);
end
if ~exist(fullfile(sherm_dir, 'rodent_brain_polar_descriptor.mat'), 'file')
    error('run_sherm_samson:setup', 'rodent_brain_polar_descriptor.mat not found in %s.', sherm_dir);
end
end

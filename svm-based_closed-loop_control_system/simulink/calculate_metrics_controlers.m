%% Control Performance Evaluation Script
% This script performs simulations for different control scenarios using
% both a SVM-based Generalized Predictive Controller (GPC) and a PID controller.
% It evaluates control performance under:
%   1. Fixed reference values (40, 50, 60)
%   2. Time-varying (triangular) reference signals
%   3. Feedback signals with Gaussian noise
%   4. Simulated infusion loss
% 
% Performance metrics calculated:
%   - Undershoot
%   - Mean Squared Error (MSE)
%   - Integral Absolute Error (IAE)
%   - Integral Squared Error (ISE)
%   - Settling time within ±5% and ±20% (when applicable)
%
% Results are saved as CSV files for each scenario and summarized in
% a final table (`control_metrics.csv`) for analysis.

%% Load necessary data
load('mats/control_init_vector_22.mat');
load('mats/LAG_70.mat');
load('../models/svr_narx_model_v10.mat');

fprintf('\nData loaded correctly');

%% Load hyperparameters
% SVM-based GPC parameters
INTEGRATIVE_PENALTY = 0.0003;
Q = 1.6;
DERIVATIVE_PENALTY = 2.5;
S = 0.0022;
MAX_ITER = 15;
EPSILON = 0.00005;
DELTA = 0.001;

% PID Controller parameters
Kp = 2.5;
Ki = 0.1;
Kd = 0.1;

SATURATION = 22;
%% Run fixed-reference simulations
REFs = [40, 50, 60];
duration = 60 * 60; % seconds

all_metrics = [];

for r = 1:length(REFs)
    ref = REFs(r);

    % Simulate SVM architecture
    out_svm = sim('control_architecture.slx');
    fprintf('\nSVM-based GPC Simulation finished successfully for ref = %d', ref);

    % Simulate PID architecture
    out_PID = sim('PID_controller.slx');
    fprintf('\nPID Simulation finished successfully for ref = %d', ref);

    % Extract BIS signals
    BIS_SVM = out_svm.get('BIS_Real_SVM');
    BIS_PID = out_PID.get('BIS_Real_PID');

    % Times
    t_svm = linspace(0, duration, length(BIS_SVM))';
    t_pid = linspace(0, duration, length(BIS_PID))';

    % Infusion rates
    inf_svm = out_svm.get('InfusionRate_SVM');
    inf_pid = out_PID.get('InfusionRate_PID');
   
    % Calculate minimum distance
    len_svm = min([length(t_svm), length(BIS_SVM), length(inf_svm)]);
    len_pid = min([length(t_pid), length(BIS_PID), length(inf_pid)]);

    % Save CSVs
    T_svm = table(t_svm(1:len_svm), BIS_SVM(1:len_svm), inf_svm(1:len_svm), ...
        'VariableNames', {'Time_s', 'BIS', 'InfusionRate'});
    T_pid = table(t_pid(1:len_pid), BIS_PID(1:len_pid), inf_pid(1:len_pid), ...
        'VariableNames', {'Time_s', 'BIS', 'InfusionRate'});
    
    writetable(T_svm, sprintf('../data/BIS_data_SVM_ref_%d.csv', ref));
    writetable(T_pid, sprintf('../data/BIS_data_PID_ref_%d.csv', ref));

    % Métrics for both controllers
    controllers = {'SVM-based', 'PID'};
    Y_all = {BIS_SVM, BIS_PID};
    t_all = {t_svm, t_pid};

    for i = 1:2
        name = controllers{i};
        y = Y_all{i};
        t = t_all{i};

        % --- Undershoot ---
        undershoot = abs(min(y) - ref) / ref * 100;

        % --- MSE ---
        mse = mean((y - ref).^2);

        % --- IAE ---
        iae = trapz(t, abs(y - ref));

        % --- ISE ---
        ise = trapz(t, (y - ref).^2);

        % --- Settling time (±5%) ---
        tolerance = 0.05 * ref;
        idx_settle = find(abs(y - ref) > tolerance, 1, 'last');
        settling_time_5 = isempty(idx_settle) * 0 + ~isempty(idx_settle) * t(idx_settle);

        % --- Settling time (±20%) ---
        tolerance = 0.2 * ref;
        idx_settle = find(abs(y - ref) > tolerance, 1, 'last');
        settling_time_20 = isempty(idx_settle) * 0 + ~isempty(idx_settle) * t(idx_settle);

        % Save metrics
        metrics_struct = struct( ...
            'Controller', name, ...
            'Reference', ref, ...
            'Undershoot', undershoot, ...
            'SettlingTime_20', settling_time_20 / 60, ...
            'SettlingTime_5', settling_time_5 / 60, ...
            'MSE', mse, ...
            'IAE', iae, ...
            'ISE', ise ...
        );

        all_metrics = [all_metrics; metrics_struct];
    end
end

%% Run variable-reference simulations
ref_str = 'Triangular';

out_svm = sim('control_architecture_triangular_ref.slx');
fprintf('\nSVM-based GPC Simulation (triangular) finished.');

out_PID = sim('PID_controller_triangular_ref.slx');
fprintf('\nPID Simulation (triangular) finished.');

% Extract signals
BIS_SVM = out_svm.get('BIS_Real_SVM');
REF_signal_SVM = out_svm.get('ref_signal_svm');
inf_svm = out_svm.get('InfusionRate_SVM');

BIS_PID = out_PID.get('BIS_Real_PID');
REF_signal_PID = out_PID.get('ref_signal_pid');
inf_pid = out_PID.get('InfusionRate_PID');

% Times
t_svm = linspace(0, duration, length(BIS_SVM))';
t_pid = linspace(0, duration, length(BIS_PID))';

% Adjust minimum distance
len_svm = min([length(BIS_SVM), length(REF_signal_SVM), length(inf_svm)]);
len_pid = min([length(BIS_PID), length(REF_signal_PID), length(inf_pid)]);

% Save CSVs
T_svm = table( ...
    t_svm(1:len_svm), ...
    BIS_SVM(1:len_svm), ...
    REF_signal_SVM(1:len_svm), ...
    inf_svm(1:len_svm), ...
    'VariableNames', {'Time_s', 'BIS', 'Reference', 'InfusionRate'});

T_pid = table( ...
    t_pid(1:len_pid), ...
    BIS_PID(1:len_pid), ...
    REF_signal_PID(1:len_pid), ...
    inf_pid(1:len_pid), ...
    'VariableNames', {'Time_s', 'BIS', 'Reference', 'InfusionRate'});

% Export
writetable(T_svm, '../data/BIS_data_SVM_triangular.csv');
writetable(T_pid, '../data/BIS_data_PID_triangular.csv');

controllers = {'SVM-based', 'PID'};
Y_all = {BIS_SVM, BIS_PID};
t_all = {t_svm, t_pid};

for i = 1:2
    name = controllers{i};
    y = Y_all{i};
    t = t_all{i};

    % Use real reference signal
    if i == 1
        len = min(length(REF_signal_SVM), length(y));
        y_ref = REF_signal_SVM(1:len);
        y = y(1:len);
        t = t(1:len);
    else
        len = min(length(REF_signal_PID), length(y));
        y_ref = REF_signal_PID(1:len);
        y = y(1:len);
        t = t(1:len);
    end

    undershoot = abs(min(y) - min(y_ref)) / mean(y_ref) * 100;
    mse = mean((y - y_ref).^2);
    iae = trapz(t, abs(y - y_ref));
    ise = trapz(t, (y - y_ref).^2);

    % It makes no sense to calculate the settling time for variable
    % reference
    settling_5 = NaN;
    settling_20 = NaN;

    metrics_struct = struct( ...
        'Controller', name, ...
        'Reference', ref_str, ...
        'Undershoot', undershoot, ...
        'SettlingTime_20', settling_20, ...
        'SettlingTime_5', settling_5, ...
        'MSE', mse, ...
        'IAE', iae, ...
        'ISE', ise ...
    );

    all_metrics = [all_metrics; metrics_struct];
end

%% Run gaussian noise in feedback loop simulations
ref = 50;

% Simulate SVM architecture
out_svm = sim('control_architecture_gaussian_noise.slx');
fprintf('\nSVM-based GPC with gaussian noise Simulation finished successfully');

% Simulate PID architecture
out_PID = sim('PID_controller_gaussian_noise.slx');
fprintf('\nPID with gaussian noise Simulation finished successfully');

% Extract BIS signals
BIS_SVM = out_svm.get('BIS_Real_SVM');
BIS_PID = out_PID.get('BIS_Real_PID');
inf_svm = out_svm.get('InfusionRate_SVM');
inf_pid = out_PID.get('InfusionRate_PID');

% Calculate minimum distance
len_svm = min([length(t_svm), length(BIS_SVM), length(inf_svm)]);
len_pid = min([length(t_pid), length(BIS_PID), length(inf_pid)]);

% Save CSVs
T_svm = table(t_svm(1:len_svm), BIS_SVM(1:len_svm), inf_svm(1:len_svm), ...
    'VariableNames', {'Time_s', 'BIS', 'InfusionRate'});
T_pid = table(t_pid(1:len_pid), BIS_PID(1:len_pid), inf_pid(1:len_pid), ...
    'VariableNames', {'Time_s', 'BIS', 'InfusionRate'});

writetable(T_svm, '../data/BIS_data_SVM_noise.csv');
writetable(T_pid, '../data/BIS_data_PID_noise.csv');

% Times
t_svm = linspace(0, duration, length(BIS_SVM))';
t_pid = linspace(0, duration, length(BIS_PID))';

% Metrics for both controllers
controllers = {'SVM-based', 'PID'};
Y_all = {BIS_SVM, BIS_PID};
t_all = {t_svm, t_pid};

for i = 1:2
    name = controllers{i};
    y = Y_all{i};
    t = t_all{i};

    % --- Undershoot ---
    undershoot = abs(min(y) - ref) / ref * 100;

    % --- MSE ---
    mse = mean((y - ref).^2);

    % --- IAE ---
    iae = trapz(t, abs(y - ref));

    % --- ISE ---
    ise = trapz(t, (y - ref).^2);

    % --- Settling time (±5%) ---
    tolerance = 0.05 * ref;
    idx_settle = find(abs(y - ref) > tolerance, 1, 'last');
    settling_time_5 = isempty(idx_settle) * 0 + ~isempty(idx_settle) * t(idx_settle);

    % --- Settling time (±20%) ---
    tolerance = 0.2 * ref;
    idx_settle = find(abs(y - ref) > tolerance, 1, 'last');
    settling_time_20 = isempty(idx_settle) * 0 + ~isempty(idx_settle) * t(idx_settle);

    % Save metrics
    metrics_struct = struct( ...
        'Controller', name, ...
        'Reference', 'Gaussian noise', ...
        'Undershoot', undershoot, ...
        'SettlingTime_20', settling_time_20 / 60, ...
        'SettlingTime_5', settling_time_5 / 60, ...
        'MSE', mse, ...
        'IAE', iae, ...
        'ISE', ise ...
    );

    all_metrics = [all_metrics; metrics_struct];
end

%% Run infusion loss simulations
ref = 50;

% Simulate SVM architecture
out_svm = sim('control_architecture_infusion_loss.slx');
fprintf('\nSVM-based GPC with infusion loss Simulation finished successfully');

% Simulate PID architecture
out_PID = sim('PID_controller_infusion_loss.slx');
fprintf('\nPID with infusion loss Simulation finished successfully');

% Extract BIS and infusion signals
BIS_SVM = out_svm.get('BIS_Real_SVM');
BIS_PID = out_PID.get('BIS_Real_PID');
inf_svm = out_svm.get('InfusionRate_SVM');
inf_pid = out_PID.get('InfusionRate_PID');

% Calculate minimum distance
len_svm = min([length(t_svm), length(BIS_SVM), length(inf_svm)]);
len_pid = min([length(t_pid), length(BIS_PID), length(inf_pid)]);

% Save CSVs
T_svm = table(t_svm(1:len_svm), BIS_SVM(1:len_svm), inf_svm(1:len_svm), ...
    'VariableNames', {'Time_s', 'BIS', 'InfusionRate'});
T_pid = table(t_pid(1:len_pid), BIS_PID(1:len_pid), inf_pid(1:len_pid), ...
    'VariableNames', {'Time_s', 'BIS', 'InfusionRate'});

writetable(T_svm, '../data/BIS_data_SVM_infusion_loss.csv');
writetable(T_pid, '../data/BIS_data_PID_infusion_loss.csv');


% Times
t_svm = linspace(0, duration, length(BIS_SVM))';
t_pid = linspace(0, duration, length(BIS_PID))';

% Metrics for both controllers
controllers = {'SVM-based', 'PID'};
Y_all = {BIS_SVM, BIS_PID};
t_all = {t_svm, t_pid};

for i = 1:2
    name = controllers{i};
    y = Y_all{i};
    t = t_all{i};

    % --- Undershoot ---
    undershoot = abs(min(y) - ref) / ref * 100;

    % --- MSE ---
    mse = mean((y - ref).^2);

    % --- IAE ---
    iae = trapz(t, abs(y - ref));

    % --- ISE ---
    ise = trapz(t, (y - ref).^2);

    % --- Settling time (±5%) ---
    tolerance = 0.05 * ref;
    idx_settle = find(abs(y - ref) > tolerance, 1, 'last');
    settling_time_5 = isempty(idx_settle) * 0 + ~isempty(idx_settle) * t(idx_settle);

    % --- Settling time (±20%) ---
    tolerance = 0.2 * ref;
    idx_settle = find(abs(y - ref) > tolerance, 1, 'last');
    settling_time_20 = isempty(idx_settle) * 0 + ~isempty(idx_settle) * t(idx_settle);

    % Save metrics
    metrics_struct = struct( ...
        'Controller', name, ...
        'Reference', 'Infusion loss', ...
        'Undershoot', undershoot, ...
        'SettlingTime_20', settling_time_20 / 60, ...
        'SettlingTime_5', settling_time_5 / 60, ...
        'MSE', mse, ...
        'IAE', iae, ...
        'ISE', ise ...
    );

    all_metrics = [all_metrics; metrics_struct];
end

%% Save final table
metrics_table = struct2table(all_metrics);

if ~exist('results', 'dir')
    mkdir('results');
end

writetable(metrics_table, 'results/control_metrics.csv');
fprintf('\nAll metrics calculated and saved in "control_metrics.csv"\n');


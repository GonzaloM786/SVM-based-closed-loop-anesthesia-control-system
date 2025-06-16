%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Script: train_svr.m
%
% Description:
% This script trains a Support Vector Regression (SVR) model following a 
% NARX (Nonlinear AutoRegressive with eXogenous input) structure to 
% predict the patient's BIS (Bispectral Index) level based on past BIS 
% values and propofol infusion rates. It performs data preprocessing, 
% normalization, training, validation with perturbation analysis, and 
% test evaluation. The script also saves the trained model and prediction 
% results to disk.
%
% Inputs:
% - Infusion and BIS data from CSV files (one per simulation scenario)
%
% Outputs:
% - Trained SVR model
% - Evaluation metrics (MSE, MAE, R²)
% - Prediction plots for validation and test sets
% - CSV files with predictions
%
% Dependencies:
% - Requires MATLAB's Statistics and Machine Learning Toolbox
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% ---------------------------- Parameters ----------------------------

NU = 70;             % Number of lags for infusion rate (u)
NY = 70;             % Number of lags for BIS output (y)
SIM_DURATION = 90;   % Duration of each simulation in seconds
DELTA = 0.9;         % Parameter used during simulation data generation
WINDOW_SIZE = 54;    % Size of moving average window for smoothing
SIGMA = 40;          % Gaussian noise level
PERTURB = 0.001;     % Control perturbation added during validation/test

%% ---------------------------- Load Data ----------------------------

% Load data from CSV
infusion_data = readmatrix(sprintf('data/Smooth_infusion_data_0_40_%d_%.1f_%d_%.1f.csv', SIM_DURATION, DELTA, WINDOW_SIZE, SIGMA));
BIS_data = readmatrix(sprintf('data/Smooth_BIS_data_0_40_%d_%.1f_%d_%.1f.csv', SIM_DURATION, DELTA, WINDOW_SIZE, SIGMA));

% Get number of simulations and time steps
[num_sims, num_steps] = size(infusion_data);

% Define indices for training, validation, and test sets
train_idx = 1:58;
val_idx = 99;
test_idx = 100;

% Compute the number of samples per simulation
samples_per_sim = num_steps - max(NU, NY);
n_train = length(train_idx) * samples_per_sim;
n_val   = length(val_idx)   * samples_per_sim;
n_test  = length(test_idx)  * samples_per_sim;

input_dim = NU + NY + 1;  % NU + NY lags + current infusion rate

% Preallocate input/output arrays
X_train = zeros(n_train, input_dim);
Y_train = zeros(n_train, 1);
X_val   = zeros(n_val, input_dim);
Y_val   = zeros(n_val, 1);
X_test  = zeros(n_test, input_dim);
Y_test  = zeros(n_test, 1);

%% ---------------------- Build Sets -------------------------

i = 1;
for sim = train_idx
    for t = max(NU, NY)+1:num_steps
        u_lags = infusion_data(sim, t-NU:t);
        y_lags = BIS_data(sim, t-NY:t-1);
        X_train(i, :) = [u_lags, y_lags];
        Y_train(i)    = BIS_data(sim, t);
        i = i + 1;
    end
end

i = 1;
for sim = val_idx
    for t = max(NU, NY)+1:num_steps
        u_lags = infusion_data(sim, t-NU:t);
        y_lags = BIS_data(sim, t-NY:t-1);
        X_val(i, :) = [u_lags, y_lags];
        Y_val(i)    = BIS_data(sim, t);
        i = i + 1;
    end
end

i = 1;
for sim = test_idx
    for t = max(NU, NY)+1:num_steps
        u_lags = infusion_data(sim, t-NU:t);
        y_lags = BIS_data(sim, t-NY:t-1);
        X_test(i, :) = [u_lags, y_lags];
        Y_test(i)    = BIS_data(sim, t);
        i = i + 1;
    end
end

fprintf('Data preprocessing done')

%% ---------------------- Normalize Inputs/Outputs -------------------

% Minimum and maximum values
X_min = min(X_train, [], 1);
X_max = max(X_train, [], 1);
Y_min = min(Y_train);
Y_max = max(Y_train);

% Normalize
X_train_norm = (X_train - X_min) ./ (X_max - X_min);
Y_train_norm = (Y_train - Y_min) ./ (Y_max - Y_min);
X_val_norm = (X_val - X_min) ./ (X_max - X_min);
Y_val_norm = (Y_val - Y_min) ./ (Y_max - Y_min);
X_test_norm = (X_test - X_min) ./ (X_max - X_min);
Y_test_norm = (Y_test - Y_min) ./ (Y_max - Y_min);

% Visualize distributions
figure;
subplot(2,2,1); histogram(X_train_norm(:)); title('X_{train} norm distribution');
subplot(2,2,2); histogram(X_test_norm(:));  title('X_{test} norm distribution');
subplot(2,2,3); histogram(Y_train_norm);    title('Y_{train} norm distribution');
subplot(2,2,4); histogram(Y_test_norm);     title('Y_{test} norm distribution');

%% ---------------------- Train SVR Model ----------------------------

kernelFunction = 'rbf';     % Radial Basis Function kernel
boxConstraint = 1;          % BoxConstraint (C)
epsilon = 0.01;             % Insensitive loss margin

svm_model = fitrsvm(X_train_norm, Y_train_norm, ...
    'KernelFunction', kernelFunction, ...
    'Standardize', false, ...
    'BoxConstraint', boxConstraint, ...
    'Epsilon', epsilon);

fprintf('Model training done\n');

%% ---------------------- Recursive Validation ------------------------ 

% Recursive prediction on validation set
Y_val_pred_norm = zeros(size(Y_val));
Y_val_pred_norm_pert = zeros(size(Y_val));

% Initialize lags with real values
X_current = X_val_norm(1, :); 

for t = 1:length(Y_val)
    % Predict actual time input
    Y_val_pred_norm(t) = predict(svm_model, X_current);

    X_real = X_current .* (X_max - X_min) + X_min;
    X_real(NU+1) = X_real(NU+1) + PERTURB;
    X_current_pert = (X_real - X_min) ./ (X_max - X_min);
    Y_val_pred_norm_pert(t) = predict(svm_model, X_current_pert);

    % Update X_current with the new prediction
    if t < length(Y_val)
        X_current = [X_current(2:NU+1), ... 
                     X_val_norm(t+1, NU+1), ...
                     X_current(NU+3:end), ...  
                     Y_val_pred_norm(t)]; % Add new prediction
    end
end

Y_val_pred = Y_val_pred_norm .* (Y_max - Y_min) + Y_min;
Y_val_pred_pert = Y_val_pred_norm_pert .* (Y_max - Y_min) + Y_min;

% Validation metrics
MSE = mean((Y_val - Y_val_pred).^2);
MAE = mean(abs(Y_val - Y_val_pred));
SS_res = sum((Y_val - Y_val_pred).^2);
SS_tot = sum((Y_val - mean(Y_val)).^2);
R2 = 1 - (SS_res / SS_tot);

% Show metrics
fprintf('\nValidation set Metrics\n')
fprintf('MSE: %.4f\n', MSE);
fprintf('MAE: %.4f\n', MAE);
fprintf('R²: %.4f\n', R2);

% Validation set Plot
% Predicted and real BIS comparation
figure;
diff_val = Y_val_pred_pert - Y_val_pred;

% Subplot 1: BIS real, predicted, and perturbed comparison
subplot(2,1,1);
plot(Y_val, 'b', 'LineWidth', 1.5);
hold on;
plot(Y_val_pred, 'r--', 'LineWidth', 1.5);
plot(Y_val_pred_pert, 'g--', 'LineWidth', 1.5);
xlabel('Time');
ylabel('BIS');
legend('Real', 'Predicted');
title('Predicted and Real BIS comparation (Validation Set)');
grid on;
param_text = sprintf('Kernel: %s\nC: %.2f\nEpsilon: %.4f\nNº simulations: %d\nWindow size: %d\nSigma: %.1f\nMSE: %.1f\nMAE: %.2f\nR2: %.4f\nNU: %d\nNY: %d' ...
    ,kernelFunction, boxConstraint, epsilon, train_idx(end), WINDOW_SIZE, SIGMA, MSE, MAE, R2, NU, NY);

% Inserttext box
annotation('textbox', [0.82, 0.77, 0.2, 0.1], 'String', param_text, ...
    'FitBoxToText', 'on', 'BackgroundColor', 'white', 'FontSize', 10);

% Subplot 2: ΔBIS
subplot(2,1,2);
hold on;
correct_idx = diff_val < 0;
incorrect_idx = diff_val >= 0;

% Initialize vectors
plot_vals = nan(size(diff_val));
plot_vals(correct_idx) = diff_val(correct_idx);
plot(plot_vals, 'g', 'LineWidth', 1.5); % Correct zones

plot_vals = nan(size(diff_val));
plot_vals(incorrect_idx) = diff_val(incorrect_idx);
plot(plot_vals, 'r', 'LineWidth', 1.5); % Incorrect zones

xlabel('Time');
ylabel('ΔBIS_{pert} - BIS_{normal}');
title('Sign of ΔBIS prediction after control perturbation');
legend('Correct (↓ BIS)', 'Incorrect (↑ BIS)', 'Location', 'best');
grid on;

% Show error percetage
porcentaje_malos = 100 * sum(diff_val >= 0) / length(diff_val);

param_text = sprintf('Incorrect percentage: %.2f\nPerturbation: %.3f', porcentaje_malos, PERTURB);

% Insertar text box
annotation('textbox', [0.3, 0.35, 0.2, 0.1], 'String', param_text, ...
    'FitBoxToText', 'on', 'BackgroundColor', 'white', 'FontSize', 10);

% Save predictions CSV (Validación)
val_table = table((1:length(Y_val))', Y_val, Y_val_pred, Y_val_pred_pert, ...
    Y_val_pred_pert - Y_val_pred, ...
    'VariableNames', {'Time', 'Y_real', 'Y_pred', 'Y_pred_pert', 'DeltaBIS'});
writetable(val_table, 'data/svr_validation_predictions.csv');
fprintf('Saved: data/svr_validation_predictions.csv\n');


%% ---------------------- Recursive Test ------------------------ 

% Recursive prediction on test set
Y_test_pred_norm = zeros(size(Y_test));
Y_test_pred_norm_pert = zeros(size(Y_test));

% Initialize lags with real values
X_current = X_test_norm(1, :); 

for t = 1:length(Y_test)
    % Predict actual time input
    Y_test_pred_norm(t) = predict(svm_model, X_current);
    
    X_real = X_current .* (X_max - X_min) + X_min;
    X_real(NU+1) = X_real(NU+1) + PERTURB;
    X_current_pert = (X_real - X_min) ./ (X_max - X_min);
    Y_test_pred_norm_pert(t) = predict(svm_model, X_current_pert);

    % Update X_current with new predictions
    if t < length(Y_test)
        X_current = [X_current(2:NU+1), ... 
                     X_test_norm(t+1, NU+1), ...
                     X_current(NU+3:end), ...  
                     Y_test_pred_norm(t)]; % Add new prediction
    end
end

Y_test_pred = Y_test_pred_norm .* (Y_max - Y_min) + Y_min;
Y_test_pred_pert = Y_test_pred_norm_pert .* (Y_max - Y_min) + Y_min;

% Validation metrics
MSE = mean((Y_test - Y_test_pred).^2);
MAE = mean(abs(Y_test - Y_test_pred));
SS_res = sum((Y_test - Y_test_pred).^2);
SS_tot = sum((Y_test - mean(Y_test)).^2);
R2 = 1 - (SS_res / SS_tot);

% Show metrics
fprintf('\nTest set Metrics\n')
fprintf('MSE: %.4f\n', MSE);
fprintf('MAE: %.4f\n', MAE);
fprintf('R²: %.4f\n', R2);

% Test set Plot
% Predicted and real BIS comparation
figure;
diff_val = Y_test_pred_pert - Y_test_pred;

% Subplot 1: BIS real, predicted, y perturbed comparison
subplot(2,1,1);
plot(Y_test, 'b', 'LineWidth', 1.5);
hold on;
plot(Y_test_pred, 'r--', 'LineWidth', 1.5);
xlabel('Time');
ylabel('BIS');
legend('Real', 'Predicted');
title('Predicted and Real BIS comparation (Test Set)');
grid on;
param_text = sprintf('Kernel: %s\nC: %.2f\nEpsilon: %.4f\nNº simulations: %d\nWindow size: %d\nSigma: %.1f\nMSE: %.1f\nMAE: %.2f\nR2: %.4f\nNU: %d\nNY: %d' ...
    ,kernelFunction, boxConstraint, epsilon, train_idx(end), WINDOW_SIZE, SIGMA, MSE, MAE, R2, NU, NY);

% Insert rtext box
annotation('textbox', [0.82, 0.77, 0.2, 0.1], 'String', param_text, ...
    'FitBoxToText', 'on', 'BackgroundColor', 'white', 'FontSize', 10);

% Subplot 2: ΔBIS
subplot(2,1,2);
hold on;
correct_idx = diff_val < 0;
incorrect_idx = diff_val >= 0;

% Initialize vectors
plot_vals = nan(size(diff_val));
plot_vals(correct_idx) = diff_val(correct_idx);
plot(plot_vals, 'g', 'LineWidth', 1.5); % Correct zones

plot_vals = nan(size(diff_val));
plot_vals(incorrect_idx) = diff_val(incorrect_idx);
plot(plot_vals, 'r', 'LineWidth', 1.5); % Incorrect zones

xlabel('Time');
ylabel('ΔBIS_{pert} - BIS_{normal}');
title('Sign of ΔBIS prediction after control perturbation');
legend('Correct (↓ BIS)', 'Incorrect (↑ BIS)', 'Location', 'best');
grid on;

% Show error percentage
porcentaje_buenos = 100 * sum(diff_val < 0) / length(diff_val);

param_text = sprintf('Correct percentage: %.2f\nPerturbation: %.3f', porcentaje_buenos, PERTURB);
% Insertar textbox
annotation('textbox', [0.3, 0.35, 0.2, 0.1], 'String', param_text, ...
    'FitBoxToText', 'on', 'BackgroundColor', 'white', 'FontSize', 10);

% Save predictions CSV (Test)
test_table = table((1:length(Y_test))', Y_test, Y_test_pred, Y_test_pred_pert, ...
    Y_test_pred_pert - Y_test_pred, ...
    'VariableNames', {'Time', 'Y_real', 'Y_pred', 'Y_pred_pert', 'DeltaBIS'});
writetable(test_table, 'data/svr_test_predictions.csv');
fprintf('Saved: data/svr_test_predictions.csv\n');

%% ---------------------- Save Model ---------------------------------

if ~exist('models', 'dir')
    mkdir('models');
end

save('models/svr_narx_model_v11.mat', 'svm_model', ...
    'X_min', 'X_max', 'Y_min', 'Y_max');

fprintf('Model saved correctly\n');
disp(svm_model);
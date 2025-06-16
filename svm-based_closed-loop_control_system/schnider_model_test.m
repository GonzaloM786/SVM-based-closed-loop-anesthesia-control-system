% Load parameters
params = config_schnider();
TYPE = 1; % 1 = default; 2 = steady infusion

% Simulation time (minutes)
t_total = 60 * 1; % mins
T = linspace(0, t_total, 10000);

% Typical infusion profile (mg/min)
infusion_rate = zeros(size(T));
if TYPE == 1
    for i = 1:length(T)
        %t = T(i) / 60; % Convert to minutes
        if T(i) <= 5
            infusion_rate(i) = 22; % High infusion for induction
        elseif T(i) <= 20
            infusion_rate(i) = 18; % Maintenance phase
        elseif T(i) <= 60
            infusion_rate(i) = 10; % Lower maintenance phase
        else 
            infusion_rate(i) = 0; % End of surgery  
        end
    end
elseif TYPE == 2
    for i = 1:length(T)
        infusion_rate(i) = 22;
    end
end

% Initial conditions [Plasma, Rapid Peripheral, Slow Peripheral, Effect Site]
C0 = [0, 0, 0, 0];

% Solve differential equations
[t, C] = ode45(@(t, C) modeloPK(t, C, params, T, infusion_rate), T, C0);

% Extract compartments
C1 = C(:,1);
C2 = C(:,2);
C3 = C(:,3);
Ce = C(:,4);

% Calculate BIS
BIS = params.BIS_base * (params.Ce50^params.gamma) ./ (params.Ce50^params.gamma + Ce.^params.gamma);

% Create table to export the data
results_table = table( ...
    t, ...
    C1, ...
    C2, ...
    C3, ...
    Ce, ...
    BIS, ...
    infusion_rate', ...
    'VariableNames', {'Time_min', 'C1_Plasma', 'C2_RapidPeripheral', 'C3_SlowPeripheral', 'Ce_EffectSite', 'BIS', 'InfusionRate'} ...
);

% Create directory if it does not exist
output_dir = './data/';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Save CSV
writetable(results_table, fullfile(output_dir, 'schnider_simulation.csv'));

fprintf('Datos guardados en: %s\n', fullfile(output_dir, 'schnider_simulation.csv'));

% Plot results
figure;

% Compartments and effect site concentratios
subplot(2,2,1);
plot(t, C1, 'b', t, C2, 'r', t, C3, 'g', t, Ce, 'k', 'LineWidth', 2);
legend('C1 (Plasma)', 'C2 (Rapid Peripheral)', 'C3 (Slow Peripheral)', 'Ce (Effect Site)');
xlabel('Time (min)');
ylabel('Concentration (mg/L)');
title('Schnider PK Model - Compartment Concentrations');
grid on;

% BIS
subplot(2,2,2);
plot(t, BIS, 'm', 'LineWidth', 2);
xlabel('Time (min)');
ylabel('BIS');
title('Schnider PD Model - BIS Index');
grid on;

% Plasma vs Effect site concentrations
subplot(2,2,3);
plot(t, C1, '-b', 'LineWidth', 2);  % Plasma 
hold on;
plot(t, Ce, '-r', 'LineWidth', 1);   % Effect site
legend('C1 (Plasma)', 'Ce (Effect site)');
xlabel('Time (min)');
ylabel('Concentration (mg/L)');
title('Comparation Plasma vs Effect site');
grid on;
ylim([0 max(C(:,1))*1.2]);

% Infusion profile
subplot(2,2,4);
plot(t, infusion_rate, 'k', 'LineWidth', 2);
xlabel('Time (min)');
ylabel('Infusion rate (mg/min)');
title('Infusion profile administrated');
grid on;

% PK model function
function dCdt = modeloPK(t, C, params, T, infusion_rate)
    % Infusion rate interpolation
    infusion = interp1(T, infusion_rate, t);
    
    % Differential equations
    dCdt = zeros(4,1);
    dCdt(1) = (infusion/params.V1) - params.k10*C(1) - params.k12*C(1) + params.k21*C(2) - params.k13*C(1) + params.k31*C(3);
    dCdt(2) = params.k12*C(1) - params.k21*C(2);
    dCdt(3) = params.k13*C(1) - params.k31*C(3);
    dCdt(4) = params.ke0 * (C(1) - C(4)); % Effect-site
end
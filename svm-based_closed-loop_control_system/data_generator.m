% Data for the SVM model generator

% Parameters
INFUSION_MIN = 0; % mg/min
INFUSION_MAX = 40; % mg/min
DURATION_MIN = 1; % Seconds
DURATION_MAX = 360; % Seconds
SIM_DURATION = 45; % Minutes
NUM_SIMULATIONS = 20;

generate_training_data(INFUSION_MIN, INFUSION_MAX, DURATION_MIN, DURATION_MAX, SIM_DURATION, NUM_SIMULATIONS);

% Function to generate training data for the SVM model
function generate_training_data(infusion_min, infusion_max, duration_min, duration_max, sim_duration, num_simulations)
    % Set seed
    rng(113);

    % Time vector (1 point per second)
    T = linspace(0, sim_duration * 60, sim_duration * 60); % Seconds

    % Matrix to store infusion profiles, BIS outputs and concentrations
    all_infusion = zeros(num_simulations, length(T));
    all_BIS = zeros(num_simulations, length(T));
    all_C1 = zeros(num_simulations, length(T));
    all_C2 = zeros(num_simulations, length(T));
    all_C3 = zeros(num_simulations, length(T));
    all_Ce = zeros(num_simulations, length(T));

    for sim = 1:num_simulations
        infusion_rate = zeros(size(T));
        t_current = 0;

        % Generate random infusion profile
        while t_current < sim_duration * 60
            pulse_duration = randi([duration_min, duration_max]); % Convert seconds to minutes
            pulse_value = rand() * (infusion_max - infusion_min) + infusion_min;

            end_time = min(t_current + pulse_duration, sim_duration * 60);
            infusion_rate(t_current + 1:end_time) = pulse_value;
            t_current = end_time;
        end

        % Store infusion profile
        all_infusion(sim, :) = infusion_rate;

        % Simulate BIS response using Schnider model
        params = config_schnider();
        C0 = [0, 0, 0, 0];
        [~, C] = ode45(@(t, C) modeloPK(t, C, params, T/60, infusion_rate), T/60, C0);
        
        all_C1(sim, :) = C(:, 1);
        all_C2(sim, :) = C(:, 2);
        all_C3(sim, :) = C(:, 3);
        all_Ce(sim, :) = C(:, 4);
        
        BIS = params.BIS_base * (params.Ce50^params.gamma) ./ (params.Ce50^params.gamma + C(:, 4).^params.gamma);
        all_BIS(sim, :) = BIS';
    end

    % Save data to CSV files
    if ~exist('data', 'dir')
        mkdir('data');
    end
    writematrix(all_infusion, sprintf('data/infusion_data_%d_%d_%d_%d.csv', infusion_min, infusion_max, duration_min, duration_max));
    writematrix(all_BIS, sprintf('data/BIS_data_%d_%d_%d_%d.csv', infusion_min, infusion_max, duration_min, duration_max));
    writematrix(all_C1, sprintf('data/C1_data_%d_%d_%d_%d.csv', infusion_min, infusion_max, duration_min, duration_max));
    writematrix(all_C2, sprintf('data/C2_data_%d_%d_%d_%d.csv', infusion_min, infusion_max, duration_min, duration_max));
    writematrix(all_C3, sprintf('data/C3_data_%d_%d_%d_%d.csv', infusion_min, infusion_max, duration_min, duration_max));
    writematrix(all_Ce, sprintf('data/Ce_data_%d_%d_%d_%d.csv', infusion_min, infusion_max, duration_min, duration_max));


    % Plot infusion profiles
    figure;
    hold on;
    for sim = 1:num_simulations
        plot(T / 60, all_infusion(sim, :));
    end
    xlabel('Time (min)');
    ylabel('Infusion rate (mg/min)');
    title('nfusion Profiles');
    grid on;
    hold off;
    set(gcf, 'Units', 'inches', 'Position', [0, 0, 5, 5]);
    print('../assets/infusion_profiles', '-dpdf', '-bestfit');

    % Plot first 3 BIS profiles
    figure;
    hold on;
    for sim = 1:num_simulations
        plot(T / 60, all_BIS(sim, :));
    end
    xlabel('Time (min)');
    ylabel('BIS');
    title('BIS Profiles');
    grid on;
    hold off;
    set(gcf, 'Units', 'inches', 'Position', [0, 0, 5, 5]);
    print('../assets/BIS_profiles', '-dpdf', '-bestfit');
end

% PK model function
function dCdt = modeloPK(t, C, params, T, infusion_rate)
    % Infusion rate interpolation
    infusion = interp1(T, infusion_rate, t);
    
    % Differential equations
    dCdt = zeros(4,1);
    dCdt(1) = (infusion/params.V1) - params.k10*C(1) - params.k12*C(1) + params.k21*C(2) - params.k13*C(1) + params.k31*C(3);
    dCdt(2) = params.k12*C(1) - params.k21*C(2);
    dCdt(3) = params.k13*C(1) - params.k31*C(3);
    dCdt(4) = params.ke0 * (C(1) - C(4)); % Effect site
end
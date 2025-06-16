% Data for the SVM model generator

% Parameters
INFUSION_MIN = 0; % mg/min
INFUSION_MAX = 40; % mg/min
SIM_DURATION = 90; % Minutes
NUM_SIMULATIONS = 100;
DELTA = 0.9;
WINDOW_SIZE = 54;
SIGMA = 40;

generate_training_data(INFUSION_MIN, INFUSION_MAX, SIM_DURATION, NUM_SIMULATIONS, DELTA, WINDOW_SIZE, SIGMA);

% Function to generate training data for the SVM model
function generate_training_data(infusion_min, infusion_max, sim_duration, num_simulations, DELTA, window_size, sigma)
    % Set seed
    rng(113);

    % Time vector (1 point per second)
    T = linspace(0, sim_duration * 60, sim_duration * 60); % Seconds

    % Matrix to store infusion profiles, BIS outputs and concentrations
    all_infusion = zeros(num_simulations, length(T));
    all_infusion_non_filtered = zeros(num_simulations, length(T));
    all_BIS = zeros(num_simulations, length(T));
    all_C1 = zeros(num_simulations, length(T));
    all_C2 = zeros(num_simulations, length(T));
    all_C3 = zeros(num_simulations, length(T));
    all_Ce = zeros(num_simulations, length(T));

    for sim = 1:num_simulations
        % New: Smooth random walk infusion profile
        infusion_rate = zeros(size(T));
        %infusion_rate(1) = rand() * (infusion_max - infusion_min) + infusion_min;  % Random initial value
        infusion_rate(1) = 22;

        for i = 2:length(T)
            delta = (rand() - 0.5) * 2 * DELTA;  % Random change in [-DELTA, DELTA]
            infusion_rate(i) = infusion_rate(i - 1) + delta;
        
            % Clamp to [infusion_min, infusion_max]
            infusion_rate(i) = max(infusion_min, min(infusion_max, infusion_rate(i)));
        end

        all_infusion_non_filtered(sim, :) = infusion_rate;
        
        % Gaussian filter
        x = -floor(window_size/2):floor(window_size/2);
        gaussian_kernel = exp(-(x.^2)/(2*sigma^2));
        gaussian_kernel = gaussian_kernel / sum(gaussian_kernel);
        
        % Padding by border replication
        pad_size = floor(window_size/2);
        infusion_rate = padarray(infusion_rate, [0 pad_size], 'replicate');
        
        % Apply convolution
        infusion_rate = conv(infusion_rate, gaussian_kernel, 'valid');
        
        % Save filtered profile
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
    writematrix(all_infusion, sprintf('data/Smooth_infusion_data_%d_%d_%d_%.1f_%d_%.1f.csv', infusion_min, infusion_max, sim_duration, DELTA, window_size, sigma));
    writematrix(all_infusion_non_filtered, sprintf('data/Smooth_infusion_data_non_filtered_%d_%d_%d_%.1f_%d_%.1f.csv', infusion_min, infusion_max, sim_duration, DELTA, window_size, sigma));
    writematrix(all_BIS, sprintf('data/Smooth_BIS_data_%d_%d_%d_%.1f_%d_%.1f.csv', infusion_min, infusion_max, sim_duration, DELTA, window_size, sigma));
    writematrix(all_C1, sprintf('data/Smooth_C1_data_%d_%d_%d_%.1f_%d_%.1f.csv', infusion_min, infusion_max, sim_duration, DELTA, window_size, sigma));
    writematrix(all_C2, sprintf('data/Smooth_C2_data_%d_%d_%d_%.1f_%d_%.1f.csv', infusion_min, infusion_max, sim_duration, DELTA, window_size, sigma));
    writematrix(all_C3, sprintf('data/Smooth_C3_data_%d_%d_%d_%.1f_%d_%.1f.csv', infusion_min, infusion_max, sim_duration, DELTA, window_size, sigma));
    writematrix(all_Ce, sprintf('data/Smooth_Ce_data_%d_%d_%d_%.1f_%d_%.1f.csv', infusion_min, infusion_max, sim_duration, DELTA, window_size, sigma));


    % Plot infusion profiles
    figure;
    hold on;
    for sim = 1:num_simulations
        plot(T / 60, all_infusion(sim, :));
    end
    xlabel('Time (min)');
    ylabel('Infusion rate (mg/min)');
    title('Infusion Profiles');
    grid on;
    hold off;
    set(gcf, 'Units', 'inches', 'Position', [0, 0, 5, 5]);
    print('../assets/infusion_profiles', '-dpdf', '-bestfit');

    % Plot BIS profiles
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
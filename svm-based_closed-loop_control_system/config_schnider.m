% Schnider model configuration for propofol
function params = config_schnider()
    % Demographic parameters
    params.age = 28; % Age in years
    params.weight = 60; % Weight in kg
    params.height = 164; % Height in cm
    params.gender = "M"; % Gender (M = Male; F = Female)
    
    % Lean body mass calculation based on gender
    if strcmp(params.gender, "M")
        params.LBM = (1.1 * params.weight) - 128 * (params.weight / params.height)^2; % Lean body mass (kg)
    else
        params.LBM = (1.07 * params.weight) - 148 * (params.weight / params.height)^2; % Lean body mass (kg)
    end
    
    params.Ce50 = 4.93; % Effect site concentration for BIS 50 (mg/L)
    params.gamma = 2.46; % Slope of the concentration-response curve
    
    % Pharmacokinetic parameters of the Schnider model
    params.V1 = 4.27; % Central compartment volume (L)
    params.V2 = 18.9 - 0.391 * (params.age - 53); % Rapid peripheral compartment volume (L)
    params.V3 = 238; % Slow peripheral compartment volume (L)
    
    params.CL1 = 1.89 + ((params.weight - 77) * 0.0456) + ((params.LBM - 59) * (-0.0681)) + ((params.height - 177) * 0.0264); % Clearance of the central compartment (L/min)
    params.CL2 = 1.29 - 0.024 * (params.age - 53); % Clearance between central and rapid peripheral compartment (L/min)
    params.CL3 = 0.836; % Clearance between central and slow peripheral compartment (L/min)
    
    % Transfer rate constants (calculated from clearances and volumes)
    params.k10 = params.CL1 / params.V1;
    params.k12 = params.CL2 / params.V1;
    params.k13 = params.CL3 / params.V1;
    params.k21 = params.CL2 / params.V2;
    params.k31 = params.CL3 / params.V3;
    
    % Pharmacodynamic (PD) effect-site parameters
    params.ke0 = 0.456; % Rate constant for equilibrium between plasma and effect site (1/min)
    
    % Additional parameters
    params.BIS_base = 100; % Baseline BIS value (no drug effect)
end

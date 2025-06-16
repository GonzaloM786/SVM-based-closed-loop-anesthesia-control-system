function [cost, y_pred, int_error_out] = compute_cost(u_candidate, y_ref, model, X_min, X_max, Q, deriv_penalty, int_penalty, int_error_in, Y_min, Y_max, LAG)
% compute_cost - Calculates the cost for a single-step predictive control action using an SVM model,
% incorporating penalties on tracking error, predicted output derivative, and integral error.
%
% Inputs:
%   u_candidate   - Candidate infusion rate control input vector
%   y_ref         - Desired BIS reference value
%   model         - Trained SVM regression model for predicting BIS
%   X_min, X_max  - Normalization bounds for model input
%   Q             - Weight for the tracking error term
%   deriv_penalty - Weight for the derivative penalty (smoothness of control)
%   int_penalty   - Weight for the accumulated integral error
%   int_error_in  - Integral error from the previous time steps
%   Y_min, Y_max  - Denormalization bounds for model output
%   LAG           - Number of input lags (used to form model input vector)
%
% Outputs:
%   cost          - Computed scalar cost for the control input
%   y_pred        - Predicted BIS output for the candidate input
%   int_error_out - Updated integral error after applying the control input

    Ts = 1;
    
    % Normalize candidate input
    u_candidate_norm = (u_candidate - X_min) ./ (X_max - X_min);

    % Predict BIS using the SVM model
    y_pred = predict(model, u_candidate_norm);
    y_pred = y_pred .* (Y_max - Y_min) + Y_min;  % Denormalize prediction

    % ---------------- Instant error ------------------
    error = y_ref - y_pred;

    % --------- Integrative accumulated error ---------
    int_error_out = int_error_in + error * Ts;
    int_error_out = max(min(int_error_out, 1000), -5000);  % windup clamp

    % --------- Cost terms ---------
    tracking_error     = Q * (error)^2;
    derivative_penalty = deriv_penalty * ((y_pred - u_candidate(LAG+2))/Ts)^2;
    integral_penalty   = int_penalty * int_error_out^2;

    % Total cost
    cost = tracking_error + derivative_penalty + integral_penalty;
end

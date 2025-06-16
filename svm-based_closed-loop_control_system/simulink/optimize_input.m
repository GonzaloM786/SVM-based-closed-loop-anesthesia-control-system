function u_opt_and_pred = optimize_input(concatenated_vector)
    % optimize_input - Performs a simple gradient-based optimization to find
    % the optimal control input minimizing a cost function that balances BIS tracking,
    % smoothness, and integral error using an SVM-based predictive model.
    %
    % This function reconstructs individual parameters from a concatenated vector,
    % loads the required model and lag structure, and then applies a basic
    % perturbation-based gradient descent to optimize the candidate control input.
    %
    % Inputs:
    %   concatenated_vector - A flattened input vector containing all required data in order:
    %       [1:(LAG*2+1)]       -> Initial candidate control input vector (u_candidate)
    %       next                -> Reference BIS value (y_ref)
    %       next                -> Integral error penalty (int_penalty)
    %       next block          -> Input normalization lower bounds (X_min)
    %       next block          -> Input normalization upper bounds (X_max)
    %       next                -> Tracking error weight (Q)
    %       next                -> Derivative penalty weight (deriv_penalty)
    %       next                -> Gradient descent step size (s)
    %       next                -> Maximum number of iterations (max_iter)
    %       next                -> Convergence threshold (epsilon)
    %       next                -> Perturbation amount for gradient estimation (delta)
    %       next                -> Output denormalization lower bound (Y_min)
    %       next                -> Output denormalization upper bound (Y_max)
    %       next                -> Previous integral error (int_error_in)
    %
    % Outputs:
    %   u_opt_and_pred - A vector with:
    %       [1] -> Optimized control input at current time step
    %       [2] -> Predicted BIS corresponding to the optimized input
    %       [3] -> Updated integral error value for next control step
    %
    % Notes:
    %   - Uses a simple finite-difference method to approximate the gradient.
    %   - Applies clamping to the infusion rate to ensure it stays within [0, 25].
    %   - The SVM model is loaded once using a persistent variable to improve efficiency.
    
    % Persistently load the lag value
    persistent LAG;
    if isempty(LAG)
        tmp = load('mats/LAG_70.mat', 'LAG');
        LAG = tmp.LAG;
    end

    % Interpertar los inputs
    u_candidate   = concatenated_vector(1:(LAG*2+1)*1);
    y_ref         = concatenated_vector((LAG*2+1)*1+1);
    int_penalty    = concatenated_vector((LAG*2+1)*2+1);
    X_min         = concatenated_vector((LAG*2+1)*3+1:(LAG*2+1)*4);
    X_max         = concatenated_vector((LAG*2+1)*4+1:(LAG*2+1)*5);
    Q             = concatenated_vector((LAG*2+1)*5+1);
    deriv_penalty = concatenated_vector((LAG*2+1)*6+1);
    s             = concatenated_vector((LAG*2+1)*7+1);
    max_iter      = concatenated_vector((LAG*2+1)*8+1);
    epsilon       = concatenated_vector((LAG*2+1)*9+1);
    delta         = concatenated_vector((LAG*2+1)*10+1); 
    Y_min         = concatenated_vector((LAG*2+1)*11+1);
    Y_max         = concatenated_vector((LAG*2+1)*12+1);
    int_error_in  = concatenated_vector((LAG*2+1)*13+1);

    % Persistently load the model
    persistent model;
    if isempty(model)
        tmp = load('../models/svr_narx_model_v10.mat', 'svm_model');
        model = tmp.svm_model;
    end

    u_optimal = u_candidate;

    % Initialize the optimization loop
    [cost_prev, y_pred, ~] = compute_cost(u_candidate, y_ref, model, X_min, X_max, Q, deriv_penalty, int_penalty, int_error_in, Y_min, Y_max, LAG);  % Calculate initial cost

    for iter = 1:max_iter
        % Perturb the candidate input slightly to estimate the gradient
        u_candidate_perturbed = u_optimal;

        u_candidate_perturbed(LAG+1) = u_candidate_perturbed(LAG+1) + delta;  % Perturb the candidate input
        
        % Calculate the new cost with the perturbed input
        [cost_perturbed, y_pred_pert, ~] = compute_cost(u_candidate_perturbed, y_ref, model, X_min, X_max, Q, deriv_penalty, int_penalty, int_error_in, Y_min, Y_max, LAG);

        % Estimate the gradient 
        if cost_perturbed > cost_prev
            gradient = 1;
        else
            gradient = -1;
        end
        
        if y_pred_pert > y_pred
            gradient = -gradient;
        end

        % Update the candidate input using gradient descent
        u_optimal(LAG+1) = u_optimal(LAG+1) - s * gradient;
        u_optimal(LAG+1) = max(0, min(25, u_optimal(LAG+1)));
            
        % Calculate the new cost with the updated input
        [cost_new, y_pred, int_error_new] = compute_cost(u_optimal, y_ref, model, X_min, X_max, Q, deriv_penalty, int_penalty, int_error_in, Y_min, Y_max, LAG);

        % If the change in cost is less than epsilon, stop the optimization
        if abs(cost_prev - cost_new) < epsilon
            break;
        end

        % Update the previous cost for the next iteration
        cost_prev = cost_new;
    end

    u_opt_and_pred = [u_optimal(LAG+1), y_pred, int_error_new];
end

function [xh, P] = ukf(F,H,x_0,P_0,y,u,Rv,Rn,dt)
    %%  [xh, P] = ukf(F,H,x_0,P_0,y,u,Rv,Rn,dt)
    %
    %   This function uses the 'Unscented Kalman Filter' (UKF) to compute 
    %   the estimation of the nonlinear dynamical system given by:
    %
    %               x_{k+1} = F(x_k, u_k) + v_k        (1)
    %               y_k     = H(x_k)      + n_k        (2)
    %
    %   The algorithm is taken from Table 7.3 in (Wan, Van der Merwe).
    %
    %   Input data:
    %
    %   - F   : Function handle in (1) (L x 1 vector) 
    %           (L process equations)
    %   - H   : Function handle in (2) (Ly x 1 vector) 
    %           (Ly observation equations)
    %   - x_0 : Initial state of the dynamical system (L x 1 vector)(L states)
    %   - P_0 : Error covariance matrix of the state x_0 (L x L matrix)
    %   - y   : Measurement matrix (Ly x N matrix)
    %           (Ly observations - N time steps)
    %   - u   : Optional control input (N x g vector) 
    %           (N time steps - g control inputs)
    %   - Rv  : Process-noise discrete covariance (L x L matrix)
    %   - Rn  : Measurement-noise discrete covariance (Ly x Ly matrix)
    %   - dt  : Time between measurements
    %
    %   Output data:
    %
    %   - x_k : State estimate observational update (LxN matrix). Every 
    %           column is the estimation at every time step.
    %   - P_k : Steady-state covariance matrix (1xN cell). Every component 
    %           of the cell is a covariance matrix at every time step.
    %
    %   Notes:
    %
    %   - Note that 'y' is an "Ly x N" matrix. It means that the number of 
    %     columns is the number of states to estimate 'k'; the number of 
    %     rows is the number of measured variables at time 'k'.
    %   - It is assumed that "v_k" and "n_k" have mean zero, and that they 
    %     are uncorrelated (E[v_j n_k] = 0).
    %
    %   Bibliography:
    %
    %   - WAN, Eric A., VAN DER MERWE, Rudolph. "The Unscented Kalman Filter".
    %     In: HAYKIN, Simon. "Kalman filtering and neural networks". John 
    %     Wiley & Sons Inc. First edition. 2001. Ontario, Canada.
    %     Available in:
    %     http://stomach.v2.nl/docs/TechPubs/Tracking_and_AR/wan01unscented.pdf
    %
    % -------------------------------------------------------
    % | Developed by:   Sebastian Jaramillo Moreno          |
    % |                 sejaramillomo@unal.edu.co           |
    % |                 National University of Colombia     |
    % |                 Manizales, Colombia.                |
    % -------------------------------------------------------
    %
    %   Date: 28 - Aug - 2018

L  = length(x_0);                       % number of states
Ly = size(y,1);                         % number of observations
N  = size(y,2);                         % number of measurements

%% verification of the consistency of the data
if size(P_0,1) ~= L || size(P_0,2) ~= L
    error('The size of the matrix P_0 is not the same as the size of x_0')
end

if size(Rv,1) ~= L || size(Rv,2) ~= L
    error('The size of the matrix Rv is not the same as the size of x_0')
end
    
if size(Rn,1) ~= Ly || size(Rn,2) ~= Ly
    error('The size of the matrix Rn is not the same as the size of y(:,i)')
end

%% filter parameters
alpha = 1e-3;                           % \in [1e-4, 1]
kappa = 3 - L;                          % 0 or 3-L
beta  = 2;                              % 2 for gaussian distributions

lambda = alpha^2*(L + kappa) - L;       % scaling parameter
gamma  = sqrt(L + lambda);              % composite scaling parameter

% mean weights (eq. 7.34)
Wm        = zeros(1, 2*L + 1);          
Wm(1)     = lambda/(L + lambda);
Wm(2:end) = 1/(2*(L + lambda));

% covariance weights (eq. 7.34)
Wc    = Wm;
Wc(1) = lambda/(L + lambda) + (1 - alpha^2 + beta);
    
%% data initialization
% initial state and covariance
xh(:,0+1) = x_0;                      % eq. 7.50
P{0+1}    = P_0;                      % eq. 7.51 
    
%%
for k = (1+1):N
    %% calculate sigma points (eq. 7.52)
    [sqrt_Pkm1, err] = chol(P{k-1},'lower');
    if err ~= 0
        error('Matrix not positive definite at iteration %d.',k)
    end
    
    %{
    Xkm1 = [xh(:,k-1) ...
            xh(:,k-1) + gamma*sqrt_Pkm1 ...
            xh(:,k-1) - gamma*sqrt_Pkm1];                       
    %}
    Xkm1 = [xh(:,k-1) ...
           (repmat(xh(:,k-1),1,L) + gamma*sqrt_Pkm1)   ...
           (repmat(xh(:,k-1),1,L) - gamma*sqrt_Pkm1)];
    %}

    %% time update 
    % equation 7.53
    X_k_km1_ast = zeros(L, 2*L+1);
    for i = 1:(2*L + 1)
        X_k_km1_ast(:,i) = rk_discrete(F, Xkm1(:,i), u(:,k-1), dt);
    end

    % equation 7.54
    % xh_kb = sum(repmat(Wm,L,1) .* X_k_km1_ast, 2);
    % is equivalent to:
    xh_kb = X_k_km1_ast*Wm';
    
    % equation 7.55
    P_kb = Rv;
    for i = 1:2*L+1
        tmp  = X_k_km1_ast(:,i) - xh_kb;
        P_kb = P_kb + Wc(i)*(tmp*tmp');
    end
    
    %% redraw sigma points
    [sqrt_Pkb, err] = chol(P_kb,'lower');
    if err ~= 0
        error('Matrix not positive definite at iteration %d.',k)
    end
    
    % equation 7.56
    % here the method explained in the footnote is used:
    % "Here we augment the sigma points with additional points derived from
    % the matrix square root of the process noise covariance. This requires
    % setting L -> 2L and recalculating the various weights Wi accordingly.
    % Alternatively, we may redraw a complete new set of sigma points, 
    % i.e.,
    % Xk_km1 = [xh_kb xh_kb + gamma*sqrt_Pkb xh_kb - gamma*sqrt_Pkb]. 
    % This alternative approach results in fewer sigma points being used, 
    % but also discards any odd-moments information captured by the 
    % original propagated sigma points"
    %{
    Xk_km1 = [xh_kb ...
              xh_kb + gamma*sqrt_Pkb ...
              xh_kb - gamma*sqrt_Pkb];
    %}
    Xk_km1 = [xh_kb ...
              (repmat(xh_kb,1,L) + gamma*sqrt_Pkb)   ...
              (repmat(xh_kb,1,L) - gamma*sqrt_Pkb)];           
    
    % equation 7.57
    Yk_km1 = H(Xk_km1);
    
    % equation 7.58
    % yh_kb = sum(repmat(Wm,L,1) .* Yk_km1, 2);
    % is equivalent to:
    yh_kb = Yk_km1*Wm';
    
    %% measurement-update
    % equation 7.59
    P_yhyh = Rn;
    for i = 1:2*L+1
        tmp    = Yk_km1(:,i) - yh_kb;
        P_yhyh = P_yhyh + Wc(i)*(tmp*tmp');
    end
    
    % equation 7.60
    P_xy = zeros(L,Ly);
    for i = 1:2*L+1
        tmp1  = Xk_km1(:,i) - xh_kb;
        tmp2  = Yk_km1(:,i) - yh_kb;
        P_xy = P_xy + Wc(i)*(tmp1*tmp2');
    end
    
    % equation 7.61
    K_k = P_xy/P_yhyh;  % = P_xy*inv(P_yhyh); 
    
    % equation 7.62
    xh(:,k) = xh_kb + K_k*(y(:,k) - yh_kb);
    
    % equation 7.63
    P{k} = P_kb - K_k*P_yhyh*K_k';
end
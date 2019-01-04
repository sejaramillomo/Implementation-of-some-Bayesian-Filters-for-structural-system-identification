function [m_k, P_k] = kf(A_km1,B_km1,H_k,m_0,P_0,y,u,Q,R)
    %%  [m_k, P_k] = kf(A_km1,B_km1,H_k,m_0,P_0,y,u,Q,R)
    %
    %   This function uses the 'Kalman Filter' (KF) to compute the 
    %   estimation of the linear dynamical system given by:
    %
    %        	x_k = A_km1*x_km1 + B_km1*x_km1 + q_km1        (1)
    %                   y_k = H_k*x_k     + r_k        (2)
    %
    %   The algorithm is taken from Theorem 4.2 in (Sarkka, S).
    %
    %   Input data:
    %
    %   - A_km1 : Transition matrix of the dynamic model
    %   - B_km1 : Transition matrix of the dynamic input
    %   - H_k   : Measurement model matrix
    %   - m_0   : Initial state of the dynamical system (L x 1 vector)(L states)
    %   - P_0   : Error covariance matrix of the state x_0 (L x L matrix)
    %   - y     : Measurement matrix (Ly x N matrix)
    %             (Ly observations - N time steps)
    %   - u     : Optional control input (N x g vector) 
    %             (N time steps - g control inputs)
    %   - R     : Process-noise discrete covariance (L x L matrix)
    %   - R     : Measurement-noise discrete covariance (Ly x Ly matrix)
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
    %   - It is assumed that "q_k" and "r_k" have mean zero, and that they 
    %     are uncorrelated (E[v_j n_k] = 0).
    %
    %   Bibliography:
    %
    %   - Sarkka, S. (2013). Bayesian filtering and smoothing (Vol. 3). 
    %     Cambridge University Press.
    %
    % -------------------------------------------------------
    % | Developed by:   Sebastian Jaramillo Moreno          |
    % |                 sejaramillomo@unal.edu.co           |
    % |                 National University of Colombia     |
% |                 Manizales, Colombia.                |
% -------------------------------------------------------
%
%   Date: 28 - Aug - 2018
L  = length(m_0);                       % number of states
Ly = size(y,1);                         % number of observations
N  = size(y,2);                         % number of measurements
%% verification of the consistency of the data
if size(P_0,1) ~= L || size(P_0,2) ~= L
    error('The size of the matrix P_0 is not the same as the size of x_0')
end
if size(Q,1) ~= L || size(Q,2) ~= L
    error('The size of the matrix Rq is not the same as the size of x_0')
end

if size(R,1) ~= Ly || size(R,2) ~= Ly
    error('The size of the matrix Rr is not the same as the size of y(:,i)')
end

%% data initialization
% initial state and covariance
m_k(:, 0+1) = m_0;
P_k{0+1}    = P_0;

for k = (1+1):N
    % the prediction step is (equation 4.20)
    m_kb = A_km1*m_k(:, k-1) + B_km1*u(:, k);
    P_kb = A_km1*P_k{k-1}*A_km1' + Q;
    
    % the update step is (equation 4.21)
    v_k = y(:, k) - H_k*m_kb;
    S_k = H_k*P_kb*H_k' + R;
    K_k = P_kb*H_k'/S_k;
    
    m_k(:, k) = m_kb + K_k*v_k;
    P_k{k}    = P_kb - K_k*S_k*K_k';
end
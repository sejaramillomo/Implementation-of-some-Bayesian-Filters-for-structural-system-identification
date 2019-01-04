function [x_k, w_k, m_xk] = p_f(F,H,x_k0,w_k0,z_k,u_k,Rv,Rn)
    %%  [x_k, w_k, m_xk] = p_f(F,H,x_k0,w_k0,z_k,u_k,Rv,Rn)
    %
    %   This function uses the 'Generic Particle Filter' (GPF) to compute 
    %   the estimation of the nonlinear dynamical system given by:
    %
    %               x_k = F(x_{k-1}, u_k, v_{k-1})	(1)
    %               y_k = H(x_k, n_k)               (2)
    %
    %   The algorithm is taken from Algorithm 3 in (Arulampalam et al).
    %
    %   Input data:
    %
    %   - F    : Function handle in (1) (L x 1 vector) 
    %            (L process equations)
    %   - H    : Function handle in (2) (Ly x 1 vector) 
    %            (Lz observation equations)
    %   - x_k0 : Initial particles of the dynamical system (Ns x L matrix)
    %   - w_k0 : Initial weigths of the dynamical system (Ns x L matrix)
    %   - z_k  : Measurement vector (Lz x N matrix)
    %            (Ly observations - N time steps)
    %   - u_k  : Optional control input (N x g vector) 
    %            (N time steps - g control inputs)
    %   - Rv   : Process-noise discrete covariance (L x L matrix)
    %   - Rn   : Measurement-noise discrete covariance (Lz x Lz matrix)
    %   - Nt   : Limit number of effective particles.
    %
    %   Output data:
    %
    %   - x_k : State estimate observational update (NsxLxN array).
    %   - w_k : Weigth estimate observational update (NsxLxN array).
    %
    %   Notes:
    %
    %   - Note that 'zk' is an "Lz x N" matrix. It means that the number of 
    %     columns is the number of states to estimate 'k'; the number of 
    %     rows is the number of measured variables at time 'k'.
    %   - It is assumed that "v_k" and "n_k" have mean zero, and that they 
    %     are uncorrelated (E[v_j n_k] = 0).
    %   - It is assumed that q_xk_given_xkm1_yk = p_xk_given_xkm1, like in
    %     the equation (62)
    %
    %   Bibliography:
    %
    %   - Arulampalam, M. S., Maskell, S., Gordon, N., & Clapp, T. (2002). 
    %     "A tutorial on particle filters for online nonlinear/non-Gaussian 
    %     Bayesian tracking". IEEE Transactions on signal processing, 
    %     50(2), 174-188.
    %
    % -------------------------------------------------------
    % | Developed by:   Sebastian Jaramillo Moreno          |
    % |                 sejaramillomo@unal.edu.co           |
    % |                 National University of Colombia     |
    % |                 Manizales, Colombia.                |
    % -------------------------------------------------------
    %
    %   Date: 18 - Oct - 2018
    
Nm = size(z_k, 1);      % number of observations
N = size(z_k, 2);       % number of steps
L = size(x_k0, 1);      % number of states
Ns = size(x_k0, 2);     % number of particles

%% verification of the consistency of the data
if size(Rv,1) ~= L || size(Rv,2) ~= L
    error('The size of the matrix Rv is not the same as the size of x_0')
end

if size(Rn,1) ~= Nm || size(Rn,2) ~= Nm
    error('The size of the matrix Rn is not the same as the size of y(:,i)')
end

%% data initialization
x_k = zeros(L, Ns, N);
w_k = zeros(1, Ns, N);
m_xk = zeros(L, N);

x_k(:, :, 1) = x_k0;
w_k(:, :, 1) = w_k0;

% likelihood function
p_zk_given_xk = @(k, xk) normpdf(H(xk, 0), z_k(:, k), Rn);

for k = 2:N
    for i = 1:Ns
        % draw the particles
        x_k(:, i, k) = F(x_k(:, i, k-1), u_k(:, k), Rv*randn(L, 1));
        
        % assign the particle a weight, w_k^i, according to (63)
        w_k(:, i, k) = w_k(:, i, k-1)*sum(diag(p_zk_given_xk(k, x_k(:, i, k))));
    end
    
    % calculate total weight
    t = sum(w_k(:, :, k), 2);
    % normalize w_k^i = w_k^i/t
    w_k(:, :, k) = w_k(:, :, k)./t;
    
    % calculate neff_g using (51)
    neff_g = 1/sum(w_k(:, :, k).^2);
    
    % 
    if neff_g < 0.5*Ns
        % resample using resample function
        [x_k(:, :, k), w_k(:, :, k)] = resample(x_k(:, :, k), w_k(:, :, k));
    end
    
    m_xk(:, k) = x_k(:, :, k)*w_k(:, :, k)';
    if isnan(m_xk(:, k))
        error('Error at iteration %d.',k)
    end
end
end

function [xk_j, wk_j] = resample(xk_i, wk_i)
    % This function uses the randsample function

    Ns = size(xk_i, 2);
    idx = randsample(1:Ns, Ns, true, wk_i);
    xk_j = xk_i(:, idx);        % extract new particles
    wk_j = repmat(1/Ns, 1, Ns); % now all particles have the same weight
end
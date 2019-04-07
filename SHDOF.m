config

%% Application of the Kalman Filter, the Unscented Kalman Filter and the
%% Particle Filter to a Single Degree of Freedom hysteretic system state
%% estimation

%  The system assumed in this code is a single degree of freedom hysteretic
%  system, which is described by the equations
%
%                 d2x(t)     dx(t)               d2x_g(t)
%               m*------ + c*----- + k*r(t) = -m*--------
%                  dt2        dt                   dt2
%
%  and
%
%       dr(t)   dx(t)     |dx(t)|                      dx(t)
%       ----- = ----- - B*|-----||r(t)|^(n-1)*r(t) - g*-----*|r|^n
%        dt      dt       | dt  |                       dt
%
%  where:
%   - m     : Mass of the system.
%   - c     : Damping coefficient of the system.
%   - k     : Stiffnes of the system.
%   - B,n,g : Parameters that define the shape of the hysteresis loops
%   - x     : Displacement of the system, its derivatives represesents the
%             velocity and the acceleration.
%   - r     : Hysteretic damage.
%   - x_g   : Its second derivative represents the ground acceleration, in
%             other words, the input of the system.
%
%  The excitation signal is the El Centro earthquake.

%% Response simulation

% The excitation signal
load elcentro_NS.dat
t   = elcentro_NS(:, 1)';   % times
x_g = elcentro_NS(:, 2)';   % ground acceleration
x_g = 5*x_g/max(abs(x_g));  % the peak acceleration is scaled to 5g
x_g = 9.82*x_g;             % (m/s^2)

dt = t(2) - t(1);           % time between measurements
N  = length(t);             % number of measurements

% The parameters are:
m = 1;                      % kN*s^2/m
c = 0.3;                    % kN*s/m
k = 9;                      % kN/m
beta  = 2;
gamma = 1;
n     = 2;

% Initial state

x_0 = [0; 0; 0];            % the system starts from rest
nx  = length(x_0);          % number of states

x = zeros(nx, N+1);         % here the evolution of the system is going to 
x(:,1) = x_0;               % be saved

% The system is written in a state-space form
% X = [x dx/dt r]'
F = @(x, u) [x(2)
             -(c*x(2) + k*x(3))/m
             x(2) - beta*abs(x(2))*abs(x(3))^(n-1)*x(3) - gamma*x(2)*abs(x(3))^n] ...
             + [0; -u; 0];

% To simulate the system, the Runge-Kutta fourth order method is used.
for i = 1:N
    % [x_kmh] = rk_discrete(diff_eq,x_0,u,h)
    x(:,i+1) = rk_discrete(F, x(:,i), x_g(i), dt);
end

% Then, the total acceleration of the system is given as
acc = -(c*x(2, 2:end) + k*x(3, 2:end))/m;

%% Measurements generation

% The filters will take the acceleration measurement as the observations
meas = zeros(1, N);
% the RMS noise-to-signal is used to add the noise
RMS = sqrt(sum(acc.^2)/N);

noise_per = 0.05;           % 5% of the RMS is asumed as noise variance

% the measures are generated
meas = acc + noise_per*RMS*randn(1, N);

%% Unscented Kalman Filter implementation

% The 'Unscented Kalman Filter' (UKF) is useful to the estimation
% of the nonlinear dynamical system given by:
%
%               x_{k+1} = F(x_k, u_k) + v_k        (1)
%               y_k     = H(x_k)      + n_k        (2)

% We redifine the functions
F = @(x, u) [x(2)
             -(c*x(2) + k*x(3))/m
             x(2) - beta*abs(x(2))*abs(x(3))^(n-1)*x(3) - gamma*x(2)*abs(x(3))^n] ...
             + [0; -u; 0];
H = @(x)    HH(x, m, c, k);

% the initial covariance matrix, process covariance matrix and the 
% measurement covariance matrix
x_0 = [0; 0; 0];
P_0 = 0.0001*eye(nx);
Q = 0.01*eye(nx);
R = 0.001;

% and a "soft" discretization of the covariance matrices
Q_nl = Q*dt;
R_nl = R/dt;

% using the Unscented Kalman Filter
%[xh, P] = ukf(F,H,x_0,P_0,y,u,Rv,Rn,dt)
tic
[x_ukf, P_ukf] = ukf(F, H, x_0, P_0, meas, x_g, Q_nl, R_nl, dt);
toc

% to compute the standard deviation at each time step
sd_ukf = zeros(size(x_ukf));
for i = 1:N
    sd_ukf(:, i) = sqrt(diag(P_ukf{i}));
end

%% Particle Filter implementation

% The 'Particle Filter' (PF) is a Monte Carlo aproach to Bayesian
% filtering, is useful to the estimation of the nonlinear dynamical 
% system given by:
%
%               x_k = F(x_{k-1}, u_k, v_{k-1})	(1)
%               y_k = H(x_k, n_k)               (2)

% where the functions are redefined as
F_pf = @(x, u, v) rk_discrete(F, x, u, dt) + v;
H_pf = @(x, n)    H(x) + n;

% The number of particles used
Ns = 1000;

% the initial particles and their respective weigths are defined as
x_k0 = mvnrnd(x_0, P_0, Ns)';       % initial particles
w_k0 = 1/Ns*ones(1, Ns);            % initial weigths

% using the Particle Filter
% [x_k, w_k, m_xk] = p_f(F,H,x_k0,w_k0,z_k,u_k,Rv,Rn)
tic
[xx_pf, w_pf, x_pf] = p_f(F_pf, H_pf, x_k0, w_k0, meas, x_g, Q_nl, R_nl);
toc

% to compute the percentile 15.87 and 84.13 (that in a gaussian distribution
% are plus one and minus one standard deviation) at each time step
per_pf = zeros(6, N);
for i = 1:N
    per = prctile(xx_pf(:, :, i), [15.87 84.13], 2);
    per_pf(:, i) = per(:);
end

%% Relative Errors

% Unscented Kalman Filter error
err_d_ukf = (x_ukf(1, :) - x(1, 2:end))./x(1, 2:end);
var_d_ukf = var(err_d_ukf); m_d_ukf = mean(err_d_ukf);
ms_d_ukf = var_d_ukf + m_d_ukf^2;
err_v_ukf = (x_ukf(2, :) - x(2, 2:end))./x(2, 2:end);
var_v_ukf = var(err_v_ukf); m_v_ukf = mean(err_v_ukf);
ms_v_ukf = var_v_ukf + m_v_ukf^2;
ms_v_ukf = var_v_ukf + m_v_ukf^2;
m_d_ukf, m_v_ukf, var_d_ukf, var_v_ukf, ms_d_ukf, ms_v_ukf
% Particle Filter error
err_d_pf = (x_pf(1, :) - x(1, 2:end))./x(1, 2:end);
var_d_pf = var(err_d_pf); m_d_pf = mean(err_d_pf);
ms_d_pf = var_d_pf + m_d_pf^2;
err_v_pf = (x_pf(2, :) - x(2, 2:end))./x(2, 2:end);
var_v_pf = var(err_v_pf); m_v_pf = mean(err_v_pf);
ms_v_pf = var_v_pf + m_v_pf^2;
m_d_pf, m_v_pf, var_d_pf, var_v_pf, ms_d_pf, ms_v_pf

%% Plots

% Acceleration measures
figure
hold on
ylabel('Acceleration [$m/s^{2}$]')
xlabel('Time [$s$]')
a1 = plot(t, acc,      '-r');
a2 = plot(t, meas, '.k', 'MarkerSize', 5);
legend([a1, a2], ...
       'True signal', 'Measurements', ...
       'Location', 'southeast')
axis tight

print('/home/sebastian/Tesis/Latex/figures/MATLAB/measures_h', '-depsc')

% Unscented Kalman filter results

figure
% Displacement
subplot(311)
hold on
d_ukf_sd = fill([t t(end:-1:1)], ...
               [x_ukf(1, :)+sd_ukf(1, :) x_ukf(1, end:-1:1)-sd_ukf(1, end:-1:1)], ...
               [0.8 0.8 1], 'EdgeColor', 'none');
d_ukf = plot(t, x_ukf(1, :),  '-b');
d = plot(t, x(1, 2:end), '--r');
legend([d, d_ukf, d_ukf_sd], 'True signal', 'UKF', '1 Standard deviation', ...
       'Location', 'southeast', ...
       'Orientation', 'horizontal')
ylabel('Displacement [$m$]')
xlabel('Time [$s$]')
ymax = ceil(max(abs([x_ukf(1, :)+sd_ukf(1, :) x_ukf(1, end:-1:1)-sd_ukf(1, end:-1:1)])));
axis([min(t) max(t) -ymax ymax])

% Velocity
subplot(312)
hold on
v_ukf_sd = fill([t t(end:-1:1)], ...
                [x_ukf(2, :)+sd_ukf(2, :) x_ukf(2, end:-1:1)-sd_ukf(2, end:-1:1)], ...
                [0.8 0.8 1], 'EdgeColor', 'none');
v_ukf = plot(t, x_ukf(2, :),  '-b');
v = plot(t, x(2, 2:end), '--r');
ylabel('Velocity [$m/s$]')
xlabel('Time [$s$]')
ymax = ceil(max(abs([x_ukf(2, :)+sd_ukf(2, :) x_ukf(2, end:-1:1)-sd_ukf(2, end:-1:1)])));
axis([min(t) max(t) -ymax ymax])

% Hysteresis
subplot(313)
hold on
h_ukf_sd = fill([t t(end:-1:1)], ...
                [x_ukf(3, :)+sd_ukf(3, :) x_ukf(3, end:-1:1)-sd_ukf(3, end:-1:1)], ...
                [0.8 0.8 1], 'EdgeColor', 'none');
h_ukf = plot(t, x_ukf(3, :),  '-b');
h = plot(t, x(3, 2:end), '--r');
ylabel('r')
xlabel('Time [$s$]')
ymax = ceil(max(abs([x_ukf(3, :)+sd_ukf(3, :) x_ukf(3, end:-1:1)-sd_ukf(3, end:-1:1)])));
axis([min(t) max(t) -ymax ymax])

print('/home/sebastian/Tesis/Latex/figures/MATLAB/ukf_shdof', '-depsc')

% Hysteresis loops
figure
hold on
h_ukf_l = plot(x_ukf(1, :), x_ukf(3, :), '-b');
h_l = plot(x(1, 2:end), x(3, 2:end), '--r');
legend([h_l, h_ukf_l], 'True signal', 'UKF', ...
       'Location', 'southeast', ...
       'Orientation', 'horizontal')
ylabel('r')
xlabel('Displacement [$m$]')
ymax = ceil(max(abs([x_ukf(3, :) x(3, 2:end)]))*10)/10;
xmax = ceil(max(abs([x_ukf(1, :) x(1, 2:end)]))*10)/10;
axis([-xmax xmax -ymax ymax])

print('/home/sebastian/Tesis/Latex/figures/MATLAB/ukf_shdof_hl', '-depsc')

% Particle filter results

figure
% Displacement
subplot(311)
hold on
d_pf_sd = fill([t t(end:-1:1)], ...
               [per_pf(1, :) per_pf(4, end:-1:1)], ...
               [0.8 0.8 1], 'EdgeColor', 'none');
d_pf = plot(t, x_pf(1, :),  '-b');
d = plot(t, x(1, 2:end), '--r');
leg_pf = sprintf('PF N = %d', Ns);
legend([d, d_pf, d_pf_sd], 'True signal', leg_pf, '$P_{15.87}$ and $P_{84.13}$', ...
       'Location', 'southeast', ...
       'Orientation', 'horizontal')
ylabel('Displacement [$m$]')
xlabel('Time [$s$]')
ymax = ceil(max(abs([per_pf(1, :) per_pf(4, end:-1:1)]))*10)/10;
axis([min(t) max(t) -ymax ymax])

% Velocity
subplot(312)
hold on
v_pf_sd = fill([t t(end:-1:1)], ...
               [per_pf(2, :) per_pf(5, end:-1:1)], ...
               [0.8 0.8 1], 'EdgeColor', 'none');
v_pf = plot(t, x_pf(2, :),  '-b');
v = plot(t, x(2, 2:end), '--r');
ylabel('Velocity [$m/s$]')
xlabel('Time [$s$]')
ymax = ceil(max(abs([per_pf(2, :) per_pf(5, end:-1:1)]))*10)/10;
axis([min(t) max(t) -ymax ymax])

% Hysteresis
subplot(313)
hold on
h_pf_sd = fill([t t(end:-1:1)], ...
               [per_pf(3, :) per_pf(6, end:-1:1)], ...
               [0.8 0.8 1], 'EdgeColor', 'none');
h_pf = plot(t, x_pf(3, :),  '-b');
h = plot(t, x(3, 2:end), '--r');
ylabel('r')
xlabel('Time [$s$]')
ymax = ceil(max(abs([per_pf(3, :) per_pf(6, end:-1:1)]))*10)/10;
axis([min(t) max(t) -ymax ymax])

print('/home/sebastian/Tesis/Latex/figures/MATLAB/pf_shdof', '-depsc')

% Hysteresis loops
figure
hold on
h_pf_l = plot(x_pf(1, :), x_pf(3, :), '-b');
h_l = plot(x(1, 2:end), x(3, 2:end), '--r');
leg_h_pf = sprintf('PF N=%i', Ns);
legend([h_l, h_pf_l], 'True signal', leg_h_pf, ...
       'Location', 'southeast', ...
       'Orientation', 'horizontal')
ylabel('r')
xlabel('Displacement [$m$]')
ymax = ceil(max(abs([x_pf(3, :) x(3, 2:end)]))*10)/10;
xmax = ceil(max(abs([x_pf(1, :) x(1, 2:end)]))*10)/10;
axis([-xmax xmax -ymax ymax])

print('/home/sebastian/Tesis/Latex/figures/MATLAB/pf_shdof_hl', '-depsc')

figure
% Displacement Error
subplot(211)
hold on
err = [err_d_ukf; err_d_pf];
b_w = (max(max(err, [], 2)) - min(min(err, [], 2)))/(3*ceil(sqrt(N)));
e_d_ukf = histogram(err_d_ukf, ...
                    'BinWidth', b_w, ...
                    'Normalization', 'probability');
e_d_pf  = histogram(err_d_pf, ...
                    'BinWidth', b_w, ...
                    'Normalization', 'probability');
l_pf = sprintf('PF N = %d', Ns);
legend([e_d_ukf, e_d_pf], 'UKF', l_pf, ...
       'Location', 'northeast', ...
       'Orientation', 'horizontal')
ylabel('Displacement error')
axis tight

% Velocity Error
subplot(212)
hold on
err = [err_v_ukf; err_v_pf];
b_w = (max(max(err, [], 2)) - min(min(err, [], 2)))/(3*ceil(sqrt(N)));
e_v_ukf = histogram(err_v_ukf, ...
                    'BinWidth', b_w, ...
                    'Normalization', 'probability');
e_v_pf  = histogram(err_v_pf, ...
                    'BinWidth', b_w, ...
                    'Normalization', 'probability');
ylabel('Velocity error')
axis tight

print('/home/sebastian/Tesis/Latex/figures/MATLAB/error_shdof', '-depsc')

function y = HH(xx,m,c,k)

    H = @(x)    -(x(2)*c + x(3)*k)/m;

    for i = 1:size(xx,2)
        y(i) = H(xx(:,i));
    end
end
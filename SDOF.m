config

%% Application of the Kalman Filter, the Unscented Kalman Filter and the
%% Particle Filter to a Single Degree of Freedom linear system state 
%% estimation

%  The system assumed in this code is a single degree of freedom linear
%  system, which is described by the equation
%
%                 d2x(t)     dx(t)               d2x_g(t)
%               m*------ + c*----- + k*x(t) = -m*--------
%                  dt2        dt                   dt2
%
%  where:
%   - m   : Mass of the system.
%   - c   : Damping coefficient of the system.
%   - k   : Stiffnes of the system.
%   - x   : Displacement of the system, its derivatives represesents the
%           velocity and the acceleration.
%   - x_g : Its second derivative represents the ground acceleration, in
%           other words, the input of the system.
%
%  The excitation signal is the El Centro earthquake.

%% Response simulation

% The excitation signal
load elcentro_NS.dat
t   = elcentro_NS(:, 1)';   % times
x_g = elcentro_NS(:, 2)';   % ground acceleration
x_g = 9.82*x_g;             % (m/s^2)

dt = t(2) - t(1);           % time between measurements
N  = length(t);             % number of measurements

% The parameters are:
m = 1;                      % kN*s^2/m
c = 0.3;                    % kN*s/m
k = 9;                      % kN/m

% Initial state

x_0 = [0; 0];               % the system starts from rest
nx  = length(x_0);          % number of states

x = zeros(nx, N+1);         % here the evolution of the system is going to 
x(:,1) = x_0;               % be saved

% The system is written in a state-space form
% X = [x dx/dt]'
A = [0      1
     -k/m   -c/m];
B = [0; 1];

F = @(x, u) A*x + B*u;

% To simulate the system, the Runge-Kutta fourth order method is used
for i = 1:N
    % [x_kmh] = rk_discrete(diff_eq,x_0,u,h)
    x(:,i+1) = rk_discrete(F, x(:,i), x_g(i), dt);
end

% Then, the total acceleration of the system is given as
acc = -(c*x(2, 2:end) + k*x(1, 2:end))/m;

%% Measurements generation

% The filters will take the acceleration measurement as the observations
meas = zeros(1, N);
% the RMS noise-to-signal is used to add the noise
RMS = sqrt(sum(acc.^2)/N);

noise_per = 0.05;           % 5% of the RMS is asumed as noise variance

% The measures are generated
meas = acc + noise_per*RMS*randn(1, N);

%% Kalman filter implementation

% The Kalman Filter only accepts linear sate-space equations in the form
%
%       dX/dt = A*X + B*u + q
%           y = H*X + r

% The matrices of the system are written again
% for the process
A  = [0      1
      -k/m   -c/m];
B  = [0; 1];
% for the measurement
Hk = [-k/m   -c/m];

% System initial mean and covariance
x_0 = [0; 0];
P_0 = 0.0001*eye(nx);

% Process noise covariance
Q = 0.01*eye(nx);

% Measurement noise covariance
R = 0.001;

% Is necessary to discretize the system, so
Ad = eye(size(A)) + A*dt;
Bd = B*dt + A*B*dt^2/2;

Qd = Q*dt + (Q*A' + A*Q)*dt^2/2 + A*Q*A'*dt^3/3;
Rd = R/dt;

% using the Kalman filter function
%[m_k, P_k] = kf(A_km1,B_km1,H_k,m_0,P_0,y,u,Q,R)
tic
[x_kf, P_kf] = kf(Ad, Bd, Hk, x_0, P_0, meas, x_g, Qd, Rd);
toc

% to compute the standard deviation at each time step
sd_kf = zeros(size(x_kf));
for i = 1:N
    sd_kf(:, i) = sqrt(diag(P_kf{i}));
end

%% Unscented Kalman Filter implementation

% The 'Unscented Kalman Filter' (UKF) is useful to the estimation
% of the nonlinear dynamical system given by:
%
%               x_{k+1} = F(x_k, u_k) + v_k        (1)
%               y_k     = H(x_k)      + n_k        (2)

% We redifine the functions
F = @(x, u) A*x + B*u;
H = @(x) Hk*x;

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
per_pf = zeros(4, N);
for i = 1:N
    per = prctile(xx_pf(:, :, i), [15.87 84.13], 2);
    per_pf(:, i) = per(:);
end

%% Relative Errors

% Kalman Filter error
err_d_kf = (x_kf(1, :) - x(1, 2:end))./x(1, 2:end);
var_d_kf = var(err_d_kf); m_d_kf = mean(err_d_kf);
ms_d_kf = var_d_kf + m_d_kf^2;
err_v_kf = (x_kf(2, :) - x(2, 2:end))./x(2, 2:end);
var_v_kf = var(err_v_kf); m_v_kf = mean(err_v_kf);
ms_v_kf = var_v_kf + m_v_kf^2;
m_d_kf, m_v_kf, var_d_kf, var_v_kf, ms_d_kf, ms_v_kf
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
zoom = [32.5 -0.2 5 0.4];
ax = [0.7 0.7 0.2 0.2];
t_zoom = (t >= zoom(1) & t <= zoom(1)+zoom(3));
ylabel('Acceleration [$m/s^{2}$]')
xlabel('Time [$s$]')
a1 = plot(t, acc,  '-r');
a2 = plot(t, meas, '.k', 'MarkerSize', 5);
rectangle('Position', zoom)
axis tight
legend([a1, a2], ...
       'True signal', 'Measurements', ...
       'Location', 'southeast')
ax_zoom = axes('Position', ax);
hold on, box on
a1_zoom = plot(t(t_zoom), acc(t_zoom),  '-r');
a2 = plot(t(t_zoom), meas(t_zoom), '.k', 'MarkerSize', 5);
axis([zoom(1) zoom(1)+zoom(3) zoom(2) zoom(2)+zoom(4)])

print('/home/sebastian/Tesis/Latex/figures/MATLAB/measures', '-depsc')

% Kalman filter results

figure
% Displacement
subplot(211)
hold on
d_kf_sd = fill([t t(end:-1:1)], ...
               [x_kf(1, :)+sd_kf(1, :) x_kf(1, end:-1:1)-sd_kf(1, end:-1:1)], ...
               [0.8 0.8 1], 'EdgeColor', 'none');
d_kf    = plot(t, x_kf(1, :),  '-b');
d       = plot(t, x(1, 2:end), '--r');
legend([d, d_kf, d_kf_sd], 'True signal', 'KF', '1 Standard deviation', ...
       'Location', 'southeast', ...
       'Orientation', 'horizontal')
ylabel('Displacement [$m$]')
xlabel('Time [$s$]')
ymax = ceil(max(abs([x_kf(1, :)+sd_kf(1, :) x_kf(1, end:-1:1)-sd_kf(1, end:-1:1)]))*10)/10;
axis([min(t) max(t) -ymax ymax])

% Velocity
subplot(212)
hold on
v_kf_sd = fill([t t(end:-1:1)], ...
               [x_kf(2, :)+sd_kf(2, :) x_kf(2, end:-1:1)-sd_kf(2, end:-1:1)], ...
               [0.8 0.8 1], 'EdgeColor', 'none');
v_kf    = plot(t, x_kf(2, :),  '-b');
v       = plot(t, x(2, 2:end), '--r');
ylabel('Velocity [$m/s$]')
xlabel('Time [$s$]')
ymax = ceil(max(abs([x_kf(2, :)+sd_kf(2, :) x_kf(2, end:-1:1)-sd_kf(2, end:-1:1)]))*10)/10;
axis([min(t) max(t) -ymax ymax])

print('/home/sebastian/Tesis/Latex/figures/MATLAB/kf_sdof', '-depsc')

% Unscented Kalman filter results

figure
% Displacement
subplot(211)
hold on
d_ukf_sd = fill([t t(end:-1:1)], ...
               [x_ukf(1, :)+sd_ukf(1, :) x_ukf(1, end:-1:1)-sd_ukf(1, end:-1:1)], ...
               [0.8 0.8 1], 'EdgeColor', 'none');
d_ukf    = plot(t, x_ukf(1, :),  '-b');
d        = plot(t, x(1, 2:end), '--r');
legend([d, d_ukf, d_ukf_sd], 'True signal', 'UKF', '1 Standard deviation', ...
       'Location', 'southeast', ...
       'Orientation', 'horizontal')
ylabel('Displacement [$m$]')
xlabel('Time [$s$]')
ymax = ceil(max(abs([x_ukf(1, :)+sd_ukf(1, :) x_ukf(1, end:-1:1)-sd_ukf(1, end:-1:1)]))*10)/10;
axis([min(t) max(t) -ymax ymax])

% Velocity
subplot(212)
hold on
v_ukf_sd = fill([t t(end:-1:1)], ...
                [x_ukf(2, :)+sd_ukf(2, :) x_ukf(2, end:-1:1)-sd_ukf(2, end:-1:1)], ...
                [0.8 0.8 1], 'EdgeColor', 'none');
v_ukf    = plot(t, x_ukf(2, :),  '-b');
v        = plot(t, x(2, 2:end), '--r');
ylabel('Velocity [$m/s$]')
xlabel('Time [$s$]')
ymax = ceil(max(abs([x_ukf(2, :)+sd_ukf(2, :) x_ukf(2, end:-1:1)-sd_ukf(2, end:-1:1)]))*10)/10;
axis([min(t) max(t) -ymax ymax])

print('/home/sebastian/Tesis/Latex/figures/MATLAB/ukf_sdof', '-depsc')

% Particle filter results

figure
% Displacement
subplot(211)
hold on
d_pf_sd = fill([t t(end:-1:1)], ...
               [per_pf(1, :) per_pf(3, end:-1:1)], ...
               [0.8 0.8 1], 'EdgeColor', 'none');
d_pf    = plot(t, x_pf(1, :),  '-b');
d       = plot(t, x(1, 2:end), '--r');
leg_pf = sprintf('PF N = %d', Ns);
legend([d, d_pf, d_pf_sd], 'True signal', leg_pf, '$P_{15.87}$ and $P_{84.13}$', ...
       'Location', 'southeast', ...
       'Orientation', 'horizontal')
ylabel('Displacement [$m$]')
xlabel('Time [$s$]')
ymax = ceil(max(abs([per_pf(1, :) per_pf(3, end:-1:1)]))*10)/10;
axis([min(t) max(t) -ymax ymax])

% Velocity
subplot(212)
hold on
v_pf_sd = fill([t t(end:-1:1)], ...
               [per_pf(2, :) per_pf(4, end:-1:1)], ...
               [0.8 0.8 1], 'EdgeColor', 'none');
v_pf    = plot(t, x_pf(2, :),  '-b');
v       = plot(t, x(2, 2:end), '--r');
leg_pf = sprintf('PF N = %d', Ns);
ylabel('Velocity [$m/s$]')
xlabel('Time [$s$]')
ymax = ceil(max(abs([per_pf(2, :) per_pf(4, end:-1:1)]))*10)/10;
axis([min(t) max(t) -ymax ymax])

print('/home/sebastian/Tesis/Latex/figures/MATLAB/pf_sdof', '-depsc')

% Error

figure
% Displacement Error
subplot(211)
hold on
err = [err_d_kf; err_d_ukf; err_d_pf];
b_w = (max(max(err, [], 2)) - min(min(err, [], 2)))/(3*ceil(sqrt(N)));
e_d_kf  = histogram(err_d_kf, ...
                    'BinWidth', b_w, ...
                    'Normalization', 'probability');
e_d_ukf = histogram(err_d_ukf, ...
                    'BinWidth', b_w, ...
                    'Normalization', 'probability');
e_d_pf  = histogram(err_d_pf, ...
                    'BinWidth', b_w, ...
                    'Normalization', 'probability');
l_pf = sprintf('PF N = %d', Ns);
legend([e_d_kf, e_d_ukf, e_d_pf], 'KF', 'UKF', l_pf, ...
       'Location', 'northeast', ...
       'Orientation', 'horizontal')
ylabel('Displacement error')
axis tight

% Velocity Error
subplot(212)
hold on
err = [err_v_kf; err_v_ukf; err_v_pf];
b_w = (max(max(err, [], 2)) - min(min(err, [], 2)))/(3*ceil(sqrt(N)));
e_v_kf  = histogram(err_v_kf, ...
                    'BinWidth', b_w, ...
                    'Normalization', 'probability');
e_v_ukf = histogram(err_v_ukf, ...
                    'BinWidth', b_w, ...
                    'Normalization', 'probability');
e_v_pf  = histogram(err_v_pf, ...
                    'BinWidth', b_w, ...
                    'Normalization', 'probability');
ylabel('Velocity error')
axis tight

print('/home/sebastian/Tesis/Latex/figures/MATLAB/error_sdof', '-depsc')
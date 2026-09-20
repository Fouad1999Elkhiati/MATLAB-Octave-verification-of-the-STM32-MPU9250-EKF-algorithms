% mekf_verification.m
%
% Verifies the 6-state multiplicative Extended Kalman Filter (MEKF)
% implemented in ekf.c / ekf.h by porting the identical math to
% MATLAB/Octave and testing it against a synthetic ground-truth
% trajectory with known gyro bias, gravity and magnetic reference.
%
% What "verification" means here:
%   1. Generate a true attitude trajectory q_true(t) from a known
%      angular-rate profile, plus a known constant gyro bias.
%   2. Synthesize noisy gyro / accelerometer / magnetometer samples
%      from that ground truth (exactly the quantities imu_port.c
%      would hand to ekf_predict / ekf_update_accel / ekf_update_mag).
%   3. Run the MEKF (ported 1:1 from ekf.c) starting from a WRONG
%      initial attitude and zero bias.
%   4. Check that the attitude error converges to a small steady-state
%      value and that the bias estimate converges to the true bias.
%
% Compatible with both MATLAB (>= R2016b, for local functions in a
% script file) and GNU Octave (>= 4.4).
%
% Run:  octave-cli mekf_verification.m        (headless)
%       or open in MATLAB and press Run.
%
% Written as a function file (function name == file name) so that
% Octave, which -- unlike MATLAB -- requires local functions in a
% script to be defined before their first call site, executes
% correctly either way: MATLAB runs a function file with no
% arguments when you press Run, and Octave does the same when you
% invoke it from the command line.

function mekf_verification()

clear; clc; close all;

% -------------------------------------------------------------------
% 1. Simulation setup
% -------------------------------------------------------------------
rng(42);                       % reproducible noise

dt_gyro   = 0.005;             % 200 Hz gyro (s)
update_every = 10;              % accel+mag correction every 10 gyro steps -> 20 Hz
T         = 40;                % total simulated time (s)
N         = round(T/dt_gyro);

% True (constant) gyro bias, rad/s
b_true = [0.010; -0.020; 0.015];

% Noise parameters (must match ekf_cfg_t fields)
cfg.gyro_noise  = 0.003;       % rad/s/sqrt(Hz)
cfg.bias_noise  = 0.0005;      % rad/s^2/sqrt(Hz)
cfg.accel_noise = 0.02;        % normalised units (variance-ish scale)
cfg.mag_noise   = 0.02;
mag_ref = [1; 0; 0.6];
mag_ref = mag_ref / norm(mag_ref);
cfg.mag_ref = mag_ref;

g_nav = [0;0;1];

% -------------------------------------------------------------------
% 2. Generate ground-truth trajectory
% -------------------------------------------------------------------
t = (0:N-1)'*dt_gyro;

% A smooth, persistently-exciting true angular rate profile (rad/s)
w_true = [0.35*sin(0.20*t), 0.25*cos(0.15*t+0.3), 0.15*sin(0.10*t+1.0)];

q_true = zeros(N,4);
q_true(1,:) = [1 0 0 0];
for k = 1:N-1
    q_true(k+1,:) = quat_integrate(q_true(k,:)', w_true(k,:)', dt_gyro)';
end

% -------------------------------------------------------------------
% 3. Synthesize sensor samples from ground truth
% -------------------------------------------------------------------
gyro_meas = w_true + b_true' + cfg.gyro_noise/sqrt(dt_gyro)*randn(N,3)*0; % see below
% Proper discrete noise: sigma_sample = gyro_noise/sqrt(dt)  (rad/s)
gyro_meas = w_true + b_true' + (cfg.gyro_noise/sqrt(dt_gyro))*randn(N,3);

acc_meas = zeros(N,3);
mag_meas = zeros(N,3);
lin_accel_spike = zeros(N,1);
for k = 1:N
    R = quat_to_R(q_true(k,:)');
    a_true_body = R' * g_nav;                    % gravity in body frame, 1g
    m_true_body = R' * mag_ref;                   % field in body frame, unit
    acc_meas(k,:) = (a_true_body + cfg.accel_noise*randn(3,1))';
    mag_meas(k,:) = (m_true_body + cfg.mag_noise*randn(3,1))';

    % Inject occasional linear-acceleration events (5% of samples) to
    % exercise the |a|~1g gating logic in ekf_update_accel.
    if rand() < 0.05
        spike = 0.6*randn(3,1);
        acc_meas(k,:) = acc_meas(k,:) + spike';
        lin_accel_spike(k) = 1;
    end
end

% -------------------------------------------------------------------
% 4. Run the MEKF, deliberately mis-initialised
% -------------------------------------------------------------------
% 30 degree initial attitude error about a random axis, zero bias guess
ax = [1;0.3;-0.2]; ax = ax/norm(ax);
q_err0 = [cos(deg2rad(30)/2); sin(deg2rad(30)/2)*ax];
q0 = quat_mul(q_err0, q_true(1,:)');

f.q    = q0;
f.bias = [0;0;0];
f.P    = blkdiag(0.5*eye(3), 0.05*eye(3));   % deliberately larger than truth
f.cfg  = cfg;

q_est   = zeros(N,4);   q_est(1,:)   = f.q';
bias_est= zeros(N,3);   bias_est(1,:)= f.bias';
n_accel_accept = 0; n_accel_reject = 0;

for k = 1:N-1
    f = mekf_predict(f, gyro_meas(k,:)', dt_gyro);

    if mod(k, update_every) == 0
        [f, accepted] = mekf_update(f, acc_meas(k,:)', g_nav, cfg.accel_noise);
        if accepted, n_accel_accept = n_accel_accept + 1;
        else,        n_accel_reject = n_accel_reject + 1; end

        [f, ~] = mekf_update(f, mag_meas(k,:)', mag_ref, cfg.mag_noise);
    end

    q_est(k+1,:)    = f.q';
    bias_est(k+1,:) = f.bias';
end

% -------------------------------------------------------------------
% 5. Evaluate accuracy
% -------------------------------------------------------------------
att_err_deg = zeros(N,1);
for k = 1:N
    att_err_deg(k) = quat_angle_error_deg(q_est(k,:)', q_true(k,:)');
end

steady = t > 10;   % ignore the initial 10 s convergence transient
rms_att_err_steady   = sqrt(mean(att_err_deg(steady).^2));
max_att_err_steady   = max(att_err_deg(steady));
bias_err_final        = bias_est(end,:)' - b_true;
bias_err_norm_final   = norm(bias_err_final);

fprintf('\n================ MEKF verification results ================\n');
fprintf('Samples                          : %d  (T = %.1f s, dt = %.4f s)\n', N, T, dt_gyro);
fprintf('Accel updates accepted / rejected: %d / %d  (gate |a-1g|)\n', n_accel_accept, n_accel_reject);
fprintf('Initial attitude error           : %.2f deg\n', att_err_deg(1));
fprintf('Steady-state (t>10s) RMS att err : %.4f deg\n', rms_att_err_steady);
fprintf('Steady-state (t>10s) MAX att err : %.4f deg\n', max_att_err_steady);
fprintf('True gyro bias   (rad/s)         : [% .5f % .5f % .5f]\n', b_true);
fprintf('Estimated bias   (rad/s), final  : [% .5f % .5f % .5f]\n', bias_est(end,:));
fprintf('Bias estimation error norm       : %.6f rad/s\n', bias_err_norm_final);
fprintf('==============================================================\n');

pass = (rms_att_err_steady < 1.0) && (bias_err_norm_final < 0.003);
if pass
    fprintf('PASS: filter converges to sub-degree attitude accuracy and\n');
    fprintf('      recovers the true gyro bias to within 0.003 rad/s.\n\n');
else
    fprintf('FAIL: filter did not converge to the expected tolerance.\n\n');
end

% -------------------------------------------------------------------
% 6. Plots (saved to PNG; headless-safe)
% -------------------------------------------------------------------
[roll_t,pitch_t,yaw_t]  = euler_from_q_series(q_true);
[roll_e,pitch_e,yaw_e]  = euler_from_q_series(q_est);

fig = figure('visible','off','Position',[100 100 900 900]);

subplot(4,1,1);
plot(t, rad2deg(roll_t), 'k', t, rad2deg(roll_e), 'r--'); ylabel('roll (deg)');
legend('true','MEKF','Location','best'); title('MEKF vs. ground truth');

subplot(4,1,2);
plot(t, rad2deg(pitch_t), 'k', t, rad2deg(pitch_e), 'r--'); ylabel('pitch (deg)');

subplot(4,1,3);
plot(t, rad2deg(yaw_t), 'k', t, rad2deg(yaw_e), 'r--'); ylabel('yaw (deg)');

subplot(4,1,4);
plot(t, att_err_deg, 'b'); ylabel('att. error (deg)'); xlabel('time (s)');
grid on;

print(fig, 'mekf_verification_euler.png', '-dpng', '-r120');

fig2 = figure('visible','off','Position',[100 100 900 500]);
plot(t, bias_est(:,1), 'r', t, bias_est(:,2), 'g', t, bias_est(:,3), 'b'); hold on;
yline_vals = [b_true(1) b_true(2) b_true(3)];
plot(t, repmat(b_true(1),size(t)), 'r:', t, repmat(b_true(2),size(t)), 'g:', ...
     t, repmat(b_true(3),size(t)), 'b:');
xlabel('time (s)'); ylabel('gyro bias (rad/s)');
title('Estimated gyro bias vs. true bias (dotted)');
legend('b_x est','b_y est','b_z est','Location','best');
print(fig2, 'mekf_verification_bias.png', '-dpng', '-r120');

fprintf('Saved: mekf_verification_euler.png, mekf_verification_bias.png\n');

end % function mekf_verification

% =====================================================================
% Local functions -- direct ports of ekf.c
% =====================================================================

function r = quat_mul(a, b)
    % Hamilton product, w-first. Direct port of quat_mul() in ekf.c.
    r = zeros(4,1);
    r(1) = a(1)*b(1) - a(2)*b(2) - a(3)*b(3) - a(4)*b(4);
    r(2) = a(1)*b(2) + a(2)*b(1) + a(3)*b(4) - a(4)*b(3);
    r(3) = a(1)*b(3) - a(2)*b(4) + a(3)*b(1) + a(4)*b(2);
    r(4) = a(1)*b(4) + a(2)*b(3) - a(3)*b(2) + a(4)*b(1);
end

function q = quat_normalize(q)
    n = norm(q);
    if n > 1e-9
        q = q / n;
    else
        q = [1;0;0;0];
    end
end

function R = quat_to_R(q)
    % Body -> nav rotation matrix. Direct port of quat_to_R() in ekf.c.
    w=q(1); x=q(2); y=q(3); z=q(4);
    R = [1-2*(y^2+z^2),   2*(x*y-w*z),     2*(x*z+w*y);
         2*(x*y+w*z),     1-2*(x^2+z^2),   2*(y*z-w*x);
         2*(x*z-w*y),     2*(y*z+w*x),     1-2*(x^2+y^2)];
end

function S = skew(v)
    S = [0 -v(3) v(2); v(3) 0 -v(1); -v(2) v(1) 0];
end

function qn = quat_integrate(q, w, dt)
    % First-order quaternion integration, identical to the
    % dq / quat_mul / quat_normalize sequence in ekf_predict().
    dq = [1; 0.5*w(1)*dt; 0.5*w(2)*dt; 0.5*w(3)*dt];
    qn = quat_normalize(quat_mul(q, dq));
end

function f = mekf_predict(f, gyro, dt)
    % Direct port of ekf_predict() in ekf.c.
    w = gyro - f.bias;

    f.q = quat_integrate(f.q, w, dt);

    N = 6;
    Phi = eye(N);
    Phi(1:3,1:3) = eye(3) - skew(w)*dt;
    Phi(1:3,4:6) = -dt*eye(3);

    Pnew = Phi * f.P * Phi';

    qg = f.cfg.gyro_noise^2 * dt;
    qb = f.cfg.bias_noise^2 * dt;
    Pnew(1:3,1:3) = Pnew(1:3,1:3) + qg*eye(3);
    Pnew(4:6,4:6) = Pnew(4:6,4:6) + qb*eye(3);

    f.P = Pnew;
end

function [f, accepted] = mekf_update(f, meas, ref_nav, noise)
    % Direct port of vector_update() in ekf.c, PLUS the accel gate
    % from ekf_update_accel() when ref_nav == [0;0;1] (gravity).
    accepted = false;

    is_gravity = isequal(ref_nav, [0;0;1]);
    if is_gravity
        mag = norm(meas);
        if mag < 0.85 || mag > 1.15
            return;   % reject, exactly like ekf_update_accel()
        end
    end

    nz = norm(meas);
    if nz < 1e-6
        return;
    end
    z = meas / nz;

    R  = quat_to_R(f.q);
    vb = R' * ref_nav;

    N = 6;
    H = zeros(3,N);
    H(:,1:3) = skew(vb);

    PHt = f.P * H';
    S = H*PHt + noise*eye(3);

    if abs(det(S)) < 1e-12
        return;
    end
    K = PHt / S;             % P*H'*inv(S)

    y  = z - vb;
    dx = K*y;

    dth = dx(1:3); db = dx(4:6);
    dq = [1; 0.5*dth(1); 0.5*dth(2); 0.5*dth(3)];
    f.q = quat_normalize(quat_mul(f.q, dq));
    f.bias = f.bias + db;

    f.P = f.P - K*(H*f.P);
    f.P = 0.5*(f.P + f.P');   % re-symmetrise, matches ekf.c

    accepted = true;
end

function [roll, pitch, yaw] = euler_from_q(q)
    w=q(1); x=q(2); y=q(3); z=q(4);
    roll  = atan2(2*(w*x+y*z), 1-2*(x^2+y^2));
    sinp  = 2*(w*y-z*x); sinp = max(min(sinp,1),-1);
    pitch = asin(sinp);
    yaw   = atan2(2*(w*z+x*y), 1-2*(y^2+z^2));
end

function [roll,pitch,yaw] = euler_from_q_series(Q)
    N = size(Q,1);
    roll=zeros(N,1); pitch=zeros(N,1); yaw=zeros(N,1);
    for k=1:N
        [roll(k),pitch(k),yaw(k)] = euler_from_q(Q(k,:)');
    end
end

function ang_deg = quat_angle_error_deg(q_est, q_true)
    % Angle of the error rotation q_true^-1 (x) q_est, in degrees.
    q_true_inv = [q_true(1); -q_true(2:4)];   % unit quaternion inverse = conjugate
    q_e = quat_mul(q_true_inv, q_est);
    q_e = q_e / norm(q_e);
    w = min(max(abs(q_e(1)),-1),1);
    ang_deg = rad2deg(2*acos(w));
end

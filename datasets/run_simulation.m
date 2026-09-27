% run_simulation.m -- simulation driver for the quadruple-tank process.
%
% model_paramters.m supplies the constants and QTank_Dynamics.m the derivatives;
% neither produces data on its own. This file adds what they require: initial
% levels, the pump and disturbance signals, the integration loop, application of
% the declared noise, sampling, and CSV output.
%
% The Python port generate_qtank_data.py is what actually generated the data in
% generated/ (no MATLAB on the build machine). This file is the MATLAB
% equivalent, kept so the project stays runnable under MATLAB.
%
% Two departures from model_paramters.m, both deliberate:
%   * Those parameters encode only the paper's P+ (nonminimum-phase) point.
%     P- is added below from the paper's published values, not from the .m file.
%   * U_k and D_k are arguments to QTank_Dynamics, so their design is chosen
%     here: PRBS for both, as in the paper's identification experiments.

clear; close all;
global QTank

rng(0);

Ts        = 5;        % s, sampling interval
HOURS     = 4;        % simulated duration
BURN_IN   = 600;      % s, discarded before recording
U_AMP     = 0.30;     % V, PRBS amplitude on each pump
D_AMP     = 0.50;     % PRBS amplitude on the disturbance
U_HOLD    = 30;       % s, pump PRBS switching interval (~T_dom/3)
D_HOLD    = 50;       % s, disturbance interval, offset from U_HOLD on purpose

points = {'P_plus', 'P_minus'};

for pi = 1:numel(points)
    name = points{pi};
    model_paramters;                       % QTank <- the .m defaults (P+)

    if strcmp(name, 'P_plus')
        h0 = [12.6; 13.0; 4.8; 4.9];  v0 = [3.15; 3.15];
    else                                   % P- : paper values, not the .m file
        QTank.gam1 = 0.70;  QTank.gam2 = 0.60;
        QTank.k1   = 3.33;  QTank.k2   = 3.35;
        h0 = [12.4; 12.7; 1.8; 1.4];  v0 = [3.00; 3.00];
    end

    nBurn = round(BURN_IN / Ts);
    nKeep = round(HOURS * 3600 / Ts);
    n     = nBurn + nKeep;

    uCmd = [v0(1) + prbs(n, round(U_HOLD/Ts), U_AMP), ...
            v0(2) + prbs(n, round(U_HOLD/Ts), U_AMP)];
    dCmd =          prbs(n, round(D_HOLD/Ts), D_AMP);

    % R_mat: unmeasured input disturbance. The plant sees command + noise while
    % the CSV records the command, so the recorded v1/v2 stay exactly binary.
    uAct = uCmd + sqrt(QTank.R_mat(1,1)) * randn(n, 2);

    h = h0;  levels = zeros(n, 4);
    for i = 1:n
        [~, Y] = ode45(@(t, x) QTank_Dynamics(t, x, uAct(i,:), dCmd(i)), [0 Ts], h);
        h = Y(end, :)';
        levels(i, :) = h';
    end

    measured = levels + sqrt(QTank.Q_mat(1,1)) * randn(n, 4);   % Q_mat
    data     = [uCmd(nBurn+1:end, :), dCmd(nBurn+1:end), measured(nBurn+1:end, :)];

    outDir = fullfile(fileparts(mfilename('fullpath')), 'generated');
    if ~exist(outDir, 'dir'); mkdir(outDir); end
    T = array2table(data, 'VariableNames', {'v1','v2','d','h1','h2','h3','h4'});
    writetable(T, fullfile(outDir, ['qtank_' name '.csv']));
    fprintf('%s: %d rows written\n', name, size(data, 1));
end


function s = prbs(n, hold, amplitude)
% Pseudorandom binary sequence, +/-amplitude, switching every `hold` samples.
% Binary rather than Gaussian: that is what the paper used, and VAR-LiNGAM's
% identifiability depends on the excitation being non-Gaussian.
    nHolds = ceil(n / hold);
    levels = amplitude * (2 * (rand(nHolds, 1) > 0.5) - 1);
    s = repelem(levels, hold);
    s = s(1:n);
end

%% run_ACTIVSg70k.m
%
% Run and diagnose the ACTIVSg70k MATPOWER case.
%
% REQUIREMENTS
% ------------
% 1. MATPOWER must be installed and on the MATLAB path.
% 2. Rename the supplied case file:
%
%       case_ACTIVSg70k(1).m
%
%    to:
%
%       case_ACTIVSg70k.m
%
% 3. Put this script in the same directory as case_ACTIVSg70k.m,
%    or otherwise make sure the case file is on the MATLAB path.
%
% This script:
%   - loads the original MATPOWER case;
%   - preserves its supplied Vm/Va operating point for comparison;
%   - runs an AC Newton power flow;
%   - reports convergence and system balance;
%   - reports voltage violations;
%   - reports generator P/Q limit violations;
%   - calculates branch MVA loading and losses;
%   - identifies the most heavily loaded branches;
%   - summarizes results by voltage level;
%   - summarizes transformer tap ratios;
%   - compares the solved state against the supplied operating point;
%   - saves the solved MATPOWER case and useful CSV tables.

clear;
clc;

%% ========================================================================
%  MATPOWER SETUP
% =========================================================================

fprintf('\n');
fprintf('====================================================================================================\n');
fprintf('ACTIVSg70k MATPOWER AC POWER FLOW\n');
fprintf('====================================================================================================\n');

% Confirm MATPOWER is available
if exist('runpf', 'file') ~= 2
    error(['MATPOWER does not appear to be on the MATLAB path.\n' ...
           'Run startup.m from your MATPOWER installation first.']);
end

fprintf('MATPOWER runpf location:\n');
fprintf('  %s\n\n', which('runpf'));


%% ========================================================================
%  LOAD CASE
% =========================================================================

case_name = 'case_ACTIVSg70k';

fprintf('Loading case: %s\n', case_name);

mpc0 = loadcase(case_name);

fprintf('Case loaded successfully.\n\n');

fprintf('System size:\n');
fprintf('  Buses:       %d\n', size(mpc0.bus, 1));
fprintf('  Generators:  %d\n', size(mpc0.gen, 1));
fprintf('  Branches:    %d\n', size(mpc0.branch, 1));
fprintf('  Base MVA:    %.1f\n', mpc0.baseMVA);

if isfield(mpc0, 'gencost')
    fprintf('  Gen costs:   %d\n', size(mpc0.gencost, 1));
end

fprintf('\n');


%% ========================================================================
%  MATPOWER COLUMN INDICES
% =========================================================================

define_constants;


%% ========================================================================
%  SAVE SUPPLIED OPERATING POINT
% =========================================================================

Vm_original = mpc0.bus(:, VM);
Va_original = mpc0.bus(:, VA);

Pg_original = mpc0.gen(:, PG);
Qg_original = mpc0.gen(:, QG);


%% ========================================================================
%  BASIC INPUT-CASE SUMMARY
% =========================================================================

fprintf('====================================================================================================\n');
fprintf('SUPPLIED CASE SUMMARY\n');
fprintf('====================================================================================================\n');

active_gen0 = mpc0.gen(:, GEN_STATUS) > 0;
active_branch0 = mpc0.branch(:, BR_STATUS) > 0;

Pload0 = sum(mpc0.bus(:, PD));
Qload0 = sum(mpc0.bus(:, QD));

Pgen0 = sum(mpc0.gen(active_gen0, PG));
Qgen0 = sum(mpc0.gen(active_gen0, QG));

fprintf('Total real load:             %12.2f MW\n', Pload0);
fprintf('Total reactive load:         %12.2f Mvar\n', Qload0);
fprintf('Supplied real generation:    %12.2f MW\n', Pgen0);
fprintf('Supplied reactive generation:%12.2f Mvar\n', Qgen0);

fprintf('\nSupplied voltage range:\n');
fprintf('  Minimum Vm: %.6f pu\n', min(Vm_original));
fprintf('  Maximum Vm: %.6f pu\n', max(Vm_original));
fprintf('  Mean Vm:    %.6f pu\n', mean(Vm_original));

fprintf('\n');


%% ========================================================================
%  POWER FLOW OPTIONS
% =========================================================================

% Newton-Raphson AC power flow.
mpopt = mpoption( ...
    'pf.alg', 'NR', ...
    'pf.tol', 1e-8, ...
    'pf.nr.max_it', 20, ...
    'pf.enforce_q_lims', 1, ...
    'verbose', 1, ...
    'out.all', 0);

% Note:
% pf.enforce_q_lims = 0 deliberately runs the ordinary MATPOWER power flow
% first. We diagnose reactive limit violations afterwards instead of
% automatically converting PV buses to PQ buses.


%% ========================================================================
%  RUN AC POWER FLOW
% =========================================================================

fprintf('====================================================================================================\n');
fprintf('RUNNING AC POWER FLOW\n');
fprintf('====================================================================================================\n');

tic;
[results, success] = runpf(mpc0, mpopt);
elapsed = toc;

fprintf('\n');
fprintf('Power-flow runtime: %.3f seconds\n', elapsed);

if success
    fprintf('STATUS: AC POWER FLOW CONVERGED\n');
else
    fprintf('STATUS: AC POWER FLOW DID NOT CONVERGE\n');
    warning('Power flow did not converge. Diagnostics below may not be meaningful.');
end

fprintf('\n');


%% ========================================================================
%  SYSTEM BALANCE
% =========================================================================

fprintf('====================================================================================================\n');
fprintf('SYSTEM POWER BALANCE\n');
fprintf('====================================================================================================\n');

active_gen = results.gen(:, GEN_STATUS) > 0;
active_branch = results.branch(:, BR_STATUS) > 0;

Pload = sum(results.bus(:, PD));
Qload = sum(results.bus(:, QD));

Pgen = sum(results.gen(active_gen, PG));
Qgen = sum(results.gen(active_gen, QG));

% Branch real/reactive losses from terminal flows.
P_loss_branch = results.branch(:, PF) + results.branch(:, PT);
Q_loss_branch = results.branch(:, QF) + results.branch(:, QT);

total_P_loss = sum(P_loss_branch(active_branch));
total_Q_absorption = sum(Q_loss_branch(active_branch));

fprintf('Total real load:              %12.3f MW\n', Pload);
fprintf('Total reactive load:          %12.3f Mvar\n', Qload);
fprintf('Total real generation:        %12.3f MW\n', Pgen);
fprintf('Total reactive generation:    %12.3f Mvar\n', Qgen);

fprintf('\n');
fprintf('Branch real-power losses:     %12.3f MW\n', total_P_loss);
fprintf('Branch net reactive exchange: %12.3f Mvar\n', total_Q_absorption);

fprintf('\n');
fprintf('Generation - load (P):        %12.3f MW\n', Pgen - Pload);
fprintf('Generation - load (Q):        %12.3f Mvar\n', Qgen - Qload);

fprintf('\n');


%% ========================================================================
%  BUS VOLTAGES
% =========================================================================

fprintf('====================================================================================================\n');
fprintf('BUS VOLTAGES\n');
fprintf('====================================================================================================\n');

Vm = results.bus(:, VM);
Va = results.bus(:, VA);
bus_number = results.bus(:, BUS_I);

[Vm_min, i_min] = min(Vm);
[Vm_max, i_max] = max(Vm);

fprintf('Minimum voltage: %.6f pu at bus %d (%.1f kV)\n', ...
    Vm_min, bus_number(i_min), results.bus(i_min, BASE_KV));

fprintf('Maximum voltage: %.6f pu at bus %d (%.1f kV)\n', ...
    Vm_max, bus_number(i_max), results.bus(i_max, BASE_KV));

fprintf('Mean voltage:    %.6f pu\n', mean(Vm));
fprintf('Median voltage:  %.6f pu\n', median(Vm));
fprintf('Std. dev.:       %.6f pu\n', std(Vm));

% Violations against case-specific limits
low_v = Vm < results.bus(:, VMIN);
high_v = Vm > results.bus(:, VMAX);

fprintf('\nVoltage-limit violations:\n');
fprintf('  Below Vmin: %d\n', nnz(low_v));
fprintf('  Above Vmax: %d\n', nnz(high_v));
fprintf('  Total:      %d\n', nnz(low_v | high_v));

% Also report more operationally interesting thresholds
fprintf('\nVoltage distribution:\n');
fprintf('  Vm < 0.90:       %6d\n', nnz(Vm < 0.90));
fprintf('  Vm < 0.95:       %6d\n', nnz(Vm < 0.95));
fprintf('  Vm < 0.97:       %6d\n', nnz(Vm < 0.97));
fprintf('  0.97-1.03:       %6d\n', nnz(Vm >= 0.97 & Vm <= 1.03));
fprintf('  Vm > 1.03:       %6d\n', nnz(Vm > 1.03));
fprintf('  Vm > 1.05:       %6d\n', nnz(Vm > 1.05));
fprintf('  Vm > 1.10:       %6d\n', nnz(Vm > 1.10));

fprintf('\n');


%% ========================================================================
%  GENERATOR LIMITS
% =========================================================================

fprintf('====================================================================================================\n');
fprintf('GENERATOR LIMITS\n');
fprintf('====================================================================================================\n');

Pg = results.gen(:, PG);
Qg = results.gen(:, QG);

Pmax = results.gen(:, PMAX);
Pmin = results.gen(:, PMIN);
Qmax = results.gen(:, QMAX);
Qmin = results.gen(:, QMIN);

tol = 1e-5;

p_above = active_gen & Pg > Pmax + tol;
p_below = active_gen & Pg < Pmin - tol;

q_above = active_gen & Qg > Qmax + tol;
q_below = active_gen & Qg < Qmin - tol;

fprintf('Active generators: %d\n', nnz(active_gen));

fprintf('\nReal-power violations:\n');
fprintf('  Above Pmax: %d\n', nnz(p_above));
fprintf('  Below Pmin: %d\n', nnz(p_below));

fprintf('\nReactive-power violations:\n');
fprintf('  Above Qmax: %d\n', nnz(q_above));
fprintf('  Below Qmin: %d\n', nnz(q_below));
fprintf('  Total Q-limit violations: %d\n', nnz(q_above | q_below));

% Generator Q utilization, where meaningful
Qcap_pos = Qmax;
Qcap_neg = abs(Qmin);

q_util = nan(size(Qg));

pos = active_gen & Qg >= 0 & Qcap_pos > 0;
neg = active_gen & Qg < 0 & Qcap_neg > 0;

q_util(pos) = Qg(pos) ./ Qcap_pos(pos);
q_util(neg) = abs(Qg(neg)) ./ Qcap_neg(neg);

fprintf('\nReactive capability utilization:\n');
fprintf('  |Q| >=  30%% directional limit: %d\n', nnz(q_util >= 0.30));
fprintf('  |Q| >=  50%% directional limit: %d\n', nnz(q_util >= 0.50));
fprintf('  |Q| >=  90%% directional limit: %d\n', nnz(q_util >= 0.90));
fprintf('  |Q| >=  99%% directional limit: %d\n', nnz(q_util >= 0.99));

fprintf('\n');


%% ========================================================================
%  BRANCH LOADING
% =========================================================================

fprintf('====================================================================================================\n');
fprintf('BRANCH LOADING\n');
fprintf('====================================================================================================\n');

branch = results.branch;

Sf = sqrt(branch(:, PF).^2 + branch(:, QF).^2);
St = sqrt(branch(:, PT).^2 + branch(:, QT).^2);

Smax = max(Sf, St);

rateA = branch(:, RATE_A);

loading_pct = nan(size(rateA));

rated = active_branch & rateA > 0;

loading_pct(rated) = 100 .* Smax(rated) ./ rateA(rated);

fprintf('Active branches:            %d\n', nnz(active_branch));
fprintf('Branches with RATE_A > 0:   %d\n', nnz(rated));
fprintf('Branches without rating:    %d\n', nnz(active_branch & rateA <= 0));

if any(rated)
    valid_loading = loading_pct(rated);

    fprintf('\nRated branch loading:\n');
    fprintf('  Mean:    %8.2f %%\n', mean(valid_loading));
    fprintf('  Median:  %8.2f %%\n', median(valid_loading));
    fprintf('  90th:    %8.2f %%\n', prctile(valid_loading, 90));
    fprintf('  95th:    %8.2f %%\n', prctile(valid_loading, 95));
    fprintf('  99th:    %8.2f %%\n', prctile(valid_loading, 99));
    fprintf('  Maximum: %8.2f %%\n', max(valid_loading));

    fprintf('\nLoading counts:\n');
    fprintf('  >  50%%: %6d\n', nnz(valid_loading > 50));
    fprintf('  >  75%%: %6d\n', nnz(valid_loading > 75));
    fprintf('  >  90%%: %6d\n', nnz(valid_loading > 90));
    fprintf('  > 100%%: %6d\n', nnz(valid_loading > 100));
end

fprintf('\n');


%% ========================================================================
%  WORST-LOADED BRANCHES
% =========================================================================

fprintf('====================================================================================================\n');
fprintf('20 MOST HEAVILY LOADED RATED BRANCHES\n');
fprintf('====================================================================================================\n');

rated_idx = find(rated);

[~, order] = sort(loading_pct(rated_idx), 'descend');

nshow = min(20, length(order));
worst_idx = rated_idx(order(1:nshow));

fprintf('%8s %8s %8s %10s %10s %10s %10s %10s\n', ...
    'row', 'from', 'to', 'rateA', 'Sf', 'St', 'load_%', 'P_loss');

for k = 1:nshow
    i = worst_idx(k);

    fprintf('%8d %8d %8d %10.2f %10.2f %10.2f %10.2f %10.3f\n', ...
        i, ...
        branch(i, F_BUS), ...
        branch(i, T_BUS), ...
        branch(i, RATE_A), ...
        Sf(i), ...
        St(i), ...
        loading_pct(i), ...
        P_loss_branch(i));
end

fprintf('\n');


%% ========================================================================
%  TRANSFORMERS / TAP RATIOS
% =========================================================================

fprintf('====================================================================================================\n');
fprintf('TRANSFORMERS AND TAP RATIOS\n');
fprintf('====================================================================================================\n');

% In MATPOWER, TAP = 0 denotes a nominal 1.0 ratio.
tap_raw = branch(:, TAP);

is_transformer = active_branch & tap_raw ~= 0;

tap_effective = tap_raw;
tap_effective(tap_effective == 0) = 1;

fprintf('Branches with explicit nonzero tap: %d\n', nnz(is_transformer));

if any(is_transformer)
    taps = tap_effective(is_transformer);

    fprintf('Tap ratio min:    %.6f\n', min(taps));
    fprintf('Tap ratio median: %.6f\n', median(taps));
    fprintf('Tap ratio mean:   %.6f\n', mean(taps));
    fprintf('Tap ratio max:    %.6f\n', max(taps));

    fprintf('Off-nominal taps: %d\n', nnz(abs(taps - 1) > 1e-9));
end

fprintf('\n');


%% ========================================================================
%  RESULTS BY VOLTAGE LEVEL
% =========================================================================

fprintf('====================================================================================================\n');
fprintf('SUMMARY BY BUS VOLTAGE LEVEL\n');
fprintf('====================================================================================================\n');

base_kv = results.bus(:, BASE_KV);
voltage_levels = unique(base_kv);
voltage_levels = sort(voltage_levels, 'descend');

fprintf('%10s %10s %12s %12s %12s %12s\n', ...
    'kV', 'n_bus', 'Pload_MW', 'Qload_Mvar', 'Vm_mean', 'Vm_min');

for k = 1:length(voltage_levels)
    kv = voltage_levels(k);

    idx = base_kv == kv;

    fprintf('%10.1f %10d %12.2f %12.2f %12.5f %12.5f\n', ...
        kv, ...
        nnz(idx), ...
        sum(results.bus(idx, PD)), ...
        sum(results.bus(idx, QD)), ...
        mean(results.bus(idx, VM)), ...
        min(results.bus(idx, VM)));
end

fprintf('\n');


%% ========================================================================
%  BRANCH FLOW BY FROM-BUS VOLTAGE
% =========================================================================

fprintf('====================================================================================================\n');
fprintf('AC BRANCH FLOWS BY FROM-BUS VOLTAGE\n');
fprintf('====================================================================================================\n');

% Map bus number -> bus row
max_bus_number = max(results.bus(:, BUS_I));
bus_row = zeros(max_bus_number, 1);
bus_row(results.bus(:, BUS_I)) = (1:size(results.bus, 1))';

from_rows = bus_row(branch(:, F_BUS));
from_kv = results.bus(from_rows, BASE_KV);

branch_voltage_levels = unique(from_kv(active_branch));
branch_voltage_levels = sort(branch_voltage_levels, 'descend');

fprintf('%10s %10s %14s %14s %14s %14s\n', ...
    'kV', 'n_branch', 'median_MVA', 'mean_MVA', 'P_loss_MW', 'median_load%');

for k = 1:length(branch_voltage_levels)
    kv = branch_voltage_levels(k);

    idx = active_branch & from_kv == kv;

    load_here = loading_pct(idx);
    load_here = load_here(~isnan(load_here));

    if isempty(load_here)
        median_load = NaN;
    else
        median_load = median(load_here);
    end

    fprintf('%10.1f %10d %14.2f %14.2f %14.2f %14.2f\n', ...
        kv, ...
        nnz(idx), ...
        median(Smax(idx)), ...
        mean(Smax(idx)), ...
        sum(P_loss_branch(idx)), ...
        median_load);
end

fprintf('\n');


%% ========================================================================
%  COMPARISON WITH SUPPLIED OPERATING POINT
% =========================================================================

fprintf('====================================================================================================\n');
fprintf('SOLVED STATE VS SUPPLIED CASE STATE\n');
fprintf('====================================================================================================\n');

dVm = Vm - Vm_original;

% Voltage angles have arbitrary reference.
% Remove a common angle offset before comparing them.
dVa_raw = Va - Va_original;
angle_offset = median(dVa_raw);
dVa = dVa_raw - angle_offset;

fprintf('Voltage-magnitude differences:\n');
fprintf('  max |dVm|:     %.10f pu\n', max(abs(dVm)));
fprintf('  mean |dVm|:    %.10f pu\n', mean(abs(dVm)));
fprintf('  RMS dVm:       %.10f pu\n', sqrt(mean(dVm.^2)));

fprintf('\nVoltage-angle differences after removing common offset:\n');
fprintf('  common offset: %.8f degrees\n', angle_offset);
fprintf('  max |dVa|:     %.8f degrees\n', max(abs(dVa)));
fprintf('  mean |dVa|:    %.8f degrees\n', mean(abs(dVa)));
fprintf('  RMS dVa:       %.8f degrees\n', sqrt(mean(dVa.^2)));

dPg = Pg - Pg_original;
dQg = Qg - Qg_original;

fprintf('\nGenerator output differences:\n');
fprintf('  max |dPg|:     %.6f MW\n', max(abs(dPg(active_gen))));
fprintf('  mean |dPg|:    %.6f MW\n', mean(abs(dPg(active_gen))));
fprintf('  max |dQg|:     %.6f Mvar\n', max(abs(dQg(active_gen))));
fprintf('  mean |dQg|:    %.6f Mvar\n', mean(abs(dQg(active_gen))));

fprintf('\n');


%% ========================================================================
%  CREATE TABLES
% =========================================================================

bus_table = table( ...
    results.bus(:, BUS_I), ...
    results.bus(:, BUS_TYPE), ...
    results.bus(:, BASE_KV), ...
    results.bus(:, PD), ...
    results.bus(:, QD), ...
    Vm_original, ...
    Va_original, ...
    Vm, ...
    Va, ...
    dVm, ...
    dVa, ...
    'VariableNames', { ...
        'bus', 'type', 'base_kv', ...
        'pd_mw', 'qd_mvar', ...
        'vm_original_pu', 'va_original_deg', ...
        'vm_solved_pu', 'va_solved_deg', ...
        'delta_vm_pu', 'delta_va_deg'});

gen_table = table( ...
    results.gen(:, GEN_BUS), ...
    results.gen(:, GEN_STATUS), ...
    Pg, ...
    Qg, ...
    Pmin, ...
    Pmax, ...
    Qmin, ...
    Qmax, ...
    q_util, ...
    'VariableNames', { ...
        'bus', 'status', ...
        'pg_mw', 'qg_mvar', ...
        'pmin_mw', 'pmax_mw', ...
        'qmin_mvar', 'qmax_mvar', ...
        'q_limit_utilization'});

branch_table = table( ...
    (1:size(branch,1))', ...
    branch(:, F_BUS), ...
    branch(:, T_BUS), ...
    from_kv, ...
    branch(:, BR_STATUS), ...
    branch(:, RATE_A), ...
    branch(:, PF), ...
    branch(:, QF), ...
    branch(:, PT), ...
    branch(:, QT), ...
    Sf, ...
    St, ...
    Smax, ...
    loading_pct, ...
    P_loss_branch, ...
    Q_loss_branch, ...
    tap_effective, ...
    'VariableNames', { ...
        'branch_row', ...
        'from_bus', 'to_bus', ...
        'from_base_kv', ...
        'status', ...
        'rate_a_mva', ...
        'pf_mw', 'qf_mvar', ...
        'pt_mw', 'qt_mvar', ...
        's_from_mva', 's_to_mva', ...
        's_max_mva', ...
        'loading_pct', ...
        'p_loss_mw', ...
        'q_net_mvar', ...
        'tap_ratio'});


%% ========================================================================
%  SAVE OUTPUT
% =========================================================================

fprintf('====================================================================================================\n');
fprintf('SAVING RESULTS\n');
fprintf('====================================================================================================\n');

save('ACTIVSg70k_pf_results.mat', ...
    'results', 'success', ...
    'bus_table', 'gen_table', 'branch_table');

writetable(bus_table, 'ACTIVSg70k_bus_results.csv');
writetable(gen_table, 'ACTIVSg70k_generator_results.csv');
writetable(branch_table, 'ACTIVSg70k_branch_results.csv');

fprintf('Saved:\n');
fprintf('  ACTIVSg70k_pf_results.mat\n');
fprintf('  ACTIVSg70k_bus_results.csv\n');
fprintf('  ACTIVSg70k_generator_results.csv\n');
fprintf('  ACTIVSg70k_branch_results.csv\n');

fprintf('\n');


%% ========================================================================
%  SIMPLE PLOTS
% =========================================================================

figure;
histogram(Vm, 100);
xlabel('Voltage magnitude (pu)');
ylabel('Number of buses');
title('ACTIVSg70k Bus Voltage Distribution');
grid on;

if any(rated)
    figure;
    histogram(loading_pct(rated), 100);
    xlabel('Branch loading (%)');
    ylabel('Number of branches');
    title('ACTIVSg70k Branch Loading Distribution');
    grid on;
end

figure;
scatter(base_kv, Vm, 5, 'filled');
xlabel('Nominal voltage (kV)');
ylabel('Voltage magnitude (pu)');
title('Bus Voltage by Nominal Voltage Level');
grid on;


%% ========================================================================
%  FINAL STATUS
% =========================================================================

fprintf('====================================================================================================\n');

if success
    fprintf('DONE: ACTIVSg70k AC POWER FLOW CONVERGED SUCCESSFULLY.\n');
else
    fprintf('DONE: ACTIVSg70k AC POWER FLOW DID NOT CONVERGE.\n');
end

fprintf('====================================================================================================\n');
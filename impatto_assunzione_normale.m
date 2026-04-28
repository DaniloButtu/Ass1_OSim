% Quanto sbaglio quando assumo che la domanda sia distribuita normalmente?
clear; clc;
seed = min(353244,362747);
rng(seed);

% Parametri economici
p = 30; c = 15; s = 8;
cu = p - c;
co = c - s;
alpha = cu / (cu + co);          % rapporto critico
z_norm = norminv(alpha);         % quantile normale standard

% Parametri delle distribuzioni vere (fisso media e varianza)
mu_true = 200;
sigma_true = 80;
var_true = sigma_true^2;

% Metto le distribuzioni in una struct
%Scelgo i parsmetri delle distribuzioni in modo da avere media e varianza
%uguali a mu_true e var_true

% Gamma
k_g = mu_true^2 / var_true;
theta_g = var_true / mu_true;
distr(1).name = 'Gamma';
distr(1).rnd = @(n) gamrnd(k_g, theta_g, n, 1);
distr(1).pd = makedist('Gamma', 'a', k_g, 'b', theta_g);
distr(1).mean = mu_true;
distr(1).var = var_true;

% Esponenziale
distr(2).name = 'Esponenziale';
distr(2).rnd = @(n) exprnd(mu_true, n, 1);
distr(2).pd = makedist('Exponential', 'mu', mu_true);
distr(2).mean = mu_true;
distr(2).var = mu_true^2;

% Lognormale
sigma_ln = sqrt(log(var_true / mu_true^2 + 1));
mu_ln = log(mu_true) - sigma_ln^2 / 2;
distr(3).name = 'Lognormale';
distr(3).rnd = @(n) lognrnd(mu_ln, sigma_ln, n, 1);
distr(3).pd = makedist('Lognormal', 'mu', mu_ln, 'sigma', sigma_ln);
distr(3).mean = mu_true;
distr(3).var = var_true;

% Uniforme
half_range = sqrt(3 * var_true);
a = mu_true - half_range;
b = mu_true + half_range;
distr(4).name = 'Uniforme';
distr(4).rnd = @(n) a + (b - a) * rand(n, 1);
distr(4).pd = makedist('Uniform', 'lower', a, 'upper', b);
distr(4).mean = mu_true;
distr(4).var = var_true;

num_distr = length(distr);

Q_opt_quantile = @(pd, alpha) icdf(pd, alpha);

% Profitto atteso secondo la formula:
% profitto_atteso = p * E[min(Q,D)] + s * E[max(Q-D,0)] - c*Q
expectedProfit = @(Q, pd) ...
    (p * integral(@(x) min(Q, x) .* pdf(pd, x), ...
                  max(0, icdf(pd, 1e-12)), icdf(pd, 1-1e-12)) ...
    + s * integral(@(x) max(Q - x, 0) .* pdf(pd, x), ...
                   max(0, icdf(pd, 1e-12)), icdf(pd, 1-1e-12)) ...
    - c * Q);

%% Simulazione con N fissato
N = 30;             % dimensione campionaria per la stima
M_outer = 2000;     % numero di repliche

Q_opt_true_all = zeros(num_distr, 1);
EP_opt_true_all = zeros(num_distr, 1);
ratio_profit = zeros(num_distr, 1);
bias_Q = zeros(num_distr, 1);
Q_est_cell = cell(num_distr, 1);

for d = 1:num_distr
    fprintf('\nDistribuzione: %s\n', distr(d).name);
    
    % Quantità ottima esatta e profitto ottimo
    Q_opt = Q_opt_quantile(distr(d).pd, alpha);
    EP_opt = expectedProfit(Q_opt, distr(d).pd);
    Q_opt_true_all(d) = Q_opt;
    EP_opt_true_all(d) = EP_opt;
    fprintf('Q* vera = %.2f, Profitto* = %.2f\n', Q_opt, EP_opt);
    
    % Simulazione con approccio normale
    Q_est_all = zeros(M_outer, 1);
    profit_est_all = zeros(M_outer, 1);
    
    for sim = 1:M_outer
        sample = distr(d).rnd(N);
        mu_hat = mean(sample);
        sigma_hat = std(sample);
        Q_est = mu_hat + z_norm * sigma_hat;
        Q_est = max(Q_est, 0);
        Q_est_all(sim) = Q_est;
        profit_est_all(sim) = expectedProfit(Q_est, distr(d).pd);
    end
    
    mean_profit_est = mean(profit_est_all);
    ratio = mean_profit_est / EP_opt;
    bias = mean(Q_est_all) - Q_opt;
    ratio_profit(d) = ratio;
    bias_Q(d) = bias;
    Q_est_cell{d} = Q_est_all;
    
    fprintf('Rapporto profitto medio = %.4f (perdita %.2f%%)\n', ratio, (1-ratio)*100);
    fprintf('Bias medio di Q = %.2f (%.2f%% di Q*)\n', bias, bias/Q_opt*100);
end

% Plot
figure;
bar(ratio_profit);
set(gca, 'XTickLabel', {distr.name});
ylabel('Rapporto profitto medio');
title('Performance relativa dell''assunzione normale');
grid on; ylim([0.8, 1]);

figure;
hold on;
all_Q_est = []; group = [];
for d = 1:num_distr
    all_Q_est = [all_Q_est; Q_est_cell{d}];
    group = [group; d * ones(length(Q_est_cell{d}), 1)];
end
boxplot(all_Q_est, group, 'Labels', {distr.name});
for d = 1:num_distr
    line([d-0.3, d+0.3], [Q_opt_true_all(d), Q_opt_true_all(d)], ...
         'Color','g','LineWidth',2,'LineStyle','--');
end
ylabel('Quantità ordinata');
title('Distribuzione di q_{est} vs q^* vera (linea verde)');
grid on;

%% Effetto del coefficiente di variazione (CV) su Gamma e Lognormale
CV_values = 0.1:0.1:0.8;
num_CV = length(CV_values);
ratio_gamma_cv = zeros(num_CV, 1);
ratio_logn_cv = zeros(num_CV, 1);
N = 30;
M_outer = 1000;
mu_fixed = 200;

for i = 1:num_CV
    CV = CV_values(i);
    sigma_cur = CV * mu_fixed;
    var_cur = sigma_cur^2;
    
    % Gamma
    k_g = mu_fixed^2 / var_cur;
    theta_g = var_cur / mu_fixed;
    pd_gamma = makedist('Gamma', 'a', k_g, 'b', theta_g);
    rnd_g = @(n) gamrnd(k_g, theta_g, n, 1);
    Q_opt_g = Q_opt_quantile(pd_gamma, alpha);
    EP_opt_g = expectedProfit(Q_opt_g, pd_gamma);
    
    temp_ratio = zeros(M_outer,1);
    for sim = 1:M_outer
        sample = rnd_g(N);
        Q_est = mean(sample) + z_norm * std(sample);
        Q_est = max(Q_est,0);
        temp_ratio(sim) = expectedProfit(Q_est, pd_gamma) / EP_opt_g;
    end
    ratio_gamma_cv(i) = mean(temp_ratio);
    
    % Lognormale
    sigma_ln = sqrt(log(var_cur / mu_fixed^2 + 1));
    mu_ln = log(mu_fixed) - sigma_ln^2 / 2;
    pd_logn = makedist('Lognormal', 'mu', mu_ln, 'sigma', sigma_ln);
    rnd_ln = @(n) lognrnd(mu_ln, sigma_ln, n, 1);
    Q_opt_ln = Q_opt_quantile(pd_logn, alpha);
    EP_opt_ln = expectedProfit(Q_opt_ln, pd_logn);
    
    temp_ratio = zeros(M_outer,1);
    for sim = 1:M_outer
        sample = rnd_ln(N);
        Q_est = mean(sample) + z_norm * std(sample);
        Q_est = max(Q_est,0);
        temp_ratio(sim) = expectedProfit(Q_est, pd_logn) / EP_opt_ln;
    end
    ratio_logn_cv(i) = mean(temp_ratio);
end

figure;
plot(CV_values, ratio_gamma_cv, 'bo-', 'LineWidth',1.5, 'DisplayName','Gamma');
hold on;
plot(CV_values, ratio_logn_cv, 'rs-', 'LineWidth',1.5, 'DisplayName','Lognormale');
xlabel('CV = \sigma/\mu');
ylabel('Rapporto profitto medio');
title('Impatto del CV sulla performance');
legend('Location','best'); grid on;

%% Effetto della dimensione campionaria N
distr_idx = [1, 3, 4];           % Gamma, Lognormale, Uniforme
line_sty = {'b-o','r-s','g-^'};
N_values = [5, 10, 20, 30, 50, 100, 200, 500];
num_N = length(N_values);
M_outer = 1000;
ratio_vs_N = zeros(length(distr_idx), num_N);

for d_idx = 1:length(distr_idx)
    d = distr_idx(d_idx);
    fprintf('\nAnalisi N per %s\n', distr(d).name);
    Q_opt = Q_opt_quantile(distr(d).pd, alpha);
    EP_opt = expectedProfit(Q_opt, distr(d).pd);
    fprintf('Q* = %.2f, Profitto* = %.2f\n', Q_opt, EP_opt);
    
    for n_idx = 1:num_N
        N_cur = N_values(n_idx);
        temp_ratios = zeros(M_outer,1);
        for sim = 1:M_outer
            sample = distr(d).rnd(N_cur);
            Q_est = mean(sample) + z_norm * std(sample);
            Q_est = max(Q_est,0);
            temp_ratios(sim) = expectedProfit(Q_est, distr(d).pd) / EP_opt;
        end
        ratio_vs_N(d_idx, n_idx) = mean(temp_ratios);
        fprintf('N = %3d, Rapporto = %.4f\n', N_cur, ratio_vs_N(d_idx, n_idx));
    end
end

figure;
hold on;
for d_idx = 1:length(distr_idx)
    plot(N_values, ratio_vs_N(d_idx,:), line_sty{d_idx}, ...
         'LineWidth',1.5, 'MarkerSize',6, 'DisplayName', distr(distr_idx(d_idx)).name);
end
xlabel('Dimensione campionaria N');
ylabel('Rapporto profitto medio');
title('Convergenza all''aumentare di N');
legend('Location','best'); grid on;
xlim([min(N_values) max(N_values)]); ylim([0.85 1]);
yline(1, 'k--', 'LineWidth',0.8, 'HandleVisibility','off');

% Limite asintotico
fprintf('\nLimite asintotico:\n');
for d_idx = 1:length(distr_idx)
    d = distr_idx(d_idx);
    mu_cur = distr(d).mean;
    sigma_cur = sqrt(distr(d).var);
    Q_inf = mu_cur + z_norm * sigma_cur;
    profit_inf = expectedProfit(Q_inf, distr(d).pd);
    ratio_inf = profit_inf / EP_opt_true_all(d);
    fprintf('%s: Q = %.2f, rapporto = %.4f (perdita %.2f%%)\n', ...
            distr(d).name, Q_inf, ratio_inf, (1-ratio_inf)*100);
end
%Ottimizzazine robusta worst-case 
clear; clc;
seed = min(353244,362747);
rng(seed);

std_loss = @(z) normpdf(z) - z .* (1 - normcdf(z));

expected_profit = @(Q, mu, sigma, p, c, s) ...
    (p - c) * mu ...
    - (c - s) * sigma .* std_loss((Q - mu) ./ sigma) ...
    - (p - c) * sigma .* std_loss(-(Q - mu) ./ sigma);

% set parametri
p = 30; c = 15; s = 8;
cu = p - c;
co = c - s;
crit_ratio_nom = cu / (cu + co);

% Parametri veri della domanda (incogniti al decisore)
mu_true = 200;
sigma_true = 70;

% Stima dei parametri
N = 20; %Dimensione del campione per la stima
sample = mu_true + sigma_true * randn(N,1);
mu_hat = mean(sample);
sigma_hat = std(sample);
fprintf('Stime: mu_hat = %.2f, sigma_hat = %.2f\n', mu_hat, sigma_hat);

% Costruzione dell'insieme di incertezza con livello di confidenza 95%
alpha = 0.05;
t_val = tinv(1 - alpha/2, N-1);
delta_mu = t_val * sigma_hat / sqrt(N);
chi2_low = chi2inv(alpha/2, N-1);
chi2_high = chi2inv(1 - alpha/2, N-1);
sigma_low = sigma_hat * sqrt((N-1)/chi2_high);
sigma_high = sigma_hat * sqrt((N-1)/chi2_low);

mu_range = [mu_hat - delta_mu, mu_hat + delta_mu];
sigma_range = [sigma_low, sigma_high];

mu_samples = linspace(mu_range(1), mu_range(2), 20);
sigma_samples = linspace(sigma_range(1), sigma_range(2), 20);
[MU, SIG] = meshgrid(mu_samples, sigma_samples);
MU = MU(:); SIG = SIG(:);

% Griglia per Q
Q_grid = linspace(0, 3*mu_true, 200);
nQ = length(Q_grid);

worst_profit = zeros(nQ,1);
for i = 1:nQ
    Q = Q_grid(i);
    profits = expected_profit(Q, MU, SIG, p, c, s);
    worst_profit(i) = min(profits);
end

% Quantità robusta
[~, idx_rob] = max(worst_profit);
Q_rob = Q_grid(idx_rob);

% Quantità nominale
z = norminv(crit_ratio_nom);
Q_nom = mu_hat + z * sigma_hat;

% Quantità teorica
Q_opt_true = mu_true + z * sigma_true;

fprintf('Quantità robusta: %.2f\n', Q_rob);
fprintf('Quantità nominale: %.2f\n', Q_nom);
fprintf('Quantità ottima teorica: %.2f\n', Q_opt_true);

% Valutazione Monte Carlo delle performance (TUTTE SULLO STESSO CAMPIONE D)
M = 10000;  
D = mu_true + sigma_true * randn(M,1);

profit_on_D = @(Q) mean(p * min(Q, D) + s * max(0, Q - D) - c * Q);
profit_sim_opt = profit_on_D(Q_opt_true);
profit_sim_nom = profit_on_D(Q_nom);
profit_sim_rob = profit_on_D(Q_rob);

ratio_nom = profit_sim_nom / profit_sim_opt * 100;
ratio_rob = profit_sim_rob / profit_sim_opt * 100;

fprintf('Profitto simulato ottimo (Q*):    %.2f\n', profit_sim_opt);
fprintf('Profitto simulato nominale:       %.2f (%.2f%% dell''ottimo simulato)\n', profit_sim_nom, ratio_nom);
fprintf('Profitto simulato robusto:        %.2f (%.2f%% dell''ottimo simulato)\n', profit_sim_rob, ratio_rob);
fprintf('Differenza (nominale - robusto):  %.2f\n', profit_sim_nom - profit_sim_rob);

fprintf('Q ottima teorica (Q*):          %.2f\n', Q_opt_true);
fprintf('Q nominale (basata su stime):   %.2f\n', Q_nom);
fprintf('Q robusta (worst-case):         %.2f\n', Q_rob);

%Plot
figure;
plot(Q_grid, worst_profit, 'b-', 'LineWidth', 2);
hold on;
xline(Q_rob, 'r--', 'LineWidth', 1.5);
xline(Q_nom, 'g--', 'LineWidth', 1.5);
xlabel('Quantità Q');
ylabel('Profitto worst-case');
title('Funzione obiettivo robusta');
legend('Worst-case profit', 'Q robusta', 'Q nominale');
grid on;







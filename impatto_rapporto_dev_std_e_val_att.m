% Impatto del rapporto deviazione standard / media (CV) nel newsvendor
clear; clc;
seed = min(353244,362747);
rng(seed);

std_loss = @(z) normpdf(z) - z .* (1 - normcdf(z));
expected_profit = @(Q, mu, sigma, p, c, s) (p - c) * mu ...
    - (c - s) * sigma .* std_loss((Q - mu) ./ sigma) ...
    - (p - c) * sigma .* std_loss(-(Q - mu) ./ sigma);
optimal_quantity = @(mu, sigma, cu, co) ...
    mu + norminv(cu / (cu + co)) * sigma;

% Parametri economici fissi
p = 30; c = 20; s = 15;          % p=30, c=20, s=15
cu = p - c;                      % 10
co = c - s;                      % 5
crit_ratio = cu / (cu + co);     % 2/3

% Parametri veri della domanda (media fissa, sigma variabile)
mu_true = 100;
sigma_values = 5:5:40;           % CV da 0.05 a 0.4
CV_values = sigma_values / mu_true;
num_sigma = length(sigma_values);

%% Effetto del CV per N fissato (N = 20)
N_fixed = 20;
M = 3000;  % repliche Monte Carlo

ratio_vs_CV_fixedN = zeros(num_sigma, 1);

for i = 1:num_sigma
    sigma_true = sigma_values(i);
    Q_opt_true = optimal_quantity(mu_true, sigma_true, cu, co);
    EP_opt_true = expected_profit(Q_opt_true, mu_true, sigma_true, p, c, s);
    
    temp_ratios = zeros(M, 1);
    for rep = 1:M
        % Campione dalla vera distribuzione
        sample = mu_true + sigma_true * randn(N_fixed, 1);
        mu_hat = mean(sample);
        sigma_hat = std(sample);
        
        % Quantità stimata con formula normale
        Q_est = optimal_quantity(mu_hat, sigma_hat, cu, co);
        Q_est = max(Q_est, 0); % Q>=0
        
        % Profitto atteso vero
        EP_est = expected_profit(Q_est, mu_true, sigma_true, p, c, s);
        temp_ratios(rep) = EP_est / EP_opt_true;
    end
    ratio_vs_CV_fixedN(i) = mean(temp_ratios);
end

% plot per N fisso
figure;
plot(CV_values, ratio_vs_CV_fixedN, 'bo-', 'LineWidth', 1.5);
xlabel('CV = \sigma / \mu');
ylabel('Rapporto profitto medio');
title(sprintf('Effetto del CV sulla degradazione (N = %d)', N_fixed));
grid on;

%% Impatto di N sull'effetto del CV
N_values = [5, 10, 20, 50, 100, 200, 500];
M_N = 2000;  % repliche per lo studio con N variabile

% Matrice: righe = N, colonne = sigma
ratio_matrix = zeros(length(N_values), num_sigma);

for n_idx = 1:length(N_values)
    N = N_values(n_idx);
    
    for s_idx = 1:num_sigma
        sigma_true = sigma_values(s_idx);
        Q_opt_true = optimal_quantity(mu_true, sigma_true, cu, co);
        EP_opt_true = expected_profit(Q_opt_true, mu_true, sigma_true, p, c, s);
        
        temp_ratios = zeros(M_N, 1);
        for rep = 1:M_N
            sample = mu_true + sigma_true * randn(N, 1);
            mu_hat = mean(sample);
            sigma_hat = std(sample); 
            Q_est = optimal_quantity(mu_hat, sigma_hat, cu, co);
            Q_est = max(Q_est, 0); % Q>=0
            EP_est = expected_profit(Q_est, mu_true, sigma_true, p, c, s);
            temp_ratios(rep) = EP_est / EP_opt_true;
        end
        ratio_matrix(n_idx, s_idx) = mean(temp_ratios);
    end
end

% plot per diversi N
figure;
hold on;
colors = lines(length(N_values));
for n_idx = 1:length(N_values)
    plot(CV_values, ratio_matrix(n_idx, :), 'o-', ...
        'Color', colors(n_idx,:), 'LineWidth', 1.5, ...
        'DisplayName', sprintf('N = %d', N_values(n_idx)));
end
hold off;
xlabel('CV = \sigma / \mu');
ylabel('Rapporto profitto medio');
title('Impatto di N sull''effetto del CV');
legend('Location', 'best');
grid on;

% plot superficie 3D (CV, N, Rapporto)
figure;
[X, Y] = meshgrid(CV_values, N_values);
surf(X, Y, ratio_matrix, 'EdgeColor', 'none');
xlabel('CV = \sigma / \mu');
ylabel('Dimensione campionaria N');
zlabel('Rapporto profitto medio');
title('Superficie di degradazione al variare di CV e N');
view(45, 30);
colorbar;
grid on;

% plot heatmap
figure;
imagesc(CV_values, N_values, ratio_matrix);
set(gca, 'YDir', 'normal');
xlabel('CV = \sigma / \mu');
ylabel('N');
title('Heatmap del rapporto profitto medio');
colorbar;
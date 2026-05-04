% Impatto dei costi di overage ed underage
clear; clc;
seed = min(353244,362747);
rng(seed);

% Standard normal loss function L(z) = pdf(z) - z*(1-cdf(z))
std_loss = @(z) normpdf(z) - z .* (1 - normcdf(z));

expected_profit = @(Q, mu, sigma, p, c, s) (p - c) * mu ...
    - (c - s) * sigma .* std_loss((Q - mu) ./ sigma) ...
    - (p - c) * sigma .* std_loss(-(Q - mu) ./ sigma);

optimal_quantity = @(mu, sigma, cu, co) ...
    mu + norminv(cu / (cu + co)) * sigma;

%% Impatto del rapporto cu/co per N fissato (N=20)

% Parametri veri della domanda
mu_true = 100;
sigma_true = 30;

c = 20;                 
cu = 10; % costo di underage, cu = p - c
p = c + cu;            

co_values = 1:20;     % co = c - s -->  s = c - co
N = 20;    % dimensione campionaria per la stima
M = 3000;  % numero di repliche Monte Carlo

ratio_vs_r = zeros(size(co_values));

for i = 1:length(co_values)
    co = co_values(i);
    s = c - co;
    
    Q_opt_true = optimal_quantity(mu_true, sigma_true, cu, co);
    EP_opt_true = expected_profit(Q_opt_true, mu_true, sigma_true, p, c, s);
    
    temp_ratios = zeros(M, 1);
    for rep = 1:M
        % Campione dalla domanda vera
        sample = mu_true + sigma_true * randn(N, 1);
        mu_hat = mean(sample);
        sigma_hat = std(sample);
        
        Q_est = optimal_quantity(mu_hat, sigma_hat, cu, co);
        Q_est = max(Q_est, 0); % Q>=0
        
        % Profitto atteso vero usando Q_est
        EP_est = expected_profit(Q_est, mu_true, sigma_true, p, c, s);
        temp_ratios(rep) = EP_est / EP_opt_true;
    end
    ratio_vs_r(i) = mean(temp_ratios);
end

% plot
figure;
semilogx(cu ./ co_values, ratio_vs_r, 'ro-', 'LineWidth', 1.5);
xlabel('r = c_u / c_o (scala logaritmica)');
ylabel('Rapporto profitto medio');
title(sprintf('Effetto del rapporto di costo sulla degradazione (N = %d)', N));
grid on;


%% Impatto di N sull'effetto del rapporto cu/co
c = 20; cu = 10; p = c + cu;
mu_true = 100; sigma_true = 30;
co_values = 1:1:20;
r_values = cu ./ co_values; 

N_values = [5, 10, 20, 50, 100, 200, 500];
M = 2000;  

% Matrice per memorizzare i rapporti medi: righe = N, colonne = co
ratio_matrix = zeros(length(N_values), length(co_values));

for n_idx = 1:length(N_values)
    N = N_values(n_idx);
    for c_idx = 1:length(co_values)
        co = co_values(c_idx);
        s = c - co;
        
        Q_opt_true = optimal_quantity(mu_true, sigma_true, cu, co);
        EP_opt_true = expected_profit(Q_opt_true, mu_true, sigma_true, p, c, s);
        
        temp_ratios = zeros(M, 1);
        for rep = 1:M
            sample = mu_true + sigma_true * randn(N, 1);
            mu_hat = mean(sample);
            sigma_hat = std(sample);
            Q_est = optimal_quantity(mu_hat, sigma_hat, cu, co);
            Q_est = max(Q_est, 0);
            EP_est = expected_profit(Q_est, mu_true, sigma_true, p, c, s);
            temp_ratios(rep) = EP_est / EP_opt_true;
        end
        ratio_matrix(n_idx, c_idx) = mean(temp_ratios);
    end
end

% plot delle curve sovrapposte per diversi N
figure;
hold on;
colors = lines(length(N_values));
for n_idx = 1:length(N_values)
    semilogx(r_values, ratio_matrix(n_idx, :), 'o-', ...
        'Color', colors(n_idx,:), 'LineWidth', 1.5, ...
        'DisplayName', sprintf('N = %d', N_values(n_idx)));
end
hold off;
xlabel('r = c_u / c_o (scala log)');
ylabel('Rapporto profitto medio');
title('Impatto di N sull''effetto del rapporto di costo');
legend('Location', 'best');
grid on;

% plot superficie 3D (r, N, Rapporto)
figure;
[X, Y] = meshgrid(r_values, N_values);
surf(X, Y, ratio_matrix, 'EdgeColor', 'none');
xlabel('r = c_u / c_o');
ylabel('Dimensione campionaria N');
zlabel('Rapporto profitto medio');
title('Superficie di degradazione al variare di r e N');
set(gca, 'XScale', 'log');
view(45, 30);
colorbar;
grid on;

% plot heatmap
figure;
imagesc(r_values, N_values, ratio_matrix);
set(gca, 'YDir', 'normal');
xlabel('r = c_u / c_o');
ylabel('N');
title('Heatmap del rapporto profitto medio');
colorbar;
xticks = [0.5, 1, 2, 5, 10];
xticklabels = arrayfun(@num2str, xticks, 'UniformOutput', false);
set(gca, 'XTick', xticks, 'XTickLabel', xticklabels);


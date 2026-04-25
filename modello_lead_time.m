% Modello con Lead Time di riordino
clear; clc;
seed = min(353244,362747);
rng(seed);

% Parametri economici
p = 30; 
c1 = 15; 
c2 = 21; 
s = 8;

% Parametri veri domanda
mu_true = 200; 
sigma_true = 50;
alpha = 0.4; % frazione osservata
rho = 0.7; %coefficente di correlazione fra mattina e pomeriggio


% Parametri simulazione
M = 10000; % Numero di repliche
Past = 150; %Numero di giorni osservati
quantili_test = 0.4:0.01:0.9; 
num_q = length(quantili_test);

% Generazione degli scenari di mercato
mu1_est = zeros(M, 1);
mu2_est = zeros(M, 1);
sigma1_est = zeros(M, 1);
sigma2_est = zeros(M, 1);
D_all = zeros(M, 1);
D1_true_all = zeros(M,1);

for sim = 1:M
    hist_d1 = zeros(Past, 1);
    hist_d2 = zeros(Past, 1);
    hist_all = zeros(Past,1);
    for h = 1:Past
        Z1_h = randn();
        Z2_h = randn();
        d1_h = mu_true * alpha + sigma_true * sqrt(alpha) * Z1_h;
        d2_h = mu_true * (1 - alpha) + sigma_true * sqrt(1 - alpha) * (rho * Z1_h + sqrt(1 - rho^2) * Z2_h);
        hist_d1(h) = max(0, d1_h);
        hist_d2(h) = max(0, d2_h);
        hist_all = hist_d2 + hist_d1; 
    end

    
    mu1_est(sim) = mean(hist_d1);
    mu2_est(sim) = mean(hist_d2);
    sigma1_est(sim) = std(hist_d1);
    sigma2_est(sim) = std(hist_d2);
    mu_hat_all = mean(hist_all); 
    sigma_hat_all = std(hist_all);
    
    Z1 = randn();
    Z2 = randn();
    
    % Generiamo d1 e d2 in modo che siano correlati con fattore rho
    d1 = mu_true * alpha + sigma_true * sqrt(alpha) * Z1;
    d2 = mu_true * (1 - alpha) + sigma_true * sqrt(1 - alpha) * (rho * Z1 + sqrt(1 - rho^2) * Z2);
    
    D1_true_all(sim) = max(0, d1);
    D_all(sim) = max(0, d1) + max(0, d2);
end

% Simulazione SENZA RIORDINO (Newsvendor Classico)
crit_ratio_classic = (p - c1) / (p - s);
z_classic = norminv(crit_ratio_classic);
Q_no_reorder = max(0, mu_hat_all + z_classic * sigma_hat_all);

sales_no = min(Q_no_reorder, D_all);
leftover_no = max(0, Q_no_reorder - D_all);
profit_no_reorder = p * sales_no + s * leftover_no - c1 * Q_no_reorder;

% Simulazione CON RIORDINO
all_profits = zeros(M, num_q);
risultati_profitto = zeros(num_q, 1);

% Parametri fissi del secondo periodo
cu2 = p - c2; 
co2 = c2 - s;
crit_ratio2 = max(0, cu2 / (cu2 + co2));
z2 = norminv(crit_ratio2);

% Aggiornamento Bayesiano 
Expected_Remaining_D = mu2_est + rho * (sigma2_est ./ sigma1_est) .* (D1_true_all - mu1_est);
Expected_Remaining_D = max(0, Expected_Remaining_D);
sigma_rem = sigma2_est * sqrt(1 - rho^2);
Target_Pomeriggio = Expected_Remaining_D + z2 * sigma_rem;
D1 = D1_true_all;
D2_true = D_all - D1;


for q_idx = 1:num_q
    current_q = quantili_test(q_idx);
    z1 = norminv(current_q);
    
    % Periodo 1: Ordine iniziale
    Q1 = max(0, mu_hat_all + z1 * sigma_hat_all);
    
    % Inventario residuo e Ordine Periodo 2
    inv_rimanente = max(0, Q1 - D1); 
    Q2 = max(0, Target_Pomeriggio - inv_rimanente);
    
    % Calcolo vendite e profitto
    sales1 = min(Q1, D1); 
    sales2 = min(inv_rimanente + Q2, D2_true); 
    
    sales = sales1 + sales2;
    leftover = (inv_rimanente + Q2) - sales2;

    all_profits(:, q_idx) = p * sales + s * leftover - c1*Q1 - c2*Q2;
    risultati_profitto(q_idx) = mean(all_profits(:, q_idx));
    var_profitto(q_idx) = std(all_profits(:, q_idx));
    fprintf('Quantile: %.2f | Profitto medio: %.2f\n', current_q, risultati_profitto(q_idx));
end

% Identificazione del Migliore e del Peggiore 
[max_profit, best_idx] = max(risultati_profitto);
[min_profit, worst_idx] = min(risultati_profitto);

best_q = quantili_test(best_idx);
worst_q = quantili_test(worst_idx);
q_analitico = (c2 - c1) / (c2 - s);

% plot dell'ottimizzazione del Quantile
figure('Color', 'w', 'Position', [100, 100, 700, 400]);
plot(quantili_test, risultati_profitto, '-o', 'LineWidth', 2, 'MarkerSize', 8, 'Color', [0 0.447 0.741]);
hold on;
yline(max_profit, '--r', ['Max: ', num2str(round(max_profit,2))], 'LabelHorizontalAlignment', 'left');
xline(best_q, '--g', ['Ottimo Empirico: ', num2str(best_q)], 'LabelVerticalAlignment', 'bottom');

grid on; title('Ottimizzazione del Quantile di Primo Ordine (Q1)');
xlabel('Quantile scelto per Q1'); ylabel('Profitto Medio Atteso');
legend('Profitto Simulato', 'Max Profitto', 'Best Quantile', 'Location', 'best');

% plot della distribuzione dei Profitti 
figure('Color', 'w', 'Position', [150, 150, 800, 500]);

histogram(profit_no_reorder, 50, 'FaceAlpha', 0.4, 'FaceColor', [0.5 0.5 0.5], 'Normalization', 'probability', ...
    'DisplayName', sprintf('Senza Riordino (Media: %.0f)', mean(profit_no_reorder)));
hold on;

histogram(all_profits(:, best_idx), 50, 'FaceAlpha', 0.6, 'FaceColor', [0.4660 0.6740 0.1880], 'Normalization', 'probability', ...
    'DisplayName', sprintf('Miglior Quantile Q=%.2f (Media: %.0f)', best_q, max_profit));

title('Confronto Distribuzioni dei Profitti');
xlabel('Profitto totale realizzato');
ylabel('Frequenza Relativa');
legend('Location', 'northwest');
grid on;

fprintf('Miglior quantile empirico: %.2f (Profitto: %.2f)\n', best_q, max_profit);
fprintf('Peggior quantile empirico: %.2f (Profitto: %.2f)\n', worst_q, min_profit);
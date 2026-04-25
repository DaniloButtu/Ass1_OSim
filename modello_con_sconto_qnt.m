% Modello con sconto sulla quantità di prodotto ordinato
clear; clc;
seed = min(353244,362747);
rng(seed);

% Parametri economici
p = 30; 
s = 8;
c1 = 15;            
c_min = 11;         
B = 100;            %Soglia da superare per lo sconto
alpha = 0.025;      %Sconto sull'ulteriore giornale

% Parametri veri domanda
mu_true = 200;
sigma_true = 60;

% Funzioni di Costo 
unit_cost = @(Q) (Q <= B) .* c1 + (Q > B) .* max(c_min, c1 - alpha .* (Q - B));
TC = @(Q) Q .* unit_cost(Q);    % Costo Totale con sconto
TC_nodisc = @(Q) Q .* c1;       % Costo Totale senza sconto

std_loss = @(z) normpdf(z) - z.*(1-normcdf(z));
Exp_Sales = @(Q, mu, sigma) mu - sigma .* std_loss((Q-mu)./sigma);
Exp_Leftover = @(Q, mu, sigma) Q - Exp_Sales(Q, mu, sigma);
Exp_Profit = @(Q, mu, sigma) p .* Exp_Sales(Q, mu, sigma) + s .* Exp_Leftover(Q, mu, sigma) - TC(Q);
Exp_Profit_nodisc = @(Q, mu, sigma) p .* Exp_Sales(Q, mu, sigma) + s .* Exp_Leftover(Q, mu, sigma) - TC_nodisc(Q);
CR = (p - c1) / (p - s);

Q_range = 0:1:600;
% Simulazione Monte Carlo 
Past = 20;
M = 10000;
actual_profit_discount = zeros(M,1);
actual_profit_nodisc = zeros(M,1);
actual_profit_teoric = zeros(M,1);


for rep = 1:M
    
    sample = max(0, mu_true + sigma_true * randn(Past,1)); 
    mu_hat = mean(sample);
    sigma_hat = std(sample);
    
    [~, idx_est_disc] = max(Exp_Profit(Q_range, mu_hat, sigma_hat));
    Q_est_disc = Q_range(idx_est_disc); % Ordine deciso CON sconto
    
    [~, idx_est_nodisc] = max(Exp_Profit_nodisc(Q_range, mu_hat, sigma_hat));
    Q_est_nodisc = Q_range(idx_est_nodisc); % Ordine deciso SENZA sconto
    
    %Confronto il quantile ottenuto tramite il metodo numerico e quello
    %teorico per il caso senza sconto
    Q_teoric = norminv(CR, mu_hat, sigma_hat);

    D_real = max(0, mu_true + sigma_true * randn()); 
     
    % Caso 1: Fornitore con Sconto 
    sales_disc = min(Q_est_disc, D_real);        
    leftover_disc = max(0, Q_est_disc - D_real);   
    actual_profit_discount(rep) = (p * sales_disc) - TC(Q_est_disc) + (s * leftover_disc);
    
    % Caso 2: Fornitore senza Sconto 
    sales_nodisc = min(Q_est_nodisc, D_real);
    leftover_nodisc = max(0, Q_est_nodisc - D_real);
    actual_profit_nodisc(rep) = (p * sales_nodisc) - TC_nodisc(Q_est_nodisc) + (s * leftover_nodisc);
    % Caso 3:Formula analitica
    sales_teoric = min(Q_teoric,D_real);
    leftover_teoric = max(0,Q_teoric - D_real);
    actual_profit_teoric(rep) = (p * sales_teoric) - TC_nodisc(Q_teoric) + (s * leftover_teoric);
end

fprintf('Profitto medio CON sconto: %.2f \n', mean(actual_profit_discount));
fprintf('Profitto medio SENZA sconto (numerico): %.2f \n', mean(actual_profit_nodisc));
fprintf('Profitto medio TEORICO: %.2f \n', mean(actual_profit_teoric));

% Plot
figure;
subplot(1,2,1); 
boxplot([actual_profit_nodisc, actual_profit_discount, actual_profit_teoric], ...
        'Labels', {'Senza Sconto','Con Sconto','Teorico'});
ylabel('Profitto Effettivo');
title('Confronto Distribuzione Profitti');
grid on;

diff_teorico_numerico = actual_profit_nodisc - actual_profit_teoric;

subplot(1,2,2);
boxplot(diff_teorico_numerico, 'Labels', {'Differenza Num vs Teor'});
ylabel('Delta Profitto');
title('Residuo: Numerico - Teorico');
grid on;

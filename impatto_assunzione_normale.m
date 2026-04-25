% Quanto sbaglio quando assumo che la domanda sia distribuita normalmente?
clear; clc;
seed = min(353244,362747);
rng(seed);

std_loss = @(z) normpdf(z) - z .* (1 - normcdf(z));

expected_profit_normal = @(Q, mu, sigma, p, c, s) ...
    (p - c) * mu ...
    - (c - s) * sigma .* std_loss((Q - mu) ./ sigma) ...
    - (p - c) * sigma .* std_loss(-(Q - mu) ./ sigma);

profit_from_samples = @(Q, D_samples, p, c, s) ...
    mean(p * min(Q, D_samples) + s * max(0, Q - D_samples) - c * Q);

% Set parametri
p = 30; c = 15; s = 8;
cu = p - c;
co = c - s;
crit_ratio_norm = cu / (cu + co);
z_norm = norminv(crit_ratio_norm);


% Definizione delle distribuzioni vere (media e varianza fissate dove possibile)
mu_true = 200;
sigma_true = 80;      
var_true = sigma_true^2;

% Gamma
k_gamma = mu_true^2 / var_true;
theta_gamma = var_true / mu_true;
distr(1).name = 'Gamma';
distr(1).rnd = @(n) gamrnd(k_gamma, theta_gamma, n, 1);
distr(1).pdf = @(x) gampdf(x, k_gamma, theta_gamma);
distr(1).mean = mu_true;
distr(1).var = var_true;

% Esponenziale
% L'esponenziale ha var = mu^2, qui non riesco ad imporre media e varianza = mu_true e sigma_ture
lambda_exp = 1 / mu_true;
distr(2).name = 'Esponenziale';
distr(2).rnd = @(n) exprnd(mu_true, n, 1);
distr(2).pdf = @(x) exppdf(x, mu_true);
distr(2).mean = mu_true;
distr(2).var = mu_true^2;

% Lognormale (impongo i parametri tali che media = mu_true, var = var_true)
% Se Y ~ N(mu_ln, sigma_ln^2), allora E[exp(Y)] = exp(mu_ln + sigma_ln^2/2) = mu_true
% Var = (exp(sigma_ln^2)-1)*exp(2*mu_ln + sigma_ln^2) = var_true
sigma_ln = sqrt(log(var_true / mu_true^2 + 1));
mu_ln = log(mu_true) - sigma_ln^2 / 2;
distr(3).name = 'Lognormale';
distr(3).rnd = @(n) lognrnd(mu_ln, sigma_ln, n, 1);
distr(3).pdf = @(x) lognpdf(x, mu_ln, sigma_ln);
distr(3).mean = mu_true;
distr(3).var = var_true;

% Uniforme (su [a,b] con media = mu_true, var = var_true)
% a + b = 2*mu_true, (b-a)^2/12 = var_true
half_range = sqrt(3 * var_true);
a = mu_true - half_range;
b = mu_true + half_range;
distr(4).name = 'Uniforme';
distr(4).rnd = @(n) a + (b - a) * rand(n, 1);
distr(4).pdf = @(x) unifpdf(x, a, b);
distr(4).mean = mu_true;
distr(4).var = var_true;

% t di Student scalata per avere media mu_true e var approssimativamente sigma_true^2
% t ha var = nu/(nu-2) per nu>2. Scegliamo nu=5.
nu = 5;
t_var = nu / (nu - 2);
scale_factor = sigma_true / sqrt(t_var);
distr(5).name = 't (nu=5)';
distr(5).rnd = @(n) mu_true + scale_factor * trnd(nu, n, 1);
distr(5).pdf = @(x) tpdf((x - mu_true) / scale_factor, nu) / scale_factor;
distr(5).mean = mu_true;
distr(5).var = var_true; % approssimativamente


%% Simulazione per ciascuna distribuzione con N fissato
N = 30;             % dimensione campionaria per la stima
M_outer = 2000;     % numero di repliche per l'esperimento
N_eval = 50000;     % campioni per la valutazione del profitto atteso

num_distr = length(distr);
ratio_profit = zeros(num_distr, 1);
bias_Q = zeros(num_distr, 1);
Q_opt_true_all = zeros(num_distr, 1);
EP_opt_true_all = zeros(num_distr, 1);
mean_Q_est_all = zeros(num_distr, 1);
mean_profit_est_all = zeros(num_distr, 1);

% Per visualizzare le distribuzioni di Q_est
Q_est_cell = cell(num_distr, 1);
profit_est_cell = cell(num_distr, 1);

for d = 1:num_distr
    fprintf('\nDistribuzione: %s\n', distr(d).name);
    
    % 1) Calcolo la quantità ottima reale per questa distribuzione
    % Uso una griglia di Q e valuto il profitto su questa griglia con un campione grande
    Q_grid = linspace(max(0, mu_true - 3*sigma_true), mu_true + 3*sigma_true, 200);
    
    D_eval = distr(d).rnd(N_eval); % Fisso un campione di domande
    
    profit_grid = zeros(size(Q_grid));
    for i = 1:length(Q_grid)
        Q = Q_grid(i);
        profit_grid(i) = profit_from_samples(Q, D_eval, p, c, s);
    end
    [EP_opt, idx_opt] = max(profit_grid);
    Q_opt = Q_grid(idx_opt);
    Q_opt_true_all(d) = Q_opt;
    EP_opt_true_all(d) = EP_opt;
    fprintf('Quantità ottima reale: %.2f\n', Q_opt);
    fprintf('Profitto atteso ottimo reale: %.2f\n', EP_opt);
    
    % 2) Simulazione dell'approccio "normale"
    Q_est_all = zeros(M_outer, 1);
    profit_est_all = zeros(M_outer, 1);
    
    for sim = 1:M_outer
        % Campione dalla distribuzione vera
        sample = distr(d).rnd(N);
        mu_hat = mean(sample);
        sigma_hat = std(sample);
        
        % Quantità stimata con formula normale
        Q_est = mu_hat + z_norm * sigma_hat;
        Q_est = max(Q_est, 0);
        Q_est_all(sim) = Q_est;
        
        % Profitto atteso vero per Q_est:
        % Interpolo il profitto usando la griglia di valutazioni dei profitti che ho precedentemente calcolato, 
        % dunque prendo il profitto che sta più vicino a Q_est, penso Q_grif come x e profit_grid come y 
        profit_est_all(sim) = interp1(Q_grid, profit_grid, Q_est, 'linear', 'extrap');
    end
    
    mean_Q_est = mean(Q_est_all);
    mean_profit_est = mean(profit_est_all);
    ratio = mean_profit_est / EP_opt;
    bias = mean_Q_est - Q_opt;
    
    % Salvo i risultati risultati nella cell della distribuzione corrente
    Q_est_cell{d} = Q_est_all;
    profit_est_cell{d} = profit_est_all;
    ratio_profit(d) = ratio;
    bias_Q(d) = bias;
    mean_Q_est_all(d) = mean_Q_est;
    mean_profit_est_all(d) = mean_profit_est;
    
    fprintf('Rapporto profitto medio: %.4f (perdita %.2f%%)\n', ratio, (1-ratio)*100);
    fprintf('Bias medio di Q: %.2f (%.2f%% dell''ottimo)\n', bias, bias/Q_opt*100);
end


% Plot
figure;
bar(ratio_profit);
set(gca, 'XTickLabel', {distr.name});
ylabel('Rapporto profitto medio');
title('Performance relativa dell''assunzione normale');
grid on;
ylim([0.8, 1]);  

figure;
hold on;
all_Q_est = [];
group = [];
for d = 1:num_distr
    all_Q_est = [all_Q_est; Q_est_cell{d}];
    group = [group; d * ones(length(Q_est_cell{d}), 1)];
end
boxplot(all_Q_est, group, 'Labels', {distr.name});
for d = 1:num_distr
    line([d-0.3, d+0.3], [Q_opt_true_all(d), Q_opt_true_all(d)], ...
        'Color', 'g', 'LineWidth', 2, 'LineStyle', '--');
end
ylabel('Quantità ordinata');
title('Distribuzione di q_{est} (assunzione normale) vs q^* vera (linea verde tratteggiata)');
grid on;


%% Effetto del coefficiente di variazione (CV)
% Focalizziamoci solo su Gamma e Lognormale
CV_values = 0.1:0.1:0.8;
num_CV = length(CV_values);
ratio_gamma_cv = zeros(num_CV, 1);
ratio_logn_cv = zeros(num_CV, 1);

N = 30;
M_outer = 1000;   
N_eval = 30000;
mu_fixed = 200;

for i = 1:num_CV
    CV = CV_values(i);
    sigma_cur = CV * mu_fixed;
    var_cur = sigma_cur^2;
    
    % Gamma
    k_g = mu_fixed^2 / var_cur;
    theta_g = var_cur / mu_fixed;
    rnd_g = @(n) gamrnd(k_g, theta_g, n, 1);
    D_eval_g = rnd_g(N_eval);
    
    % Griglia per ottimo
    Q_grid = linspace(max(0, mu_fixed - 3*sigma_cur), mu_fixed + 3*sigma_cur, 150);
    profit_grid_g = zeros(size(Q_grid));
    for j = 1:length(Q_grid)
        profit_grid_g(j) = profit_from_samples(Q_grid(j), D_eval_g, p, c, s);
    end
    [EP_opt_g, idx_opt_g] = max(profit_grid_g);
    Q_opt_g = Q_grid(idx_opt_g);
    
    % Simulazione stima normale
    temp_ratio_g = zeros(M_outer, 1);
    for sim = 1:M_outer
        sample = rnd_g(N);
        mu_hat = mean(sample);
        sigma_hat = std(sample);
        Q_est = mu_hat + z_norm * sigma_hat;
        Q_est = max(Q_est, 0);
        prof_est = interp1(Q_grid, profit_grid_g, Q_est, 'linear', 'extrap');
        temp_ratio_g(sim) = prof_est / EP_opt_g;
    end
    ratio_gamma_cv(i) = mean(temp_ratio_g);
    
    % Lognormale
    sigma_ln = sqrt(log(var_cur / mu_fixed^2 + 1));
    mu_ln = log(mu_fixed) - sigma_ln^2 / 2;
    rnd_ln = @(n) lognrnd(mu_ln, sigma_ln, n, 1);
    D_eval_ln = rnd_ln(N_eval);
    
    profit_grid_ln = zeros(size(Q_grid));
    for j = 1:length(Q_grid)
        profit_grid_ln(j) = profit_from_samples(Q_grid(j), D_eval_ln, p, c, s);
    end
    [EP_opt_ln, idx_opt_ln] = max(profit_grid_ln);
    Q_opt_ln = Q_grid(idx_opt_ln);
    
    temp_ratio_ln = zeros(M_outer, 1);
    for sim = 1:M_outer
        sample = rnd_ln(N);
        mu_hat = mean(sample);
        sigma_hat = std(sample);
        Q_est = mu_hat + z_norm * sigma_hat;
        Q_est = max(Q_est, 0);
        prof_est = interp1(Q_grid, profit_grid_ln, Q_est, 'linear','extrap');
        temp_ratio_ln(sim) = prof_est / EP_opt_ln;
    end
    ratio_logn_cv(i) = mean(temp_ratio_ln);
end

% plot
figure;
plot(CV_values, ratio_gamma_cv, 'bo-', 'LineWidth', 1.5, 'DisplayName', 'Gamma');
hold on;
plot(CV_values, ratio_logn_cv, 'rs-', 'LineWidth', 1.5, 'DisplayName', 'Lognormal');
xlabel('Coefficiente di variazione (CV = \sigma/\mu)');
ylabel('Rapporto profitto medio');
title('Impatto della specificazione errata al variare di CV');
legend('Location', 'best');
grid on;



%% Effetto della dimensione campionaria N sull'errata specificazione
% Selezioniamo tre distribuzioni rappresentative:
% - Gamma (asimmetrica a destra)
% - Lognormale (asimmetrica a destra, code più pesanti)
% - Uniforme (simmetrica ma con supporto limitato)
distr_to_plot = [1, 3, 4];  
line_styles = {'b-o', 'r-s', 'g-^'};

N_values = [5, 10, 20, 30, 50, 100, 200, 500];
num_N = length(N_values);

M_outer = 1000;    % repliche per ogni N
N_eval = 30000;    % campioni per valutazione profitto atteso e creazione griglia

% Matrice risultati: righe = distribuzioni, colonne = N
ratio_vs_N = zeros(length(distr_to_plot), num_N);

% Per ogni distribuzione, calcolo l'ottimo vero e la griglia di profitto
for d_idx = 1:length(distr_to_plot)
    d = distr_to_plot(d_idx);
    fprintf('Analisi N per distribuzione: %s\n', distr(d).name);
    
    % Genera campione di domande fisso per questa distribuzione
    D_eval = distr(d).rnd(N_eval);
    
    mu_cur = distr(d).mean;
    sigma_cur = sqrt(distr(d).var);
    Q_grid = linspace(max(0, mu_cur - 3*sigma_cur), mu_cur + 3*sigma_cur, 150);
    
    % Calcola profitto su griglia
    profit_grid = zeros(size(Q_grid));
    for i = 1:length(Q_grid)
        profit_grid(i) = profit_from_samples(Q_grid(i), D_eval, p, c, s);
    end
    [EP_opt, idx_opt] = max(profit_grid);
    Q_opt = Q_grid(idx_opt);
    fprintf('Quantità ottima reale: %.2f, Profitto ottimo: %.2f\n', Q_opt, EP_opt);
    
    % Per ogni N
    for n_idx = 1:num_N
        N = N_values(n_idx);
        temp_ratios = zeros(M_outer, 1);
        
        for sim = 1:M_outer
            % Campione dalla distribuzione vera
            sample = distr(d).rnd(N);
            mu_hat = mean(sample);
            sigma_hat = std(sample);
            
            % Stima normale
            Q_est = mu_hat + z_norm * sigma_hat;
            Q_est = max(Q_est, 0);
            
            % Profitto vero (interpolato)
            prof_est = interp1(Q_grid, profit_grid, Q_est, 'linear', 'extrap');
            temp_ratios(sim) = prof_est / EP_opt;
        end
        
        ratio_vs_N(d_idx, n_idx) = mean(temp_ratios);
        fprintf('N = %3d, Rapporto profitto = %.4f\n', N, ratio_vs_N(d_idx, n_idx));
    end
end

% Plot curve di convergenza
figure;
hold on;
for d_idx = 1:length(distr_to_plot)
    plot(N_values, ratio_vs_N(d_idx, :), line_styles{d_idx}, ...
        'LineWidth', 1.5, 'MarkerSize', 6, ...
        'DisplayName', distr(distr_to_plot(d_idx)).name);
end
xlabel('Dimensione campionaria N');
ylabel('Rapporto profitto medio');
title('Convergenza del rapporto di profitto all''aumentare di N');
legend('Location', 'best');
grid on;
xlim([min(N_values), max(N_values)]);
ylim([0.85, 1.0]);  

yline(1, 'k--', 'LineWidth', 0.8, 'HandleVisibility', 'off');

% Calcolo e visualizzazione del limite asintotico (N -> infinito)
fprintf('\nLimite asintotico (N -> infinito)\n');
colors = lines(length(distr_to_plot));  % matrice di colori
for d_idx = 1:length(distr_to_plot)
    d = distr_to_plot(d_idx);
    mu_cur = distr(d).mean;
    sigma_cur = sqrt(distr(d).var);
    Q_inf = mu_cur + z_norm * sigma_cur;
    % Valutazione profitto atteso per Q_inf (usando la griglia)
    D_eval = distr(d).rnd(N_eval);
    profit_inf = profit_from_samples(Q_inf, D_eval, p, c, s);
    EP_opt = EP_opt_true_all(d);
    ratio_inf = profit_inf / EP_opt;
    fprintf('%s: Q_inf = %.2f, rapporto asintotico = %.4f (perdita %.2f%%)\n', ...
        distr(d).name, Q_inf, ratio_inf, (1-ratio_inf)*100);
end
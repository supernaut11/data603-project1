function [result] = meta_all_random()
    pca_dim = 18:23;
    n_train = 90:120;
    trials = 1:10;
    k = 5;
    
    result = zeros(size(n_train, 2)*size(pca_dim,2), 8);
    idx = 1;
    for p_dim = pca_dim
        for n = n_train
            tmp_result = zeros(size(trials,2), 4);
            for trial = trials
                [tmp_result(trial,1), tmp_result(trial,2), tmp_result(trial,3), tmp_result(trial,4)] = project1(p_dim, n, 1, k);

                
            end
            
            m_result = mean(tmp_result, 1);
            g_cor = m_result(1,1);
            g_wro = m_result(1,2);
            k_cor = m_result(1,3);
            k_wro = m_result(1,4);
            
            g_acc = g_cor / (g_cor + g_wro);
            k_acc = k_cor / (k_cor + k_wro);

            result(idx,:) = [p_dim n g_cor g_wro g_acc k_cor k_wro k_acc];
            idx = idx + 1;
        end
    end
    
    figure;
    hold on; grid;
    for idx = 1:size(pca_dim,2)
        startidx = (idx-1)*size(n_train,2)+1;
        endidx = startidx+size(n_train,2)-1;
        max_acc_g = max(result(startidx:endidx,5));
        max_acc_k = max(result(startidx:endidx,8));
        plot3(result(startidx:endidx,1), result(startidx:endidx,2), result(startidx:endidx,5), 'DisplayName', 'Gaussian', 'color', '#0072BD', 'LineWidth', 3*sqrt(max_acc_g));
        plot3(result(startidx:endidx,1), result(startidx:endidx,2), result(startidx:endidx,8), 'DisplayName', 'k-NN', 'color', '#EDB120', 'LineWidth', 3*sqrt(max_acc_k));
    end
    title('Classifier Accuracy with Varying Parameters');
    xlabel('PCA dimension');
    ylabel('Training Size n');
    zlabel('Model Accuracy');
    view(3);
    legend('Gaussian', 'k-NN', 'Location', 'best');
end
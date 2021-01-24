function [result] = metatrainrandom()
    n_train = 1:199;
    
    pca_dim = 20;
    trials = 100;
    result = zeros(size(n_train, 2), 7);
    for idx = n_train
        tmp_result = zeros(trials, 4);
        for jdx = 1:trials
            [tmp_result(jdx,1), tmp_result(jdx,2), tmp_result(jdx,3), tmp_result(jdx,4)] = project1(pca_dim, idx, 1);
        end
        
        m_result = mean(tmp_result, 1);
        g_cor = m_result(1,1);
        g_wro = m_result(1,2);
        k_cor = m_result(1,3);
        k_wro = m_result(1,4);
        g_acc = g_cor / (g_cor + g_wro);
        k_acc = k_cor / (k_cor + k_wro);
        result(idx,:) = [idx g_cor g_wro g_acc k_cor k_wro k_acc];
    end
    
    figure;
    plot(result(:,1), [result(:,4) result(:,7)]);
    title('Classifier Accuracy with Train/Test Variation');
    xlabel('Training Set Size');
    ylabel('Model Accuracy');
    legend('Gaussian', 'k-NN', 'Location', 'best');
end
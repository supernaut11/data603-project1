function [result] = metatrain()
    n_train = 1:199;
    
    pca_dim = 20;
    result = zeros(size(n_train, 2), 7);
    for idx = n_train
        [g_cor, g_wro, k_cor, k_wro] = project1(pca_dim, idx);
        
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
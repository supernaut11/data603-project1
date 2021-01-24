function [result] = metaknn()
    k=1:2:99;
    
    pca_dim = 20;
    n_train = 100;
    result = zeros(size(n_train, 2), 7);
    for idx = k
        [g_cor, g_wro, k_cor, k_wro] = project1(pca_dim, n_train, 0, idx);
        
        g_acc = g_cor / (g_cor + g_wro);
        k_acc = k_cor / (k_cor + k_wro);
        
        result((idx-1)/2+1,:) = [idx g_cor g_wro g_acc k_cor k_wro k_acc];
    end
    
    figure;
    plot(result(:,1), [result(:,4) result(:,7)]);
    title('Classifier Accuracy with Varying k');
    xlabel('k-NN parameter k');
    ylabel('Model Accuracy');
    legend('Gaussian', 'k-NN', 'Location', 'best');
end
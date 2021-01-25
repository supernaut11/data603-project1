function [result] = metapca()
    pca_dims = 1:200;
    
    result = zeros(size(pca_dims, 2), 7);
    for idx = pca_dims
        [g_cor, g_wro, k_cor, k_wro] = project1(idx);
        
        g_acc = g_cor / (g_cor + g_wro);
        k_acc = k_cor / (k_cor + k_wro);
        
        result(idx,:) = [idx g_cor g_wro g_acc k_cor k_wro k_acc];
    end
    
    figure;
    plot(result(:,1), [result(:,4) result(:,7)]);
    title('Classifier Accuracy with PCA Variation');
    xlabel('PCA Dimension');
    ylabel('Model Accuracy');
    legend('Gaussian', 'k-NN', 'Location', 'best');
end
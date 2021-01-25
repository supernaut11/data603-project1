function [result] = metapca_p2()
    num_train = 10;
    pca_dims = 10:65;
    
    result = zeros(size(pca_dims, 2), 4);
    for idx = pca_dims
        [g_cor, g_wro] = project1_part2(idx, num_train);
        
        g_acc = g_cor / (g_cor + g_wro);
        
        result(idx,:) = [idx g_cor g_wro g_acc];
    end
    
    figure;
    plot(result(:,1), result(:,4));
    title('Classifier Accuracy with PCA Variation');
    xlabel('PCA Dimension');
    ylabel('Model Accuracy');
    legend('Gaussian', 'k-NN', 'Location', 'best');
end
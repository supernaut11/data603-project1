function [result] = metatrain_p2()
    pca_dim = 10;
    n_train = 5:10;
    
    result = zeros(size(n_train, 2), 4);
    for idx = n_train
        [g_cor, g_wro] = project1_part2(pca_dim, idx);
        
        g_acc = g_cor / (g_cor + g_wro);
        
        result(idx-4,:) = [idx g_cor g_wro g_acc];
    end
    
    figure;
    plot(result(:,1), result(:,4));
    title('Classifier Accuracy with Train/Test Variation');
    xlabel('Training Set Size');
    ylabel('Model Accuracy');
    legend('Gaussian', 'Location', 'best');
end
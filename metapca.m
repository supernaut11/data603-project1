function [result] = metapca()
    pca_dims = 1:5:400;
    
    result = zeros(size(pca_dims, 2), 7);
    for idx = 1:200
        [g_cor, g_wro, k_cor, k_wro] = project1(idx);
        
        g_acc = g_cor / (g_cor + g_wro);
        k_acc = k_cor / (k_cor + k_wro);
        
        result(idx,:) = [idx g_cor g_wro g_acc k_cor k_wro k_acc];
    end
end
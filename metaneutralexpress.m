function [result] = metaneutralexpress()
    pca_dim = 10:30;
    n_train = 80:120;
    k = 5;
    
    result = zeros(size(n_train, 2)*size(pca_dim,2), 8);
    idx = 1;
    for p_dim = pca_dim
        for n = n_train
            [g_cor, g_wro, k_cor, k_wro] = project1(p_dim, n, 0, k);

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
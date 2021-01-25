function [result] = metaknn_p2()
%     pca_dim = 100;
%     n_train = 10;
%     k = 5;
    
    pca_dim = 70:150;
    n_train = 5:10;
    k = 5;
    
    result = zeros(size(pca_dim,2)*size(n_train, 2), 5);
    idx = 1;
    for p_dim = pca_dim
        for n = n_train
            [k_cor, k_wro] = project1_part2_knn(p_dim, n, k);

            k_acc = k_cor / (k_cor + k_wro);

            result(idx,:) = [p_dim n k_cor k_wro k_acc];
            idx = idx + 1;
        end
    end
    
    figure;
    hold on; grid;
    for idx = 1:size(pca_dim,2)
        startidx = (idx-1)*size(n_train,2)+1;
        endidx = startidx+size(n_train,2)-1;
        plot3(result(startidx:endidx,1), result(startidx:endidx,2), result(startidx:endidx,5), 'DisplayName', 'k-NN', 'color', '#EDB120');
    end
    title('Classifier Accuracy with Varying Parameters');
    xlabel('PCA dimension');
    ylabel('Training Size n');
    zlabel('Model Accuracy');
    view(3);
    legend('Gaussian', 'k-NN', 'Location', 'best');
end
function [Y_mda] = mymda(X_all, Xi)
    [dimension,num_poses,num_subjects] = size(Xi);

    % Compute the mean for each subject
    class_mean = zeros(dimension,1,num_subjects);
    for idx = 1:num_subjects
        class_mean(:,1,idx) = mean(Xi(:,:,idx),2);
    end

    % figure;
    % colormap gray;
    % imagesc(reshape(Xi_mu(:,:,1), [d1 d2]));

    class_centered = zeros(dimension,num_poses,num_subjects);
    for idx = 1:num_subjects
        class_centered(:,:,idx) = Xi(:,:,idx) - class_mean(:,:,idx);
    end

    class_scatter = zeros(dimension,dimension,num_subjects);
    % Compute the scatter matrix for each subject
    for idx = 1:num_subjects
        curX = class_centered(:,:,idx);
        class_scatter(:,:,idx) = curX*curX';
    end

    Sw = sum(class_scatter,3);
    
    total_mean = mean(X_all,2);
    total_centered = X_all-total_mean;
    Stotal = total_centered*total_centered';
    Sb = Stotal - Sw;

    % Solve generalized eigenvalue problem to come up with MDA projection
    [evec, eval] = eig(Sb, Sw);
    [~, isort] = sort(diag(eval), 'descend');
    evec = evec(:,isort);
    w = evec(:,1:num_subjects-1);
    Y_mda = w'*X_all;
end
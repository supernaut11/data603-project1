function [w] = mymda(Xi)
    [~,num_poses,num_subjects] = size(Xi);

    % Compute the mean for each subject
    Xi_mu = [];
    for idx = 1:num_subjects
        Xi_mu(:,1,idx) = mean(Xi(:,:,idx),2);
    end

    % figure;
    % colormap gray;
    % imagesc(reshape(Xi_mu(:,:,1), [d1 d2]));

    Xi_centered = [];
    for idx = 1:num_subjects
        Xi_centered(:,:,idx) = Xi(:,:,idx) - Xi_mu(:,:,idx);
    end

    Sw_i = [];
    % Compute the scatter matrix for each subject
    for idx = 1:num_subjects
        curX = Xi_centered(:,:,idx);
        Sw_i(:,:,idx) = curX*curX';
    end

    figure;
    imagesc(Sw_i(:,:,1));
    colorbar;

    Sw = sum(Sw_i,3);

    figure;
    imagesc(Sw(:,:,1));
    colorbar;

    %Sb_i = zeros(d1*d2, d1*d2, num_subjects);
    Sb_i = [];
    total_mean = sum(num_poses*Xi_mu,3)/(num_poses*num_subjects);

    for idx = 1:num_subjects
        m_i = Xi_mu(:,:,idx);

        Sb_i(:,:,idx) = num_poses*(m_i-total_mean)*(m_i-total_mean)';
    end

    Sb = sum(Sb_i,3);

    %% Solve generalized eigenvalue problem to come up with MDA projection
    [evec, eval] = eig(Sb, Sw);
    [esort, isort] = sort(diag(eval), 'descend');
    evec = evec(:,isort);
    w = evec(:,1:num_subjects-1);
end
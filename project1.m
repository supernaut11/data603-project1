function [gauss_correct, gauss_wrong, knn_correct, knn_wrong] = project1(varargin)
    data = load('data.mat');

    face = data.face;
    face_neutral = face(:,:,1:3:end);
    face_express = face(:,:,2:3:end);
    face_illumin = face(:,:,3:3:end);

    [d1,d2,n] = size(face_neutral);

    if nargin > 0
        pca_dimensions = varargin{1};
    else
        pca_dimensions = 20;
    end

    if nargin > 1
        n_train = varargin{2};
        n_test = n - n_train;
    else
        n_train = 100;
        n_test = n - n_train;
    end

    X_neutral = reshape(face_neutral, [d1*d2 n]);
    if nargin > 2 && varargin{3} == 1
        X_neutral = X_neutral(:,randperm(n));
    end
    X_express = reshape(face_express, [d1*d2 n]);
    if nargin > 2 && varargin{3} == 1
        X_express = X_express(:,randperm(n));
    end
    X_illumin = reshape(face_illumin, [d1*d2 n]);

    X = [X_neutral X_express];

%     figure;
%     colormap gray;
%     subplot(1,2,1);
%     imagesc(reshape(mean(X_neutral,2), [24 21]));
% 
%     subplot(1,2,2);
%     imagesc(reshape(mean(X_express,2), [24 21]));

    %% Perform PCA on the training data set to reduce dimensionality
    % SVD solution

    [Yc, Uc] = mypca(X, pca_dimensions);

    % built_in = pca(X');
    % Yc = X'*built_in(:,1:num_dimensions);
    % Yc = Yc';

    % figure;
    % hold on; grid;
    % plot(Yc(1,1:200),Yc(2,1:200),'.','Markersize',20, 'color', 'k');
    % plot(Yc(1,201:400),Yc(2,201:400),'.','Markersize',20, 'color', 'r');
    % title('PCA for data.mat, p=2');
    % legend('Neutral', 'Expression');
    % 
    % figure;
    % hold on; grid;
    % plot3(Yc(1,1:200),Yc(2,1:200),Yc(3,1:200),'.','Markersize',20, 'color', 'k');
    % plot3(Yc(1,201:400),Yc(2,201:400),Yc(3,201:400),'.','Markersize',20, 'color', 'r');
    % %plot3(Yc(401:600,1),Yc(401:600,2),Yc(401:600,3),'.','Markersize',20, 'color', 'b');
    % view(3);
    % title('PCA for data.mat, p=3');
    % legend('Neutral', 'Expression');

    %% Perform Bayes classification

    % Get maximum likelihood estimate for Gaussian parameters. In this case,
    % it will just be the sample covariance and the sample mean for each class
    [Y_neutral_train, Y_neutral_test] = serialsplit(Yc(:,1:n), n_train);
    [Y_express_train, Y_express_test] = serialsplit(Yc(:,n+1:2*n), n_train);

    Yc_train = [Y_neutral_train Y_express_train];
    Yc_test = [Y_neutral_test Y_express_test];

    [Ym_neutral, Ycov_neutral] = gaussianparams(Y_neutral_train, n_train);
    [Ym_express, Ycov_express] = gaussianparams(Y_express_train, n_train);

    % figure;
    % subplot(1,2,1);
    % imagesc(Ycov_neutral);
    % colorbar;
    % subplot(1,2,2);
    % imagesc(Ycov_express);
    % colorbar;

    % Define a discriminant function that will leverage the above parameters
    % and classify test points
    neutral_dist = buildgaussian(Ym_neutral, Ycov_neutral);
    express_dist = buildgaussian(Ym_express, Ycov_express);

    gauss_classifier = @(x_d)neutral_dist(x_d) - express_dist(x_d);

    %% Classify test data (illumination set) using Gaussian classifier

    gauss_correct = 0;
    gauss_wrong = 0;
    
    neutral_gauss_result = zeros(1,n_test);

    for idx = 1:n_test
        neutral_gauss_result(1,idx) = sign(gauss_classifier(Y_neutral_test(:,idx)));
    end

    num_neutral = length(find(neutral_gauss_result > 0));
    num_express = length(find(neutral_gauss_result < 0));

    gauss_correct = gauss_correct + num_neutral;
    gauss_wrong = gauss_wrong + num_express;
    
    fprintf("neutral test points\n");
    fprintf("neutral count = %d\nexpress count = %d\n", num_neutral, num_express);

    express_gauss_result = zeros(1,n_test);

    for idx = 1:n_test
        express_gauss_result(1,idx) = sign(gauss_classifier(Y_express_test(:,idx)));
    end

    num_neutral = length(find(express_gauss_result > 0));
    num_express = length(find(express_gauss_result < 0));

    gauss_correct = gauss_correct + num_express;
    gauss_wrong = gauss_wrong + num_neutral;
    
    fprintf("express test points\n");
    fprintf("neutral count = %d\nexpress count = %d\n", num_neutral, num_express);

    % figure;
    % hold off;
    % predicted = discretize([ones(1,100) -1*ones(1,100)], [-1 0 1], 'categorical', {'expression', 'neutral'});
    % actual = discretize([sign(neutral_gauss_result) sign(express_gauss_result)], [-1 0 1], 'categorical', {'expression', 'neutral'});
    % confusionchart(predicted, actual);

    %% Try KNN
    if nargin > 3
        k = varargin{4};
    else
        k = 5;
    end
    
    neutral_correct = 0;
    neutral_wrong = 0;
    labels = [zeros(1,n_train) ones(1,n_train)];
    for idx = 1:n_test
        [classification] = myknn(k, Yc_train, Y_neutral_test(:,idx), labels);

        if classification == 0
            neutral_correct = neutral_correct + 1;
        else
            neutral_wrong = neutral_wrong + 1;
        end
    end

    express_correct = 0;
    express_wrong = 0;
    for idx = 1:n_test
        [classification] = myknn(k, Yc_train, Y_express_test(:,idx), labels);

        if classification == 1
            express_correct = express_correct + 1;
        else
            express_wrong = express_wrong + 1;
        end
    end

    knn_correct = neutral_correct + express_correct;
    knn_wrong = neutral_wrong + express_wrong;
    
%     figure;
%     hold off;
%     predicted = discretize([ones(1,100) -1*ones(1,100)], [-1 0 1], 'categorical', {'expression', 'neutral'});
%     actual = discretize([-1*ones(1,neutral_wrong) ones(1,neutral_correct) -1*ones(1,express_correct) ones(1,express_wrong)], [-1 0 1], 'categorical', {'expression', 'neutral'});
%     confusionchart(predicted, actual);

    %% Classify test data using KNN classifier

%     max_correct = 0;
%     best_k = 0;
%     for k = 1:2:21
%         labels = [zeros(1,n_train) ones(1,n_train)];
% 
%         neutral_correct = 0;
%         neutral_wrong = 0;
%         for idx = 1:n_test
%             [classification] = myknn(k, Yc_train, Y_neutral_test(:,idx), labels);
% 
%             if classification == 0
%                 neutral_correct = neutral_correct + 1;
%             else
%                 neutral_wrong = neutral_wrong + 1;
%             end
%         end
% 
%         fprintf("k = %d\n", k);
%         fprintf("neutral correct = %d\nneutral wrong = %d\n", neutral_correct, neutral_wrong);
% 
%         express_correct = 0;
%         express_wrong = 0;
%         for idx = 1:n_test
%             [classification] = myknn(k, Yc_train, Y_express_test(:,idx), labels);
% 
%             if classification == 1
%                 express_correct = express_correct + 1;
%             else
%                 express_wrong = express_wrong + 1;
%             end
%         end
% 
%         fprintf("k = %d\n", k);
%         fprintf("express correct = %d\nexpress wrong = %d\n", express_correct, express_wrong);
% 
%         total_correct = neutral_correct + express_correct;
%         if total_correct > max_correct
%             max_correct = total_correct;
%             best_k = k;
%         end
%     end
% 
%     fprintf("best k = %d, correct = %d\n", best_k, max_correct);

end

%% Finished section 1

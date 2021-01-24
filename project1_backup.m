data = load('data.mat');

face = data.face;
face_neutral = face(:,:,1:3:end);
face_express = face(:,:,2:3:end);
face_illumin = face(:,:,3:3:end);

[d1,d2,n] = size(face_neutral);

X_neutral = zeros(n,d1*d2);
X_express = zeros(n,d1*d2);
X_illumin = zeros(n,d1*d2);

for idx = 1:n
    aux = face_neutral(:,:,idx);
    X_neutral(idx,:) = aux(:)';
    aux = face_express(:,:,idx);
    X_express(idx,:) = aux(:)';
    aux = face_illumin(:,:,idx);
    X_illumin(idx,:) = aux(:)';
end

X = [X_neutral; X_express];

% X = [X_neutral; X_express; X_illumin];

% face_train = [face_neutral; face_express];
% mean_train = mean(face_train);
% face_train = face_train - ones(400,1)*mean_train;

%% Perform PCA on the training data set to reduce dimensionality
% SVD solution

% Center all of the data
Xm = mean(X, 1);
Xc = X - ones(n*2,1)*Xm;

% Solve for the eigenvalues of the scatter matrix of the centered data
[Uc,Sigmac,Vc] = svd(Xc', 'econ');

esort = diag(Sigmac);

figure;
plot(esort,'.','Markersize',20);
grid;

num_dimensions = 20;
Yc = X*Uc(:,1:num_dimensions);

figure;
hold on; grid;
plot(Yc(1:200,1),Yc(1:200,2),'.','Markersize',20, 'color', 'k');
plot(Yc(201:400,1),Yc(201:400,2),'.','Markersize',20, 'color', 'r');
set(gca,'Fontsize',20);

figure;
hold on; grid;
plot3(Yc(1:200,1),Yc(1:200,2),Yc(1:200,3),'.','Markersize',20, 'color', 'k');
plot3(Yc(201:400,1),Yc(201:400,2),Yc(201:400,3),'.','Markersize',20, 'color', 'r');
%plot3(Yc(401:600,1),Yc(401:600,2),Yc(401:600,3),'.','Markersize',20, 'color', 'b');
view(3);

% %% Built-in Matlab solution
% built_in = pca(X);
% 
% Y_b = X*built_in(:,1:num_dimensions);
% 
% figure;
% hold on; grid;
% plot(Y_b(1:200,1),Y_b(1:200,2),'.','Markersize',20, 'color', 'k');
% plot(Y_b(201:400,1),Y_b(201:400,2),'.','Markersize',20, 'color', 'r');
% %colorbar
% %caxis([0,pi]);
% set(gca,'Fontsize',20);
% daspect([1,1,1])
% 
% figure;
% hold on; grid;
% plot3(Y_b(1:200,1),Y_b(1:200,2),Y_b(1:200,3),'.','Markersize',20, 'color', 'k');
% plot3(Y_b(201:400,1),Y_b(201:400,2),Y_b(201:400,3),'.','Markersize',20, 'color', 'r');
% %colorbar
% %caxis([0,pi]);
% view(3);
% %set(gca,'Fontsize',20);
% %daspect([1,1,1])

%% Perform Bayes classification

% Get maximum likelihood estimate for Gaussian parameters. In this case,
% it will just be the sample covariance and the sample mean for each class

n_train = 125;
n_test = n - n_train;

Y_neutral_train = Yc(1:n_train,:);
Y_neutral_test = Yc(n_train+1:n,:);
Y_express_train = Yc(n+1:n+n_train,:);
Y_express_test = Yc(n+n_train+1:n*2,:);

Yc_train = [Y_neutral_train; Y_express_train];
Yc_test = [Y_neutral_test; Y_express_test];

Ym_neutral = mean(Y_neutral_train, 1);
Ym_express = mean(Y_express_train, 1);

Y_neutral_c = Y_neutral_train - ones(n_train,1)*Ym_neutral;
Y_express_c = Y_express_train - ones(n_train,1)*Ym_express;

Ycov_neutral = (1/(n_train-1))*(Y_neutral_c'*Y_neutral_c);
Ycov_express = (1/(n_train-1))*(Y_express_c'*Y_express_c);

figure;
imagesc(Ycov_neutral);
colorbar;

figure;
imagesc(Ycov_express);
colorbar;

% Define a discriminant function that will leverage the above parameters
% and classify test points

iYcov_neutral = inv(Ycov_neutral);
iYcov_express = inv(Ycov_express);

neutral_dist = @(x_d)-0.5*log(det(Ycov_neutral)) - 0.5*((x_d - Ym_neutral')'*iYcov_neutral*(x_d - Ym_neutral'));
express_dist = @(x_d)-0.5*log(det(Ycov_express)) - 0.5*((x_d - Ym_express')'*iYcov_express*(x_d - Ym_express'));

gauss_classifier = @(x_d)neutral_dist(x_d) - express_dist(x_d);

%% Classify test data (illumination set) using Gaussian classifier

%Y_neutral_test = X_illumin*Uc(:,1:num_dimensions);
%Y_express_test = Y_express(n_train+1:end);

neutral_gauss_result = zeros(1,n_test);

for idx = 1:n_test
    neutral_gauss_result(1,idx) = sign(gauss_classifier(Y_neutral_test(idx,:)'));
end

num_neutral = length(find(neutral_gauss_result > 0));
num_express = length(find(neutral_gauss_result < 0));

fprintf("neutral test points\n");
fprintf("neutral count = %d\nexpress count = %d\n", num_neutral, num_express);

express_gauss_result = zeros(1,n_test);

for idx = 1:n_test
    express_gauss_result(1,idx) = sign(gauss_classifier(Y_express_test(idx,:)'));
end

num_neutral = length(find(express_gauss_result > 0));
num_express = length(find(express_gauss_result < 0));

fprintf("express test points\n");
fprintf("neutral count = %d\nexpress count = %d\n", num_neutral, num_express);

%% Classify test data using KNN classifier

max_correct = 0;
best_k = 0;
for k = 1:2:21
    neutral_knn_result = zeros(1,n_test);

    for idx = 1:n_test
        cur_distances = vecnorm(Yc_train' - Y_neutral_test(idx,:)', 2);

        [B,I] = mink(cur_distances, k);
        knn_neutral_votes = length(find(I < n_train+1));
        knn_express_votes = k - knn_neutral_votes;

        neutral_knn_result(1,idx) = sign(knn_neutral_votes - knn_express_votes);
    end

    num_neutral = length(find(neutral_knn_result > 0));
    num_express = length(find(neutral_knn_result < 0));
    neutral_correct = num_neutral;
    fprintf("k = %d\n", k);
    fprintf("neutral count = %d\nexpress count = %d\n", num_neutral, num_express);
    
    express_knn_result = zeros(1,n_test);

    for idx = 1:n_test
        cur_distances = vecnorm(Yc_train' - Y_express_test(idx,:)', 2);

        [B,I] = mink(cur_distances, k);
        knn_neutral_votes = length(find(I < n_train+1));
        knn_express_votes = k - knn_neutral_votes;

        express_knn_result(1,idx) = sign(knn_neutral_votes - knn_express_votes);
    end

    num_neutral = length(find(express_knn_result > 0));
    num_express = length(find(express_knn_result < 0));
    fprintf("k = %d\n", k);
    fprintf("neutral count = %d\nexpress count = %d\n", num_neutral, num_express);
    
    express_correct = num_express;
    
    total_correct = neutral_correct + express_correct;
    if total_correct > max_correct
        max_correct = total_correct;
        best_k = k;
    end
end

fprintf("best k = %d, correct = %d\n", best_k, max_correct);

%% Finished section 1

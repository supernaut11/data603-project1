function [g_correct, g_wrong] = project1_part2(num_pca_dimensions, num_train)
%% Load data
% Load the data set
data = load('pose.mat');
pose = data.pose;

num_subjects = 5;
pca_pose = pose(:,:,:,1:num_subjects);

[d1,d2,num_poses,num_subjects] = size(pca_pose);

X_all = [];
for idx = 1:num_subjects
    for jdx = 1:num_poses
        X_all(:,(idx-1)*num_poses+jdx) = reshape(pca_pose(:,:,jdx,idx), [d1*d2 1]);
    end
end

% figure;
% colormap gray;
% 
% for idx = 1:num_poses
%     subplot(3,5,idx);
%     imagesc(reshape(X_all(:,idx), [d1 d2]));
% end

%% Perform PCA
[Y_pca, ~] = mypca(X_all, num_pca_dimensions);

%% Perform MDA
[~,~,num_poses,num_subjects] = size(pca_pose);

% Create a matrix that contains image data in each column
Yi_pca = [];
for idx = 1:num_subjects
    for jdx = 1:num_poses
        Yi_pca(:,jdx,idx) = Y_pca(:,(idx-1)*num_poses+jdx);
    end
end

Y_mda = mymda(Y_pca, Yi_pca);

% Split up PCA+MDA data by class to make Gaussian parameter estimation
% easier
Yi_mda = [];
for idx = 1:num_subjects
    cur_start = (idx-1)*num_poses;
    for jdx = 1:num_poses
        Yi_mda(:,jdx,idx) = Y_mda(:,cur_start+jdx);
    end
end

%% Split data into training and test sets
num_test = num_poses - num_train;
Yi_mda_train = [];
Yi_mda_test = [];
for idx = 1:num_subjects
    [Yi_mda_train(:,:,idx), Yi_mda_test(:,:,idx)] = serialsplit(Yi_mda(:,:,idx), num_train);
end

%% Estimate gaussian parameters using the PCA+MDA processed data

% Build a discriminant function for each subject
gauss_dists = cell(num_subjects,1);

for idx = 1:num_subjects
    [mu, C] = gaussianparams(Yi_mda_train(:,:,idx), num_poses);
    gauss_dists{idx} = buildgaussian(mu, C);
end

%% Test the accuracy of the Gaussian classifier
g_correct = 0;
g_wrong = 0;

for idx = 1:num_subjects
    num_class_correct = 0;
    for jdx = 1:num_test
        cur_test = Yi_mda_test(:,jdx,idx);
        gauss_results = zeros(1,num_subjects);
        for kdx = 1:num_subjects
            gauss_results(kdx) = gauss_dists{kdx}(cur_test);
        end
        
        [M,I] = max(gauss_results);
        
        if I(1) == idx
            g_correct = g_correct + 1;
            num_class_correct = num_class_correct + 1;
        else
            g_wrong = g_wrong + 1;
        end
    end
    
    fprintf("CLASS %d RESULTS\n", idx);
    fprintf("================\n");
    fprintf("classified %d correctly\n", num_class_correct);
end

fprintf("TOTAL RESULTS\n");
fprintf("=============\n");
fprintf("classified %d correctly\nclassified %d incorrectly\n", g_correct, g_wrong);

%% Perform KNN classification

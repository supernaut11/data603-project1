%% Load data
% Load the data set
data = load('pose.mat');
pose = data.pose;

[d1,d2,num_poses,num_subjects] = size(pose);

X_all = [];
for idx = 1:num_subjects
    for jdx = 1:num_poses
        X_all(:,(idx-1)*num_poses+jdx) = reshape(pose(:,:,jdx,idx), [d1*d2 1]);
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
num_dimensions = 100;
[Y_pca, Uc] = mypca(X_all, num_dimensions);

%% Perform MDA
% Create a matrix that contains image data in each column
Yi_pca = [];
for idx = 1:num_subjects
    for jdx = 1:num_poses
        Yi_pca(:,jdx,idx) = Y_pca(:,(idx-1)*num_poses+jdx);
    end
end

Y_mda = mymda(Y_pca, Yi_pca);

% Split up PCA+MDA data by class to make KNN easier
Yi_mda = [];
for idx = 1:num_subjects
    cur_start = (idx-1)*num_poses;
    for jdx = 1:num_poses
        Yi_mda(:,jdx,idx) = Y_mda(:,cur_start+jdx);
    end
end

%% Split data into training and test sets
num_train = 10;
num_test = num_poses - num_train;
Yi_mda_train = [];
Yi_mda_test = [];
for idx = 1:num_subjects
    [Yi_mda_train(:,:,idx), Yi_mda_test(:,:,idx)] = serialsplit(Yi_mda(:,:,idx), num_train);
end

Y_mda_train = [];
for idx = 1:num_subjects
    for jdx = 1:num_train
        Y_mda_train(:,(idx-1)*num_train+jdx) = Yi_mda_train(:,jdx,idx);
    end
end

%% Perform KNN classification

labels = zeros(1, num_subjects*num_train);
for idx = 1:size(labels,2)
    labels(1,idx) = floor((idx-1) / num_train) + 1;
end

k = 5;
num_total_correct = 0;
num_total_wrong = 0;
for idx = 1:num_subjects
    num_class_correct = 0;
    num_class_wrong = 0;
    for jdx = 1:num_test
        [classification] = myknn(k, Y_mda_train, Yi_mda_test(:,jdx,idx), labels);
        
        if classification == idx
            num_class_correct = num_class_correct + 1;
            num_total_correct = num_total_correct + 1;
        else
            num_class_wrong = num_class_wrong + 1;
            num_total_wrong = num_total_wrong + 1;
        end
    end
    
    fprintf("CLASS %d RESULTS\n", idx);
    fprintf("================\n");
    fprintf("classified %d correctly\n", num_class_correct);
    fprintf("classified %d incorrectly\n", num_class_wrong);
end

fprintf("TOTAL RESULTS\n");
fprintf("=============\n");
fprintf("classified %d correctly\n", num_total_correct);
fprintf("classified %d incorrectly\n", num_total_wrong);

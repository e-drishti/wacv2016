
function predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats)
categories = unique(train_labels); 
num_categories = length(categories);

% Lambda value for the error in SVM
lambda = 0.001;
% Cumulative matrices of the 1-vs-all decision boundaries
w_cum = zeros(size(train_image_feats,2),num_categories);
b_cum = zeros(1,num_categories);

% Train the 1-vs-all SVM classifiers
for i=1:num_categories
    cat = categories{i};
    classes = strcmp(cat,train_labels);
    classes = 2*(classes -0.5);
    [w,b] = vl_svmtrain(train_image_feats',classes,lambda);
    w_cum(:,i) = w;
    b_cum(:,i) = b;
end

% Get the confidence value of test images from decision boundary
conf = test_image_feats * w_cum;

for i=1:size(conf,1)
    conf(i,:) = conf(i,:)+b_cum;
end

% Assign test image to the class with max confidence 
[v,i] = max(conf');
predicted_categories = categories(i);







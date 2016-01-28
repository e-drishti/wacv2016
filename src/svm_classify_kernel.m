function predicted_categories = svm_classify_kernel(train_image_feats, train_labels, test_image_feats)
categories = unique(train_labels); 
num_categories = length(categories);

% Lambda value for SVM
lambda = 0.0005;
% The scaling factor for the kernel 
N = 28;

% Project the train images to higher dimension using CHI-sqr kernel
dataset = vl_homkermap(train_image_feats',N,'kernel','kinters');
% Cumulative matrices to store all decision surfaces
w_cum = zeros(size(dataset,1),num_categories);
b_cum = zeros(1,num_categories);
%dataset = train_image_feats;
size(dataset)
% Training the 1-vs-all classifiers
for i=1:num_categories
    cat = categories{i};
    classes = strcmp(cat,train_labels);
    classes = 2*(classes -0.5); % class labels (-1,1)
    [w,b] = vl_svmtrain(dataset,classes,lambda);
    w_cum(:,i) = w;
    b_cum(:,i) = b;
end

% Project the testing images to the same space as training
testset = vl_homkermap(test_image_feats',N,'kernel','kinters');
size(testset)
size(w_cum)

% Get confidence values for each class
conf = testset' * w_cum;

for i=1:size(conf,1)
    conf(i,:) = conf(i,:)+b_cum;
end

% Assign test image to the class with highest confidence
[v,i] = max(conf');
predicted_categories = categories(i);







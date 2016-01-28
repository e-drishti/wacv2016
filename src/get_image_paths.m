% Starter code prepared by James Hays for CS 143, Brown University

%This function returns cell arrays containing the file path for each train
%and test image, as well as cell arrays with the label of each train and
%test image. By default all four of these arrays will be 1500x1 where each
%entry is a char array (or string).
function [train_image_paths, train_labels] = ... 
    get_image_paths(data_path, categories, num_train_per_cat)

num_categories = length(categories); %number of scene categories.
number_train = containers.Map('KeyType','char','ValueType','int32');
number_test = containers.Map('KeyType','char','ValueType','int32');
number_train('1') = 400;
number_train('2') = 2245;
%number_train('2') = 0;
number_train('3') = 1763;

number_test('1') = 80;
number_test('2') = 1000;
number_test('3') = 900;

total_train = number_train('1')+number_train('2')+number_train('3');
total_test = number_test('1')+number_test('2')+number_test('3');

%This paths for each training and test image. By default it will have 1500
%entries (15 categories * 100 training and test examples each)
train_image_paths = cell(total_train, 1);
%test_image_paths  = cell(num_categories * total_test, 1);

%The name of the category for each training and test image. With the
%default setup, these arrays will actually be the same, but they are built
%independently for clarity and ease of modification.
train_labels = zeros(total_train, 1);
%test_labels  = cell(num_categories * total_test, 1);

k=1;
for i=1:num_categories
   images = dir( fullfile(data_path, 'train', categories{i}, '*.jpg'));
   num = number_train(categories{i});
   for j=1:num
       train_image_paths{k} = fullfile(data_path, 'train', categories{i}, images(j).name);
       %fprintf('%s\n',images(j).name);
			 train_labels(k) = str2double(categories{i});
       k = k+1;
   end
end
%for i=1:num_categories
%   num = number_test(categories{i});
%   images = dir( fullfile(data_path, 'test', categories{i}, '*.jpg'));
%   for j=1:num
%       test_image_paths{(i-1)*total_test + j} = fullfile(data_path, 'test', categories{i}, images(j).name);
%       test_labels{(i-1)*total_test + j} = categories{i};
%   end


end




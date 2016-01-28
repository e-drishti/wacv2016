addpath(genpath('/home/aditya/edhristi/code/mkl'));
% FEATURE = 'gabor';
FEATURE = 'hog';

% CLASSIFIER = 'support vector machine';
% CLASSIFIER = 'support vector machine kernel';
CLASSIFIER = 'support vector machine mkl';

% set up paths to VLFeat functions. 
% See http://www.vlfeat.org/matlab/matlab.html for VLFeat Matlab documentation
% This should work on 32 and 64 bit versions of Windows, MacOS, and Linux
vl_setup;
%run('./vlfeat/toolbox/vl_setup') % If required, modify the 'vlfeat' in the path with the specific version number you have downloaded. Please see your downloaded folder and modify this appropriately

data_path = '../data/'; %change if you want to work with a network copy

%This is the list of categories / directories to use. The categories are
%somewhat sorted by similarity so that the confusion matrix looks more
%structured (indoor and then urban and then rural).
% categories = {'Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office', ...
%        'Industrial', 'Suburb', 'InsideCity', 'TallBuilding', 'Street', ...
%        'Highway', 'OpenCountry', 'Coast', 'Mountain', 'Forest'};
categories = {'1', '2', '3'};
   
%This list of shortened category names is used later for visualization.
% abbr_categories = {'Kit', 'Sto', 'Bed', 'Liv', 'Off', 'Ind', 'Sub', ...
%     'Cty', 'Bld', 'St', 'HW', 'OC', 'Cst', 'Mnt', 'For'};

abbr_categories = {'1', '2', '3'};
    
%number of training examples per category to use. Max is 100. For
%simplicity, we assume this is the number of test cases per category, as
%well.
num_train_per_cat = 45; 

%This function returns cell arrays containing the file path for each train
%and test image, as well as cell arrays with the label of each train and
%test image. By default all four of these arrays will be 1500x1 where each
%entry is a char array (or string).
fprintf('Getting paths and labels for all train and test data\n')
[train_image_paths, train_labels] = ...
    get_image_paths(data_path, categories, num_train_per_cat);          

%exit(1);
fprintf('Using %s representation for images\n', FEATURE)

switch lower(FEATURE)    
        
    case 'gabor'
        if ~exist('train_feats_gabor.mat','file')
            train_image_feats = get_gabor_features(train_image_paths);
            save('train_feats_gabor.mat','train_image_feats')
        else
            load('train_feats_gabor.mat');
        end
     
		case 'lbp'
        if ~exist('train_feats_lbp.mat','file')
            train_image_feats = get_lbp_features(train_image_paths);
            save('train_feats_lbp.mat','train_image_feats')
        else
            load('train_feats_lbp.mat');
        end
   
	 case 'hog'
        if ~exist('train_feats_hog.mat','file')
            train_image_feats = get_hog_features(train_image_paths);
            save('train_feats_hog.mat','train_image_feats')
        else
            load('train_feats_hog.mat');
        end
	
	%	case 'intraface'
  %      if ~exist('train_feats_intraface.mat','file')
  %          train_image_feats = csvread('/home/aditya/honours/demo/feat.csv');
	%					save('train_feats_intraface.mat','train_image_feats')
  %      else
  %          load('train_feats_intraface.mat');
	%			end
				%train_labels = csvread('/home/aditya/honours/demo/labels.csv');
		
		case 'intraface'
				train_image_feats = csvread('/home/aditya/honours/demo/aradhya/feats.csv');
				train_labels = train_image_feats(:,end);
				train_image_feats = train_image_feats(:,end-1);


        %train_image_feats = get_gabor_features(train_image_paths);
        %test_image_feats = get_gabor_features(test_image_paths);
        
    case 'bag of gabor'
        
        if ~exist('vocab_gabor.mat','file')
            fprintf('Building gabor vocab\n');
            vocab_size = 400;
            vocab_gabor = build_vocabulary_gabor(train_image_paths,vocab_size);
            save('vocab_gabor.mat','vocab_gabor');
        end
        
        if ~exist('train_feats_bag_gabor.mat','file')
            train_image_feats = get_gabor_bag(train_image_paths);
            save('train_feats_bag_gabor.mat','train_image_feats')
        else
            load('train_feats_bag_gabor.mat');
        end
        
        %train_image_feats = get_gabor_features(train_image_paths);
        %test_image_feats = get_gabor_features(test_image_paths);
        
    case 'bag of sift'
        if ~exist('vocab.mat', 'file')
            fprintf('No existing visual word vocabulary found. Computing one from training images\n')
            vocab_size = 100; %Larger values will work better (to a point) but be slower to compute
            vocab = build_vocabulary(train_image_paths, vocab_size);
            save('vocab.mat', 'vocab')
        end
        
        if ~exist('train_feats.mat','file')
            train_image_feats = get_bags_of_sifts(train_image_paths);
            save('train_feats.mat','train_image_feats')
        else
            load('train_feats.mat');
        end
        
        
    otherwise
        error('Unknown feature type')
end


%% Step 2: Classify each test image by training and using the appropriate classifier
% Each function to classify test features will return an N x 1 cell array,
% where N is the number of test cases and each entry is a string indicating
% the predicted category for each test image. Each entry in
% 'predicted_categories' must be one of the 15 strings in 'categories',
% 'train_labels', and 'test_labels'. See the starter code for each function
% for more details.

fprintf('Using %s classifier to predict test set categories\n', CLASSIFIER)

switch lower(CLASSIFIER)    
        
    case 'support vector machine'
    
        predicted_categories = svm_classify(train_image_feats, train_labels, test_image_feats);
    
    case 'support vector machine kernel'
    
        predicted_categories = svm_classify_kernel(train_image_feats, train_labels, test_image_feats); 
    
    case 'support vector machine mkl'
    
        predicted_categories = svm_classify_mkl(train_image_feats, train_labels);

    otherwise
        error('Unknown classifier type')
end



%% Step 3: Build a confusion matrix and score the recognition system
% You do not need to code anything in this section. 

% If we wanted to evaluate our recognition method properly we would train
% and test on many random splits of the data. You are not required to do so
% for this project.

% This function will recreate results_webpage/index.html and various image
% thumbnails each time it is called. View the webpage to help interpret
% your classifier performance. Where is it making mistakes? Are the
% confusions reasonable?
%create_results_webpage( train_image_paths, ...
%                        test_image_paths, ...
%                        train_labels, ...
%                        test_labels, ...
%                        categories, ...
%                        abbr_categories, ...
%                        predicted_categories)

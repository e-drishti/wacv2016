% Starter code prepared by James Hays for CS 143, Brown University

%This function will sample SIFT descriptors from the training images,
%cluster them with kmeans, and then return the cluster centers.

function vocab = build_vocabulary_gabor( image_paths, vocab_size )
% The inputs are images, a N x 1 cell array of image paths and the size of 
% the vocabulary.

% The output 'vocab' should be vocab_size x 128. Each row is a cluster
% centroid / visual word.

%{
Useful functions:
[locations, SIFT_features] = vl_dsift(img) 
 http://www.vlfeat.org/matlab/vl_dsift.html
 locations is a 2 x n list list of locations, which can be thrown away here
  (but possibly used for extra credit in get_bags_of_sifts if you're making
  a "spatial pyramid").
 SIFT_features is a 128 x N matrix of SIFT features
  note: there are step, bin size, and smoothing parameters you can
  manipulate for vl_dsift(). We recommend debugging with the 'fast'
  parameter. This approximate version of SIFT is about 20 times faster to
  compute. Also, be sure not to use the default value of step size. It will
  be very slow and you'll see relatively little performance gain from
  extremely dense sampling. You are welcome to use your own SIFT feature
  code! It will probably be slower, though.

[centers, assignments] = vl_kmeans(X, K)
 http://www.vlfeat.org/matlab/vl_kmeans.html
  X is a d x M matrix of sampled SIFT features, where M is the number of
   features sampled. M should be pretty large! Make sure matrix is of type
   single to be safe. E.g. single(matrix).
  K is the number of clusters desired (vocab_size)
  centers is a d x K matrix of cluster centroids. This is your vocabulary.
   You can disregard 'assignments'.

  Matlab has a build in kmeans function, see 'help kmeans', but it is
  slower.
%}

% Load images from the training set. To save computation time, you don't
% necessarily need to sample from all images, although it would be better
% to do so. You can randomly sample the descriptors from each image to save
% memory and speed up the clustering. Or you can simply call vl_dsift with
% a large step size here, but a smaller step size in make_hist.m. 

% For each loaded image, get some SIFT features. You don't have to get as
% many SIFT features as you will in get_bags_of_sift.m, because you're only
% trying to get a representative sample here.

% Once you have tens of thousands of SIFT features from many training
% images, cluster them with kmeans. The resulting centroids are now your
% visual word vocabulary.

% Feature matrix for each image. More work to preallocate and copy
feature_mat = [];

num_paths = size(image_paths);
num_paths = num_paths(1);
% Big matrix to store the feature representation of all images
dim = 32;
gabor_size = 40;
G = create_gabor();
faceDetector = vision.CascadeObjectDetector();
for i=1:num_paths
    path = image_paths{i};
    img = rgb2gray(im2double(imread(path)));
    img = histeq(img);
    % Detect Face and crop only face
    bbox = step(faceDetector, img);
    max_ind = 1;
    max_area = 0;
    if(isempty(bbox))
        continue
    end
    for j=1:size(bbox,1)
        if((bbox(j,3)*bbox(j,4))>max_area)
            max_area = bbox(j,3)*bbox(j,4);
            max_ind = j;
        end
    end
    
    img = imcrop(img,bbox(max_ind,:));
    % Subsample face to fixed scale
    img = imresize(img,[dim dim]);
    for k=1:size(G,1)
        for j= 1:size(G,2)
            filt = imfilter(img,G{k,j},'same');
            %size(filt)
            filt = filt(:);
            filt = filt';
            filt = filt - mean(filt);
            filt = filt/norm(filt);
            feature_mat = [feature_mat; filt];
        end
    end
end

% Cluster the features into vocab_size clusters. using plusplus initialization
[centers, assignments] = vl_kmeans(single(feature_mat'), vocab_size,'Initialization', 'plusplus');
% the vocab matrix is the cluster centers returned 
vocab = centers';






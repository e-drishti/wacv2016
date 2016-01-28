% Starter code prepared by James Hays for CS 143, Brown University

%This feature is inspired by the simple tiny images used as features in 
%  80 million tiny images: a large dataset for non-parametric object and
%  scene recognition. A. Torralba, R. Fergus, W. T. Freeman. IEEE
%  Transactions on Pattern Analysis and Machine Intelligence, vol.30(11),
%  pp. 1958-1970, 2008. http://groups.csail.mit.edu/vision/TinyImages/

function image_feats = get_gabor_features(image_paths)
% image_paths is an N x 1 cell array of strings where each string is an
%  image path on the file system.
% image_feats is an N x d matrix of resized and then vectorized tiny
%  images. E.g. if the images are resized to 16x16, d would equal 256.

% To build a tiny image feature, simply resize the original image to a very
% small square resolution, e.g. 16x16. You can either resize the images to
% square while ignoring their aspect ratio or you can crop the center
% square portion out of each image. Making the tiny images zero mean and
% unit length (normalizing them) will increase performance modestly.

% suggested functions: imread, imresize

num_paths = size(image_paths);
num_paths = num_paths(1);
% Big matrix to store the feature representation of all images
dim = 32;
gabor_size = 40;
image_feats = zeros(num_paths,dim*dim*gabor_size);
G = create_gabor();
faceDetector = vision.CascadeObjectDetector();
for i=1:num_paths
    path = image_paths{i};
    path
    img = im2double(imread(path));
    img = histeq(img);
    % Detect Face and crop only face, if more than one bbox, take max area
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
    % Apply gabor filters and concatenate features
    features = zeros(1,dim*dim*gabor_size);
    offset = 1;
    for k=1:size(G,1)
        for j= 1:size(G,2)
            filt = imfilter(img,G{k,j},'same');
            %size(filt)
            filt = filt(:);
            filt = filt';
            %size(filt)
            %offset
            features(1,offset:offset-1+(dim*dim)) = filt;
            offset = offset+(dim*dim);
        end
    end
    % Make it 0-mean and normalize
    features = features - mean(features);
    features = features/norm(features);
    image_feats(i,:) = features;
end

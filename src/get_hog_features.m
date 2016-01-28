function [ image_feats ] = get_hog_features( image_paths )
% Function to get HOG features of images

num_paths = size(image_paths);
num_paths = num_paths(1);
% Big matrix to store the feature representation of all images
dim = 100;
cell_size = 100;
hog_dim = (dim/cell_size)*(dim/cell_size)*31;
image_feats = zeros(num_paths,hog_dim);
faceDetector = vision.CascadeObjectDetector();
for i=1:num_paths
    path = image_paths{i};
		img = im2double(imread(path));
    img = imresize(img,[100 100]);
		num_dimensions = size(size(img),2);
		if num_dimensions == 3
			img = rgb2gray(img);
		end
		img = histeq(img);
    % Detect Face and crop only face, if more than one bbox, take max area
    %bbox = step(faceDetector, img);
    %max_ind = 1;
    %max_area = 0;
    %if(isempty(bbox))
    %    continue
    %end
    %for j=1:size(bbox,1)
    %    if((bbox(j,3)*bbox(j,4))>max_area)
    %        max_area = bbox(j,3)*bbox(j,4);
    %        max_ind = j;
    %    end
    %end
    %
    %img = imcrop(img,bbox(max_ind,:));
    %% Subsample face to fixed scale
    %img = imresize(img,[dim dim]);
    % Apply hog filters and concatenate features
    hog = vl_hog(single(img),cell_size);
    features = hog(:);
    features = features / norm(features);
		image_feats(i,:) = features;
end


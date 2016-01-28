function [ bbox ] = face_detect( img )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% Create a cascade detector object.
faceDetector = vision.CascadeObjectDetector();

% Read a video frame and run the detector.
bbox            = step(faceDetector, img);

% Draw the returned bounding box around the detected face.
videoOut = insertObjectAnnotation(img,'rectangle',bbox,'Face');
figure, imshow(videoOut), title('Detected face');

end


%% Function to extract frames from a video file
% The function takes as parameter a string which is the path to the video
% file and a string which would be prepended to the output images.
function [ ] = extract_frames( path, path_prepend)

% Read the video
mov = VideoReader(path);

% Output folder for extracted images
opFolder = fullfile(cd, 'snaps');

if ~exist(opFolder, 'dir')

    mkdir(opFolder);
end


numFrames = mov.NumberOfFrames;
numFrames
% start index for collected frames
offset = 1;
% limiting the number of frames collected
% if numFrames > 1000
%     numFrames = 12500
% end

numFramesWritten = 0;

% Collecting frames from the video with a step of 10 frames
for t = offset : 250: offset+numFrames
    % Get the t-th frame
    currFrame = read(mov, t);    
    opBaseFileName = sprintf('%s_%3.3d.png', path_prepend,t);
    opFullFileName = fullfile(opFolder, opBaseFileName);
    % Write the frame as an image
    imwrite(currFrame, opFullFileName, 'png'); 

    progIndication = sprintf('Wrote frame %4d of %d.', t, numFrames);
    disp(progIndication);
    numFramesWritten = numFramesWritten + 1;
end     
progIndication = sprintf('Wrote %d frames to folder "%s"',numFramesWritten, opFolder);
disp(progIndication);

end


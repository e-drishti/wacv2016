function [G] = create_gabor()
scales = 5;
orientations = 8;
if ~exist('gabor.mat','file')
    G = cell(scales, orientations);
    for s = 1:scales
        for j = 1:orientations
            G{s,j}=zeros(32,32);
        end
    end
    for s = 1:scales
        for j = 1:orientations
            G{s,orientations+1-j} = gabor([32 32],(s-1),j-1,pi,sqrt(2),pi);
        end
    end

    figure;
    for s = 1:scales
        for j = 1:orientations       
            subplot(scales,orientations,(s-1)*orientations+j);        
            imshow(real(G{s,j}),[]);
        end
    end

    for s = 1:scales
        for j = 1:orientations       
            G{s,j}=fft2(G{s,j});
        end
    end
    save('gabor.mat', 'G');
else
    load('gabor.mat');
end
end

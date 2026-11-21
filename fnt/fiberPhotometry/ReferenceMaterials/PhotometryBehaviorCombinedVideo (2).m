%% Retrieve relevant data
close all
clear all

% Photometry data must be pre-processed!
% Load dordata_proc file for the trial
data1 = importdata('NYM005_OFT_0_Tue, Dec 10, 2019.dat');
trace405 = data1.data(:, [3]);
trace473 = data1.data(:, [2]);

% Calculate dF/F
% dFF= fitcaldff(dordata_proc.isosbestic,dordata_proc.gcamp);
dFF = filteredfitcaldff(trace405,trace473);

% Retrieve Doric time
time= data1.data(:,1);

% Get TTLs for video frames
ttl1= data1.data(:,4); % round to nearest Doric timestamp (.01)
ttl1 = ttl1<1;
ttl=zeros(length(ttl1),1);
ttl(find(diff(ttl1)==1))=1;


% Remove duplicate TTL timestamps
% i= 1; % good TTL counter
% for j= 1:length(ttl)-1
%     if ttl(j) ~= ttl(j+1) % if this TTL isn't identical to the next one
%         ttl2(i)= ttl(j); % then save this ttl as a good one
%         i= i+1; % increase the counter
%     end
% end
% ttl2(length(ttl2)+1)= max(ttl); % add last TTL value
% ttl2= ttl2';

%% Create 'data' matrix that has time (1), dFF (2), and video TTL (3)
data(:,1) = time;
data(:,2) = dFF;
data(:,3) = ttl;
% b= 1; % ttl2 counter
% for a= 1:length(time)
%     if b <= length(ttl2) % if haven't reached end of ttl2 array
%         if ttl2(b) == time(a) % if the TTL timestamp matches the Doric timestamp
%             data(a,3) = 1; % mark that timestamp with a 1
%             b= b+1; % move to the next ttl2 value
%         else
%             data(a,3)= 0; % Doric timestamps without video
%         end
%     end
% end

%% Process data, create video
% Identify video file
% fp = 'B:\Data\mPFC-VTA Photometry\Robot OFT\Videos'; % CHANGE PER MOUSE
% videoobj = VideoReader(fullfile(fp,'886_20190228a_orig.mp4')); % CHANGE PER MOUSE
videoobj = VideoReader('NYM005_OFT_2019-12-10-160730-0000.mp4');

% Set parameters for video
width = 2500; % frames for x axis
frameN = videoobj.NumberOfFrames;
cameraOut = sum(data(:,3)); % number of video frames according to TTL pulses
cframeN = cameraOut; % % number of video frames according to TTL pulses
mframeN = cameraOut-frameN; % missing number of frames
fframeN = cframeN-mframeN; % TTL pulse frames minus missing number of video frames
onCamera = find(data(:,3)==1,1); % Index at which the camera turns on
offCamera = max(find(data(:,3)==1,fframeN)); % Index at which the camera takes its last frame
% Some frames are missing. Took out the last few missing frames for now.
% Not sure when frames are missing

frameIndex = find(data(:,3)); % Indicies at which frames were taken

% Cut off frame indicies after the video (?)
frameIndex = frameIndex(find(frameIndex < length(data)-width));
if length(frameIndex) > frameN
    frameIndex(length(frameIndex)-(length(frameIndex)-frameN):length(frameIndex)) = [];
end

tic
% Call video object again (this is necessary, but not sure why)
% videoobj = VideoReader(fullfile(fp,'886_20190228a_orig.mp4')); % CHANGE PER MOUSE
videoobj = VideoReader('NYM005_OFT_2019-12-10-160730-0000.mp4');

% Specify what the written video will be called
% aviobj = VideoWriter(fullfile(fp,'combined')); % CHANGE PER MOUSE
aviobj = VideoWriter('combined3');
aviobj.FrameRate=30;
open(aviobj);

% Find frame after 1000 samples(=width)
framestart= find(frameIndex > width, 1);

% Set up the figure
h=figure('Visible','on');
set(h,'position',[100 100 640 512]);
ylim([min(data(:,2)) max(data(:,2))]);
x1=data(frameIndex(framestart)-width,1);
x2=data(frameIndex(framestart)+width,1);

tic
plotimage= cell(length(frameIndex),1);

% Combined video will save every -interval- video frames; 5 is good
interval= 2;

for k = framestart:interval:length(frameIndex)
    hold on;
    
    % Plot the photometry data and current time vertical line for this
    % video frame's width
    h6=plot(data([frameIndex(k)-width:frameIndex(k)+width],1),data([frameIndex(k)-width:frameIndex(k)+width],2),'k');
    h7=line([data(frameIndex(k),1) data(frameIndex(k),1)],[min(data(:,2)) max(data(:,2))],'Color',[1 0 0]);
    
    xlim([x1 x2]);
    x1=data(frameIndex(k)-width,1);
    x2=data(frameIndex(k)+width,1);
    xlabel('Time (s)')
    ylabel('dF/F (%)')
    frame = getframe(gcf);
    plotimage = frame.cdata;
    delete(h6);
    delete(h7);
    
    currentimage = read(videoobj,k);
    catimage=cat(1,currentimage,plotimage);
    writeVideo(aviobj,catimage);
end
toc;

% Write the combined video
close(aviobj);
toc

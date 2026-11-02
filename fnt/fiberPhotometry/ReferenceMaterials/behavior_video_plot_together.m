%NOT WORKING YET. IN PROGRESS

%read the video and convert into grayscale
%%
path='G:\Akash ethovision data\1320 070315 all behaviors\1320 061915 all behaviors\Media Files\Trial     8';
%import traces as extractedtraces and time as time
%path=uigetfile('*.*', 'Select the behavior movie','G:\Akash ethovision data');
% vobj=VideoReader(strcat(path,'.mpg'));
% nframes=vobj.NumberOfFrames;
% %nframes=5400;
% vid=read(vobj);
%obj=VideoWriter('graymovie.mpg');
%open(obj);
%%
% %mov = zeros(vobj.Height, vobj.Width, 5400);
% for i=1:nframes
%     mov(:,:,i)=rgb2gray(vid(:,:,:,i));
%    
% end
% %writeVideo(obj,mov);
% %close(obj);
% [m1,n1,o1]=size(mov);
ttl= data1.data(:,4); 
figure; plot(ttl)
ttl2=ttl<0.5;
ttl_good=zeros(length(ttl2),1);
ttl_good(find(diff(ttl2)==1))=1;
num_of_ttl_frames=find(ttl_good==1);
extracted_traces_full=dFF(find(ttl_good==1));

%%

hvid=VideoReader('E:\Data\4_DA_2020_Territory_proc\12.10.2019_NY_M005_OFT_P001\vid.mp4');
numofframes=hvid.NumberofFrames;
total_duration=hvid.Duration;
for ts=0:1/30:total_duration
    hvid.CurrentTime=ts;
    prefinalImage1=hvid.readFrame;
end
[m2,n2,o2]=size(prefinalImage1);
if numofframes<num_of_ttl_frames
    extracted_traces=extracted_traces_full(1:numofframes);
end
    
%% making offset plot
[m,n]=size(extractedtraces);
%offt=input('Enter offset for raw traces');
% offt=0.05;
% for i=1:n
%        offextractedtraces(:,i)= (extractedtraces(:,i)+((i+1)*offt));
% end
%%
%concatenating the 2 matrices and subplotting the window plot and writing it into a file
writerobj=VideoWriter(strcat(hvid.Path,'\combinedvideo2','.avi'));
writerobj.Quality=80;
writerobj.FrameRate=30;
open(writerobj);
mov2=vertcat(prefinalImage1,zeros((m2),n2,o2));

    for i=1:o2
    
        finalmovie(:,:,i)=uint8([mov2(:,:,i)]);
        subplot(2,2,1:2)
        imshow(finalmovie(:,:,i),'DisplayRange',[])
        %colormap gray;
        if (i+100)<o2
            subplot(2,2,3:4)
            plot(time((i:i+100),:),extractedtraces((i:i+100),:));
        else
            subplot(2,2,3:4)
            plot(time((i:o2),:),extractedtraces((i:o2),:));
        end
        writeVideo(writerobj,getframe(gca));
    end
close(writerobj);












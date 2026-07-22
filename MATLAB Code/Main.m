%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Implementation of 
% % "HP-MEF: Human Perception-based Multi-Exposure Image Fusion"
% % by
% % Ashish Vanmali, Alfiya Shaikh, Kavish Rathod, Prapti Raut, Shaista Khanam, Trupti Shah
% %
% % Written by Ashish V. Vanmali, India
% % e-mail: ashishvanmaliiitb@gmail.com, vanmaliashish@gmail.com
% %
% % Last Updated - June 2026
% %
% % This work is submitted to -
% % Information Fusion 
% %
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; clear; close all; warning off;

folder_name = '1_Eiffel_Tower';
% folder_name = '4_Office';

I = load_images(folder_name);  
N = size(I,4);

figure('Name','Input Images');
for i = 1:N
    subplot(2, ceil(N/2), i);
    imshow(I(:,:,:,i));
    title(['Image ' num2str(i)]);
end

[R, W] = exposure_fusion(I);

figure('Name','Normalized Weight Maps');
for i = 1:N
    subplot(2, ceil(N/2), i);
    imshow(W(:,:,i),[]);
    title(['Weight Map ' num2str(i)]);
end

%% Post Processing
R(R<0) = 0;
R(R>1) = 1;

[H, S, V] = rgb2hsv(R);

H2 = H * 1.1 ;
H2(H2>1) = 1;

S2 = S * 1.2 ;
S2(S2>1) = 1;

gamma = 0.85;       %% gamma preset to 0.85
lambda = 0.4;       %% lambda preset to 0.4

hsharp = [-1 -1 -1; -1 8 -1; -1 -1 -1] / 3;     %construct a sharpening mask
Recon_sharp = imfilter(V,hsharp,'replicate');
V2 = V.^gamma + lambda*Recon_sharp ; 

R2(:,:,1) = H2;
R2(:,:,2) = S2;
R2(:,:,3) = V2;
R_out = hsv2rgb(R2);

figure('Name','Final Output');
imshow(R_out,[]);


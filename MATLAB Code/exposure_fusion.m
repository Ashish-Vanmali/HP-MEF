function [R, W] = exposure_fusion(I)
r = size(I,1);
c = size(I,2);
N = size(I,4);

[Cont, AvgC] = contrast_measure(I);
[Sat,  AvgS] = saturation_measure(I);
[Wexp, AvgW] = wellexp_measure(I);
[Entr, AvgE] = entropy_measure(I);

nC = AvgC ;
nS = AvgS ;
nW = AvgW ;
nE = AvgE ;

W_par = [nC(:), nS(:), nW(:), nE(:)];  
disp('Adaptive weight matrix W_par (N x 4):');
disp(W_par);

Score = prod(W_par,2)
PS = Score ./ sum(Score)

W = zeros(r,c,N);
for i = 1:N
    W(:,:,i) = ( Cont(:,:,i).^( 1+PS(i)) ).* ...
               ( Sat(:,:,i) .^( 1+PS(i)) ).* ...
               ( Wexp(:,:,i).^( 1+PS(i)) ).* ...
               ( Entr(:,:,i).^( 1+PS(i)) );               
end

% Normalize weight maps
W = W + 1e-12;
W = W ./ repmat(sum(W,3), [1 1 N]);

% Multiresolution blending
pyr = gaussian_pyramid(zeros(r,c,3));  % Initialize pyramid
nlev = length(pyr);

for i = 1:N
    pyrW = gaussian_pyramid(W(:,:,i));
    pyrI = laplacian_pyramid(I(:,:,:,i));

    for l = 1:nlev
        w = repmat(pyrW{l}, [1 1 3]);
        pyr{l} = pyr{l} + w .* pyrI{l};
    end
end

% Reconstruct final fused image
R = reconstruct_laplacian_pyramid(pyr);
end

%% ----------------------------
%% CONTRAST
function [C, AvgC] = contrast_measure(I)
h = [0 1 0; 1 -4 1; 0 1 0];
N = size(I,4);
C = zeros(size(I,1), size(I,2), N);
AvgC = zeros(1,N);

for i = 1:N
    mono = rgb2gray(I(:,:,:,i));
    C(:,:,i) = ( abs(imfilter(mono, h, 'replicate')) );
end
C = mat2gray(C);
for i = 1:N
    tmp = C(:,:,i);
    AvgC(i) = mean(tmp(:));
end
end

%% ----------------------------
%% SATURATION
function [C, AvgS] = saturation_measure(I)
N = size(I,4);
C = zeros(size(I,1), size(I,2), N);
AvgS = zeros(1,N);

for i = 1:N
    R = I(:,:,1,i); G = I(:,:,2,i); B = I(:,:,3,i);
    mu = (R + G + B)/3;
    C(:,:,i) = ( sqrt(((R-mu).^2 + (G-mu).^2 + (B-mu).^2)/3) );
end
C = mat2gray(C);
for i = 1:N    
    tmp = C(:,:,i);
    AvgS(i) = mean(tmp(:));
end
end

%% ----------------------------
%% WELL-EXPOSEDNESS
function [C, AvgW] = wellexp_measure(I)
sig = 0.2;
N = size(I,4);
C = zeros(size(I,1), size(I,2), N);
AvgW = zeros(1,N);

for i = 1:N
    R = exp(-0.5*(I(:,:,1,i)-0.5).^2/sig^2);
    G = exp(-0.5*(I(:,:,2,i)-0.5).^2/sig^2);
    B = exp(-0.5*(I(:,:,3,i)-0.5).^2/sig^2);
    C(:,:,i) = ( R .* G .* B );
end
C = mat2gray(C);
for i = 1:N
    tmp = C(:,:,i);
    AvgW(i) = mean(tmp(:));
end
end

%% ----------------------------
%% ENTROPY
function [C, AvgE] = entropy_measure(I)
window = true(9);
N = size(I,4);
C = zeros(size(I,1), size(I,2), N);
AvgE = zeros(1,N);

for i = 1:N
    mono = rgb2gray(I(:,:,:,i));
    C(:,:,i) = entropyfilt(mono, window);  % raw entropy
    tmp = C(:,:,i);                  
    C_min = min(tmp(:));
    C_max = max(tmp(:));
    C(:,:,i) = 7 * (tmp - C_min) / (C_max - C_min);
end
C = mat2gray(C);
for i = 1:N
    tmp2 = C(:,:,i);                 
    AvgE(i) = mean(tmp2(:));         
end
end




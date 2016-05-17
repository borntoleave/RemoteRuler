function imeasurer(f1,f2,translation)
% UNTITLED Summary of this function goes here
% Detailed explanation goes here
% Real objects are measured in mm. Image are measured in pixels.
% Final Project, Computer Vision
% Rongzhong, Saurabh, Xin


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%Set All the Parameters Manually %%%%%%%%%%%%%%%%%
%thresholds for matching features in two images
tau_Ed = 0.9;
tau_H = 0.7; 
tau_inliner = 0.8;
% [width,heigth] around the selected pixel point
win = [35 4];
% show meters
measure_scale = 10^3;
% canny edge parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  mLow : the threshold is set adaptively: low_threshold = mLow* mean_intensity(im_gradient)
mLow=0.3;
%  mHigh: the threshol is set adaptively: high_threshold= mHigh*low_threshold
mHigh=9;
%  sig  : the value of sigma for the derivative of gaussian operator
sig=2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%Set the Camera Parameters%%%%%%%%%%%%%%%%%%%%%%
%translation= 30000     %body_l-10 %297*3; %210; %body_l-5;   %mm
focal_l=4.12;   %mm
sensor_w=4.89;  %mm
sensor_h=3.67;   %mm
image_w=3264;     %pxl
image_h=2448;     %pxl
%sensor_d=sqrt(sensor_w^2+sensor_h^2);
%fov=atan(sensor_w/2/focal_l)/pi*180*2;
%body_l=123.8;   %mm
%image_d=sqrt(image_w^2+image_h^2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

img1 = imread(f1);
img2 = imread(f2);
re_scale = 0.2;
% canny edge
img1 = imresize(img1,re_scale);
img2 = imresize(img2,re_scale);
%G = fspecial('gaussian',[9 9],4);
%%% Filter it
%img1 = imfilter(img1,G,'same');
%img2 = imfilter(img2,G,'same');
[B1,T1] = canny(img1,mLow,mHigh,sig);
[B2,T2] = canny(img2,mLow,mHigh,sig);

% pick the edge point manually, select window around it automatically
[w1, h1, l1, B1_edge1, B1_edge2] = pick(B1,win);
[w2, h2, l2, B2_edge1, B2_edge2] = pick(B2,win);
B1_size_r1 = size(B1_edge1);
B1_size_r2 = size(B1_edge2);
B2_size_r1 = size(B2_edge1);
B2_size_r2 = size(B2_edge2);
B1_r = [5 5];
B2_r = [5 5];

B1_circles1= [B1_edge1,[B1_r(1)*ones(B1_size_r1(1),1)]];
B1_circles2= [B1_edge2,[B1_r(2)*ones(B1_size_r2(1),1)]];
B2_circles1= [B2_edge1,[B2_r(1)*ones(B2_size_r1(1),1)]];
B2_circles2= [B2_edge2,[B2_r(2)*ones(B2_size_r2(1),1)]];

% extract the 128*N descriptors for the N features(key points) of two images
% edge1
sift_arr1 = find_sift(B1, B1_circles1);
sift_arr2 = find_sift(B2, B2_circles1);
% edge2
sift_arr3 = find_sift(B1, B1_circles2);
sift_arr4 = find_sift(B2, B2_circles2);

% compute the matched points from the edges of the two images
try
    [img1_pts_edge1,img2_pts_edge1] = ComputeMatch(B1_edge1, B2_edge1,sift_arr1,sift_arr2,tau_Ed);
catch
    disp('No matched edge, Please select points again')
    return;
end
try
    [img1_pts_edge2,img2_pts_edge2] = ComputeMatch(B1_edge2, B2_edge2,sift_arr3,sift_arr4,tau_Ed);
catch
    disp('No matched edge, Please select points again')
    return;
end

% use RANSAC to find the REAL matched edge points
[img1_edge1_inliners,img2_edge1_inliners] = RANSAC(img1_pts_edge1,img2_pts_edge1,tau_H,tau_inliner)
[img1_edge2_inliners,img2_edge2_inliners] = RANSAC(img1_pts_edge2,img2_pts_edge2,tau_H,tau_inliner)


[a1,b1]=size(img1_edge1_inliners);
[a2,b2]=size(img1_edge2_inliners);

% compute all the heights.
total_real_l = [];
for i = 1:b1
    for j = 1:b2
        l1 = img1_edge2_inliners(2,j)- img1_edge1_inliners(2,i);
        l2 = img2_edge2_inliners(2,j)- img2_edge1_inliners(2,i);
        l1= l1*(1/re_scale);
        l2= l2*(1/re_scale);
        % The algorithms to calculate the real height
        real_l=abs(translation/focal_l/image_w*sensor_w/(1/l1-1/l2));
        total_real_l = [total_real_l real_l];
    end
end

height_mean = mean(total_real_l)/measure_scale
height_std = std(total_real_l)/measure_scale
height_max = max(total_real_l)/measure_scale
height_min =min(total_real_l)/measure_scale
%hist(round(total_real_l/measure_scale))
figure();
hist(total_real_l/measure_scale) 


end


function [a,b,c,edge1,edge2]=pick(img_mat,win)
% manually pick a point from top and bottom of the subject
% make a widnow around the point to select more features of the edge points

figure
imshow(img_mat);
w=size(img_mat,2);
h=size(img_mat,1);
% get 2 coordinates 
[x y]=ginput(2); 
% make a window around the picked point 
for i = 1:2
    edge2 = [];
    for m = max(1,round(x(i))-win(1)):min(round(x(i))+win(1),w)
        for n = max(1,round(y(i))-win(2)):min(round(y(i))+win(2),h)          
            if img_mat(n,m) == 1 
                edge2 = [edge2;[m n]];
            end            
        end
    end
    if i == 1
        edge1 = edge2;
    end    
end
close
figure
imagesc(img_mat); 
axis image;
colormap gray;
hold on   
% select all the edge points in the window 
plot(edge1(:,1),edge1(:,2),'b.','MarkerSize',1);
plot(edge2(:,1),edge2(:,2),'b.','MarkerSize',1)
             
center=1/2*[w h];
% Distance = sqrt((x(1)-x(2))^2 + (y(1)-y(2))^2)
a=norm([x(1) y(1)]-center);
b=norm([x(2) y(2)]-center);
%c=norm([x(1) y(1)]-[x(2) y(2)]);
c=y(2)-y(1);

end

function [img1_pts_inliners,img2_pts_inliners] = RANSAC(img1_pts_all,img2_pts_all,tau_H,tau_inliner)
img1_pts_all = img1_pts_all';
img2_pts_all = img2_pts_all';
bestH = [];
bestInlinerFrac = 0;
numofmatches = length(img1_pts_all);
% start to run RANSAC
N = 1000;
for i =1: N
    disp(i);
    % for each run, random choosing 4 points in each image to do the homography transform
    ord = randperm(numofmatches);
    ord = ord(1:4);
    img1_pts = img1_pts_all(:,ord);
    img2_pts = img2_pts_all(:,ord);
    Hi = ComputeHomography(img1_pts,img2_pts);
    % get the Hi, and calculate the predicted points in the second image using that specific Hi
    img2_pts_all_predicted = Hi*[img1_pts_all;ones(1,numofmatches)];
    img2_pts_all_predicted = [img2_pts_all_predicted(1, :)./ img2_pts_all_predicted(3, :);...
        img2_pts_all_predicted(2, :)./ img2_pts_all_predicted(3, :)];
    
    % calculate the distance between the real points and the predicted points
    dist = sqrt(sum((img2_pts_all-img2_pts_all_predicted).^2,1));
    % set a threshold for the dist to find the inliners index
    indx_inliners = find(dist < tau_H);
    indx_inliners_best = indx_inliners;
    FracofInliners = length(indx_inliners)/numofmatches;
    
    % find the best H
    if FracofInliners >= tau_inliner
        bestH = Hi;
        bestInlinerFrac = FracofInliners;
        indx_inliners_best = indx_inliners;
        break;
    end
    if FracofInliners >= bestInlinerFrac
        bestInlinerFrac = FracofInliners;
        bestH = Hi;
        indx_inliners_best = indx_inliners;
    end
end

if isempty(bestH)
    H = [];
else
    H = ComputeHomography(img1_pts_all(:,indx_inliners), img2_pts_all(:,indx_inliners));
end

% the x,y-coordinates of the inliners points in two images
img1_pts_inliners = img1_pts_all(:,indx_inliners_best);
img2_pts_inliners = img2_pts_all(:,indx_inliners_best);

end


function H = ComputeHomography(img1_pts,img2_pts)
% compute the Homography tranform

A = [];
n = size(img1_pts, 2);
for i=1:n
    xyi = img1_pts(:,i);
    xi = xyi(1);yi = xyi(2);
    xyi_prime = img2_pts(:,i);
    xi_prime = xyi_prime(1);yi_prime = xyi_prime(2);
    Ai = [xi yi 1 0   0  0 (-1)*xi_prime*xi (-1)*xi_prime*yi (-1)*xi_prime;...
         0   0 0 xi yi  1 (-1)*yi_prime*xi (-1)*yi_prime*yi (-1)*yi_prime];
    A = [A;Ai];
end
[U, E, V] = svd(A);
h = V(:,end);
H = reshape(h,3,3)';

end

function [img1_pts_all,img2_pts_all] = ComputeMatch(B1_edge1, B2_edge1,sift_arr1, sift_arr2,tau_Ed)
% calculate the Euclidean distance between descriptors in two images
Ed_dist = dist2(sift_arr1, sift_arr2);
% sort by column
[Ed_dist, idx] = sort(Ed_dist, 'ascend');
% set a threshold to compute the matched points in two images
selected = (Ed_dist(1, :) ./ Ed_dist(2, :)) < tau_Ed;
matches = [];
if (sum(selected) > 0)
    matches = zeros(sum(selected), 2);
    tmpIdx = 1:size(sift_arr2,1);
    matches(:, 2) = tmpIdx(selected);
    matches(:, 1) = idx(1, selected);
end
ind1 = matches(:, 1);
ind2 = matches(:, 2);

% the x,y-coordinates of the matched points in two images
img1_pts_all = [B1_edge1(ind1,:)];
img2_pts_all = [B2_edge1(ind2,:)];
% plot(img1_pts_all(:,1),img1_pts_all(:,2),'b.');

end




function n2 = dist2(x, c)
%DIST2	Calculates squared distance between two sets of points.
%
%	Description
%	D = DIST2(X, C) takes two matrices of vectors and calculates the
%	squared Euclidean distance between them.  Both matrices must be of
%	the same column dimension.  If X has M rows and N columns, and C has
%	L rows and N columns, then the result has M rows and L columns.  The
%	I, Jth entry is the  squared distance from the Ith row of X to the
%	Jth row of C.
%
%	See also
%	GMMACTIV, KMEANS, RBFFWD
%

%	Copyright (c) Christopher M Bishop, Ian T Nabney (1996, 1997)

[ndata, dimx] = size(x);
[ncentres, dimc] = size(c);
if dimx ~= dimc
	error('Data dimension does not match dimension of centres')
end

n2 = (ones(ncentres, 1) * sum((x.^2)', 1))' + ...
  		ones(ndata, 1) * sum((c.^2)',1) - ...
  		2.*(x*(c'));

end








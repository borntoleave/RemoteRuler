function [  ] = simple( f1,height )
%UNTITLED Summary of this function goes here
%Detailed explanation goes here
%Real objects are measured in mm. Image are measured in pixels.


focal_l=4.12;%mm
sensor_w=4.89;%mm
sensor_h=3.67;%mm
sensor_d=sqrt(sensor_w^2+sensor_h^2);
%fov=atan(sensor_w/2/focal_l)/pi*180*2;

image_w=3264;%pxl
image_h=2448;%pxl
image_d=sqrt(image_w^2+image_h^2);
pom=image_w/sensor_w;%pixel over milli-meter

[w,h,x,y]=pick(f1);
%t1=tan((h/2-y(2))/pom/focal_l);
%t2=tan((y(1)-h/2)/pom/focal_l);

dist=height*focal_l*pom/(y(2)-h/2)
real_h=height+(h/2-y(1))/pom*dist/focal_l

 


end

function [w,h,x,y]=pick(img)
imshow(img);
img_mat=(imread(img));

[x y]=ginput(2);%get 2 coordinates

w=size(img_mat,2);
h=size(img_mat,1);

end


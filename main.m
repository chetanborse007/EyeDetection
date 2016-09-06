clear; clc; close all;

img_path = './validation/';
img_num = 10;
img_dir = dir([img_path,'*.jpg']);
load('validation_gt.mat'); % ground truth

normlized_dist = zeros(img_num,1);

for i = 1:img_num
    
    img = imread([img_path,img_dir(i).name]);
    [left_x, right_x, left_y, right_y] = eye_detection(img);
    
    close all;
    f = figure;
    imshow(img);
    hold on;
    plot([left_x, right_x], [left_y, right_y],'r*');
    saveas(f,['output_',img_dir(i).name]);
    
    [h,w,~] = size(img);
    left_dist = sqrt( (x(i,1)-left_x).^2 + (y(i,1)-left_y) );
    right_dist = sqrt( (x(i,2)-right_x).^2 + (y(i,2)-right_y) );
    normlized_dist(i) = (left_dist + right_dist) / sqrt(h^2+w^2);
        
end

display('distance between groud truth:')
display(normlized_dist)
clear all;
close all;
% this is a model of current spread from a disk electrode 
% The I(r) function was obtained by inverting the relationship between threshold and distance
% from the edge of a 200-?m diameter platinum disc electrode (previously reported in Ahuja et al.36). 

cd('C:\Users\Ione Fine\Documents\Work\Science\Projects\Ione Fine\ProstheticModel');
%% Set up array
a.ahuja=[23.8 .0017 1.69];
a.radius=[130 260];
a.separation=800; % distance between the centers of each electrodes, microns
box=800; % the length and width of the box in which each electrode is simulated, microns
a.nelectrodes=101; % square array configuration
a.dsfac=50;
[x,y] = meshgrid(-box:1:box);  %microns
rad =sqrt(x.^2+y.^2);
% create each electrode
disp('creating the electrode array');
distlist=[0];
for dd=1:length(distlist)
for rr=1:length(a.radius) % for each electrode size
    ee=zeros(size(rad));
    ee(rad<=a.radius(rr))=1;  
    ee(rad>a.radius(rr))=a.ahuja(1)./ ...
        (a.ahuja(1)+(a.ahuja(2)*(rad(rad>a.radius(rr))-a.radius(rr)).^a.ahuja(3)));
    ee=ee.*a.ahuja(1)./ ...
        (a.ahuja(1)+(a.ahuja(2)*distlist(dd).^a.ahuja(3)));
    % Threshold current versus distance from the edge of a stimulating
    % electrode taken from Ahuja et al 2012
   subplot(2, 4, dd)
   image(ee*255); axis equal ; colormap(gray(256))
   maxval(dd)=max(ee(:));
axis off; 
end 
end
%plot(distlist, 1./maxval, '-k.', 'MarkerSize', 30)



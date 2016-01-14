% this is the beginning of a re-write of the model done in England. 
% worth quickly looking at and discussing discrepancies with Ione

clear all;
close all;

cd('C:\Users\Ione Fine\Documents\Work\Science\Projects\Ione Fine\ProstheticModel');
flag='fit'; %
threshold=1.5; % threshold for R4
% 'fit' finds the current needed to reach a fixed threshold,
% 'model' finds the brightness given the amplitude

%% Set up array
a.ahuja=[23.8 .0017 1.69];
a.radius=[130 260];
a.separation=800; % distance between the centers of each electrodes, microns
box=800; % the length and width of the box in which each electrode is simulated, microns
a.nelectrodes=2; % square array configuration
a.dsfac=50;
[x,y] = meshgrid(-box:1:box);  %microns
rad =sqrt(x.^2+y.^2);
% create each electrode
disp('creating the electrode array');
for rr=1:length(a.radius) % for each electrode size
    ee=zeros(size(rad));
    ee(rad<=a.radius(rr))=1;
    
    ee(rad>a.radius(rr))=a.ahuja(1)./ ...
        (a.ahuja(1)+(a.ahuja(2)*(rad(rad>a.radius(rr))-a.radius(rr)).^a.ahuja(3)));
    
    % Threshold current versus distance from the edge of a stimulating
    % electrode taken from Ahuja et al 2012
    electrode(rr).e=ee;
end

% tile the electrodes
stpt=1:a.separation:max([a.nelectrodes a.nelectrodes])*a.separation;
endpt=stpt+(box*2);
for ii=1:a.nelectrodes
    for jj=1:a.nelectrodes
        array=zeros(max(endpt));
        % image of single electrodes, then downsample
        array(stpt(ii):endpt(ii), stpt(jj):endpt(jj))= electrode(mod(ii+jj, 2)+1).e;
        array_rs=array(1:a.dsfac:end, 1:a.dsfac:end);
        A(sub2ind([a.nelectrodes,a.nelectrodes], ii, jj),:)=array_rs(:);
    end
end

% % visualize the array
% figure(1); clf
% for aa=1:size(A, 1)
%     imagesc(reshape(A(aa, :), size(array_rs))); colormap(gray(256)); axis off; axis equal; drawnow;
%
% end
% imagesc(reshape(mean(A, 1), size(array_rs))); colormap(gray(256)); axis off; axis equal; drawnow;

% A is a nelectrodes x spatial resolution array showing the current
% spread for each electrode


%% Stimulation pulse trains
% time is expressed in seconds.
disp('creating the stimulation protocol')
Exp='paired';
switch Exp
    case 'amplitude'
        aList=[30.0000   37.5000   45.0000   60.0000  120.0000  180.0000];
        fList=[20*ones(size(aList))];
        for aa=1:length(aList)
            stim(aa).electref=1; % stimulating electrode 1
            stim(aa).tsample =.01/1000*9; % in secs
            stim(aa).dur = .5;
            stim(aa).gapdur=0; % in between cathodic and anodic
            stim(aa).type='biphasic cathodic';
            stim(aa).pulsedur=.45/1000;
            stim(aa).amp=aList(aa);
            stim(aa).freq=fList(aa);
            stim(aa).I=zeros(a.nelectrodes*a.nelectrodes, round(stim(aa).dur./stim(aa).tsample));
            stim(aa).I(stim(aa).electref, :)=MakePulseTrain(stim(aa));
            stim(aa).IA=stim(aa).I'*A;
        end
    case 'frequency'
        fList=[13.3333 20 26.6667 40 80 120];
        aList=37.5 *ones(size(fList));
        for aa=1:length(aList)
            stim(aa).electref=1; % stimulating electrode 1
            stim(aa).tsample =  .01/1000*9;
            stim(aa).dur = .5;
            stim(aa).gapdur=0/1000;  % in between cathodic and anodic
            stim(aa).type='biphasic cathodic';
            stim(aa).pulsedur=.45/1000;
            stim(aa).amp=aList(aa); stim(aa).freq=fList(aa);
            stim(aa).I=zeros(a.nelectrodes*a.nelectrodes, round(stim(aa).dur./stim(aa).tsample));
            stim(aa).I(stim(aa).electref, :)=MakePulseTrain(stim(aa));
            stim(aa).IA=stim(aa).I'*A;
        end
    case 'paired'
        aList1=linspace(15, 20, 5);
        aList2=[10 10 10 10 10];
        phList2=[ 0 .075 .375 .5 1 1.5 1.8]./1000;
        for aa1=1:length(aList1)
            for aa2=1:length(aList2)
                stim(aa1,aa2).freq=50;
                stim(aa1,aa2).electref1=1; % stimulating electrode 1
                stim(aa1,aa2).electref2=2; % stimulating electrode 2
                stim(aa1,aa2).tsample =  .01/1000*9;
                stim(aa1,aa2).dur = .5;
                stim(aa1,aa2).gapdur=.45/1000;  % in between cathodic and anodic
                stim(aa1,aa2).type='biphasic cathodic';
                stim(aa1,aa2).pulsedur=.45/1000;
                % stimulation on the first electrode
                stim(aa1, aa2).phdelay=0;
                stim(aa1,aa2).I=zeros(a.nelectrodes*a.nelectrodes, round(stim(aa1, aa2).dur./...
                    stim(aa1, aa2).tsample));
                stim(aa1,aa2).amp=aList1(aa1);
                stim(aa1,aa2).I(stim(aa1, aa2).electref1, :)=MakePulseTrain(stim(aa1, aa2));
                % stimulation on the second electrode
                stim(aa1,aa2).amp=aList2(aa2);
                stim(aa1, aa2).phdelay=phList2(aa2);
                stim(aa1,aa2).I(stim(aa1, aa2).electref2, :)=MakePulseTrain(stim(aa1, aa2));
                stim(aa1,aa2).IA=zeros(size(A, 2), size(stim(aa1, aa2).I,2));
                stim(aa1,aa2).ampRec=[aList1(aa1) aList2(aa2)];
                for ee=1:a.nelectrodes^2
                    disp(['calculating stimulation for electrode ', num2str(ee)])
                    stim(aa1,aa2).IA = stim(aa1,aa2).IA+(A(ee,:)'*stim(aa1,aa2).I(ee,:));
                end
            end
        end
    case 'movies'
        flist={'clips\body_02.mp4'};
        for aa=1:length(flist); % for each movie
            stim(aa).tsample =.01/1000*9;
            
            stim(aa).gapdur=0/1000;  % in between cathodic and anodic
            stim(aa).type='movie amp biphasic cathodic';
            stim(aa).pulsedur=.45/1000;
            stim(aa).freq=20;
            xyloObj = VideoReader(flist{aa});
            stim(aa).dur = .2;% xyloObj.Duration;
            stim(aa).I=zeros(a.nelectrodes*a.nelectrodes, round(stim(aa).dur./stim(aa).tsample))
            vidFrames = read(xyloObj);
            sampFac=floor(size(vidFrames, 1)./a.nelectrodes);
            r1=[ 1 1 size(vidFrames, 2) size(vidFrames, 1)];
            r2=[1 1 a.nelectrodes*sampFac a.nelectrodes*sampFac];
            [rect,dh,dv] = CenterRect(r2,r1);
            stim(aa).vidFrames=double(squeeze(vidFrames(rect(2):rect(4), rect(1):rect(3), 1,:)));
            for ii=1:a.nelectrodes
                for jj=1:a.nelectrodes
                    tmp = zeros(size(stim(aa).vidFrames,1),size(stim(aa).vidFrames,2));
                    tmp(((ii-1)*sampFac+1):ii*sampFac,((jj-1)*sampFac+1):jj*sampFac)=1;
                    ind = find(tmp);
                    for t=1:size(stim(aa).vidFrames, 3)
                        tmp=stim(aa).vidFrames(:, :, t); tmp=tmp(:);
                        lv(t)=mean(tmp(ind));
                    end
                    stim(aa).lumVec(sub2ind([a.nelectrodes,a.nelectrodes], ii, jj), :)=scaleif(lv, 10, 40);
                end
            end
            stim(aa).I=MakePulseTrain(stim(aa));
            disp('finding the current over time for each electrode');
            stim(aa).IA=zeros(size(A, 2), size(stim(aa).I,2));
            for ee=1:a.nelectrodes^2
                disp(['calculating stimulation for electrode ', num2str(ee)])
                stim(aa).IA = stim(aa).IA+(A(ee,:)'*stim(aa).I(ee,:));
            end
            
        end
    otherwise
        disp('Experiment not defined');
    return
end
% stim.IA is the current over the whole array, over time


%% Model Parameters - Threshold
p.tau1 = .42/1000;  %.24 - .65
p.tau2 = 45.25/1000;  %38-57
p.tau3 =  26.25/1000; % 24-33
p.e = 2.25;  %2-3 for threshold or 8-10 for suprathreshold

p.asymptote=14;
p.slope=.3;
p.shift=47;
%% create 1-D filters
% fast impulse response function
p.t=stim(1).tsample:stim(1).tsample:stim(1).dur;
t1 = stim(1).tsample:stim(1).tsample:.005;
p.G1 = LeakyIntegrator(1, p.tau1, t1);
% create the slow impulse response and convolution
t2 = 0:stim(1).tsample:.3;
p.G2 = LeakyIntegrator(1, p.tau2, t2);
% create the final slow low pass filter
t3 = 0:stim(1).tsample:.4;
p.G3 = LeakyIntegrator(3,p.tau3,t3);

%% loop through all stimulation conditions (amp and freq coding)
% what we do is calculate the response for each possible value of beta
% we and then store for each condition:
% the final outputs for each possible beta  - I(c).img1D
% the max (over time) R2 value over space - I(c).R2space

if strcmp(flag, 'model') % just find the brightness for each stimulus
    plotscfac=1000;
    for aa1=1:size(stim, 1)
        disp(['processing condition 1 - ', num2str(aa1)]);
        for aa2=1:size(stim, 2)
            disp(['processing condition 2 - ', num2str(aa2)]);
            [s] = NLmodel(p,stim(aa1, aa2));
            bright(aa1,aa2)=max(s.R4(:));
        end
    end
else %fitting to find a threshold
    for aa1=1:size(stim, 1)
        disp(['processing condition 1 - ', num2str(aa1)]);
        for aa2=1:size(stim, 2)
            disp(['processing condition 2 - ', num2str(aa2)]);        
            lo = 0; hi = 30;     
            for i=1:10
                mid = (hi+lo)/2;
                stim(aa1,aa2).amp = mid;
                stim(aa1,aa2).I(stim(aa1, aa2).electref2, :)=MakePulseTrain(stim(aa1, aa2));
                stim(aa1,aa2).IA=zeros(size(A, 2), size(stim(aa1, aa2).I,2));
                stim(aa1,aa2).ampRec=[aList1(aa1) aList2(aa2)];
                for ee=1:a.nelectrodes^2
                    stim(aa1,aa2).IA = stim(aa1,aa2).IA+(A(ee,:)'*stim(aa1,aa2).I(ee,:));
                end
                [s] = NLmodel(p,stim(aa1, aa2));
                out=max(s.R4(:));
                if out<threshold
                    lo = mid;
                else
                    hi = mid;
                end
                disp(sprintf('Iteration %d, amp = %f, out = %f',i,mid,out));
            end
            stim(aa1,aa2).amp = (hi+lo)/2;  
        end
    end
end

save('allmystuff_pairedStimulation')
for aa2=1:length(aList2)
    gamma=(17.9443-(aList1+aList2(aa2)))./aList1.*aList2(aa2)
    plot(aList1, [cat(1, stim(:, aa2).amp)],...
        '-.', 'MarkerSize', 20); hold on;
end

% calculate nonlinearity as a function of amplitude on the retina (R2)
% 13 is the max R2 value for threshold, 78 the max R2 value for the highest
% amplitude stimulus

p_t.ia=[0 exp(linspace(log(8), log(18), 100))];
p_t.slope=.3; % larger number = shallower slope
thresholdR2=13.1; % R2 at threshold, need the nonlinearity to be near zero here
threshold1_5R2=16.37; % R2 at 1.5x threshold. Need the nonlinear asymptote (p_t.asymptote) to be
% close to this for the brightness data to asymptote at lower brightness
% levels than the frequency data
p_t.asymptote=14;

p_t.shift=47; % shifts curve along x-axis
p_t.ia_out=p_t.asymptote./(1+exp(-(p_t.ia./p_t.slope)+p_t.shift));
% this is how the current is scaled inside the alan model function

if 1
    figure(10); clf
    plot(p_t.ia, p_t.ia_out, '-r.', 'MarkerSize', 5); hold on
    plot([thresholdR2 thresholdR2 ], [ 0 50], 'k--');
    plot([threshold1_5R2 threshold1_5R2 ], [ 0 50], 'k-.');
    xlabel('amplitude');
    ylabel('output');
end


%% plot final output
figure(13); clf
for c=1:size(STIM, 2)
    %make final 2-D image from 1-D image
    finalimage=1000*interp1(r, I(c).img1D, rad);
    subplot(2, 6, c)
    image(finalimage); hold on
    axis square
    axis off
    title(sprintf('%s%s%s%s',num2str(conditions(1,c)),'xTh/',...
        num2str(round(conditions(2,c))), 'Hz'));
    maxval(c)=max(I(c).img1D(:));
    text(10,300,num2str(100*maxval, 2));
    colormap(hot(256))
end
set(gcf, 'Name', 'final output')


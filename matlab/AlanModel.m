%function img=AlanModel(p,STIM,x,y)

%predicts stimulation data based on the following four stages:
%
%1. fast leaky integrator (time constant around .3 ms)
%2. slow decay in response over time (temporal uncertainty,...,attention?)
%3. rectification (set all values below zero to zero)
%4. probability summation over time
%
%Model parameters:
%
%p.nG       number of integrators in the ganglion cell cascade (always set to 1)
%p.tauG     time constant of ganglion cell leaky integrator (ms)
%p.beta     probability summation parameter
%p.thresh   threshold voltage
%
%11/8/05 if,ah,gmb

%stimulus parameters
STIM.electrodeRAD = 260;
%STIM.freq = 20;  %Hz
STIM.dur = 1;
STIM.pulsedur = .45/1000;  %sec
STIM.tsample = .01/1000;

t = 0:STIM.tsample:.5;

%% model parameters

p.tau1 = .42/1000;  %.24 - .65
p.tau2 = 45.25/1000;  %38-57
p.tau3 =  26.25/1000; % 24-33
p.e = 8.73;  %2-3 for threshold or 8-10 for suprathreshold
% p.beta = .6;  %3-4.2 for threshold or .6-1 for suprathreshold

p.beta = 3.5;


%%

%make cspread
[x,y] = meshgrid(-1000:25:1000);  %microns
rad = sqrt(x.^2+y.^2);
cspread = ones(size(x));
cspread(rad>STIM.electrodeRAD) = 2/pi*(asin(STIM.electrodeRAD./rad(rad>STIM.electrodeRAD)));

%%

freqList = [10,20,40,80,160 10,20,40,80,160];
STIM.amp = [30 30 30 30 30 90 90 90 90 90];   %30 uA
clear R
for freqNum = 1:length(freqList)
    STIM.freq = freqList(freqNum);

    sawtooth = STIM.freq*mod(t,1/STIM.freq);

    on  = sawtooth > STIM.pulsedur*STIM.freq & sawtooth < 2*STIM.pulsedur*STIM.freq;
    off = sawtooth < STIM.pulsedur*STIM.freq;
    tsform = on-off;

    figure(1)
    clf
    subplot(5,1,1);
    plot(t,tsform,'b-');title('tsform')


    %% Alan's model (with space)
    t1 = 0:STIM.tsample:.005;
    G1 = Gamma(1, p.tau1, t1); % fast impulse response function

    %convolve the stimulus with the ganglion cell impulse response
    R1 = STIM.tsample.*conv(G1,tsform);
    R1 = STIM.amp(freqNum).*R1(1:length(t)); %cut off end due to convolution

    subplot(5,1,2)
    plot(t,R1);title('R1');

    %%

    % create the slow impulse response and convolution
    t2 = 0:STIM.tsample:.30;
    G2 = Gamma(1, p.tau2, t2);

    % total charge in the system as a function of time
    chargeacc=STIM.tsample*cumsum(max(tsform,0));

    tmp = p.e*STIM.tsample*conv(G2,chargeacc);
    tmp = tmp(1:length(t));


    R2 =R1- tmp;

    subplot(5,1,3)
    plot(t,R2);title('R2')

    %%
    R3 = max(R2,0).^p.beta;
    subplot(5,1,4)

    plot(t,R3);title('R3');


    %%
    t3 = 0:STIM.tsample:p.tau3*10;

    G3 = Gamma(3,p.tau3,t3);

    R4 =  STIM.tsample*conv(G3,R3);


    %(STIM.amp*STIM.tsample(:)).^p.beta;


    R4 = R4(1:length(t));
    subplot(5,1,5)

    plot(t,R4);title('R4');

    R(freqNum) = max(R4);
    freqNum
end
close all
plot(freqList, R)

%%
% 
% ampList = 2000./fliplr(freqList);
% figure(2)
% clf
% 
% scfac = 256/(R(end)*(ampList(end)').^p.beta);
% count =0;
% for i=1:length(ampList)
%     for j=1:length(freqList)
%         img = R(j)*(ampList(i)*cspread).^p.beta;
% 
%         count = count+1;
%         subplot(length(ampList),length(freqList),count);
%         image(img*scfac)
%         colormap(gray(256));
%         axis equal
%         axis off
%         title(sprintf('%5.0f Hz, %5.0f uA',freqList(j),ampList(i)));
%     end
% end
% %%

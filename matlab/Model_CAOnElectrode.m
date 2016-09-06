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
clear all
close all
%stimulus parameters
STIM.electrodeRAD = 260;
%STIM.freq = 20;  %Hz
STIM.dur = .5;
STIM.pulsedur = .45/1000;  %sec
STIM.tsample = .01/1000;

STIM.t = 0:STIM.tsample:STIM.dur;

%% model parameters

p.tau1 = .42/1000;  %.24 - .65
p.tau2 = 45.25/1000;  %38-57
p.tau2k = 45.25/1000;  %38-57
p.tau3 =  26.25/1000; % 24-33
p.e = 8.73;  %2-3 for threshold or 8-10 for suprathreshold
p.ek = 8.73;  %2-3 for threshold or 8-10 for suprathreshold

p.asymptote=14;
p.slope=3;
p.shift=16;

%%

%plot stuff
colorlist=hsv(20);
figure(1); clf
%%

%make cspread
[x,y] = meshgrid(-1000:25:1000);  %microns
rad = sqrt(x.^2+y.^2);
cspread = ones(size(x));
cspread(rad>STIM.electrodeRAD) = 2/pi*(asin(STIM.electrodeRAD./rad(rad>STIM.electrodeRAD)));

% create gammas
t1 = 0:STIM.tsample:.05;
p.G1 = Gamma(1, p.tau1, t1); % fast impulse response function

t2 = 0:STIM.tsample:.30;
p.G2 = Gamma(1, p.tau2, t2);

t3 = 0:STIM.tsample:p.tau3*10;
p.G3 = Gamma(3,p.tau3,t3);


STIM.freqList =[10 10 30 30 80 80]; %[10 20 40 80 160 10 20 40 80 160  10 20 40 80 160];
STIM.amp = [30 90 30 90 30 90]; %[30 30 30 30 30 60 60 60 60 60 90 90 90 90 90];   %30 uA
clear R
[p,err] = fit('fitKrishnan2Nanduri',p,{'ek'}, STIM);

for freqNum =1:length(STIM.freqList)
    Rn=ModelNanduri(p, STIM, freqNum);
    Rk=ModelKrishnan(p, STIM, freqNum);
    
    plot(STIM.t, Rn, '-.', 'Color', colorlist(freqNum, :)); title('on electrode');  hold on 
    plot(STIM.t, Rk, '--', 'Color', colorlist(freqNum, :)); title('on electrode');  hold on
    drawnow;
end
%
% for freqNum = 1:length(freqList)
%     % stimulus
%     STIM.freq = freqList(freqNum);
%     sawtooth = STIM.freq*mod(STIM.t,1/STIM.freq);
%     on  = sawtooth > STIM.pulsedur*STIM.freq & sawtooth < 2*STIM.pulsedur*STIM.freq;
%     off = sawtooth < STIM.pulsedur*STIM.freq;
%     STIM.tsform = on-off;
%
%     % total charge in the system as a function of time
%     chargeacc=STIM.tsample*cumsum(max(STIM.tsform,0));
%     tmp = p.e*STIM.tsample*conv(p.G2,chargeacc);
%     STIM.chargeacc = tmp(1:length(STIM.t));
%
%     % model nanduri
%     R(freqNum).Rn=ModelNanduri(p, STIM, freqNum);
% end
%
%     R(freqNum).Rk=ModelKrishnan(p, STIM, freqNum);
%
%
%
% end

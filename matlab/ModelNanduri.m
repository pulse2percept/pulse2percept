function R4=ModelNanduri(p, STIM, freqNum)
if nargin<3,freqNum=1;end

%% generate pulse train
STIM.t = 0:STIM.tsample:STIM.dur;
STIM.freq = STIM.freqList(freqNum);
pulse = getSquarePulse(STIM.t, STIM.freq, STIM.ampList(freqNum), ...
    STIM.pulsedur);


%% fast response
%convolve the stimulus with the ganglion cell impulse response
t1 = 0:STIM.tsample:.005;
G1 = Gamma(1, p.tau1, t1);
R1 = STIM.tsample.*conv(G1,pulse);
R1 = R1(1:length(STIM.t));

%% charge accumulation
% total charge in the system as a function of time
chargeacc=STIM.tsample*cumsum(max(pulse,0));
t2 = 0:STIM.tsample:.30;
G2 = Gamma(1, p.tau2, t2);
tmp = p.epsilon * STIM.tsample * conv(G2,chargeacc);
STIM.chargeacc = tmp(1:length(STIM.t));

R2=R1-STIM.chargeacc;

%% stationary nonlinearity
R3 = max(R2,0);
R3norm = R3 / max(R3);

scFac=p.asymptote./(1+exp(-(max(R3(:))-p.shift)./p.slope));
R3 = R3norm .* scFac ;

%% slow convolution

t3 = 0:STIM.tsample:p.tau3*10;
G3 = Gamma(3,p.tau3,t3);
R4 =  STIM.tsample*conv(G3,R3);

R4 = R4(1:length(STIM.t));

end
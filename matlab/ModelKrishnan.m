function R4=ModelKrishnan(p, STIM, freqNum)
if nargin<3,freqNum=1;end

%% generate pulse train
STIM.t = 0:STIM.tsample:STIM.dur;
STIM.freq = STIM.freqList(freqNum);
pulse = getSquarePulse(STIM.t, STIM.freq, STIM.ampList(freqNum), ...
    STIM.pulsedur);


%% charge accumulation
% total charge in the system as a function of time
t2 = 0:STIM.tsample:.30;
G2 = Gamma(1, p.tau2, t2);
chargeacc=STIM.tsample*cumsum(max(pulse,0));
tmp = p.epsilonK * STIM.tsample * conv(chargeacc,G2);
STIM.chargeacc = tmp(1:length(STIM.t));


%% fast response
% convolve the stimulus with the ganglion cell impulse response
t1 = 0:STIM.tsample:.005;
G1 = Gamma(1, p.tau1, t1);
R1 = STIM.tsample.*conv(pulse-STIM.chargeacc, G1);
R1 = R1(1:length(STIM.t)); %cut off end due to convolution

%% stationary nonlinearity
R3 = max(R1,0);
R3norm = R3 / max(R3);

scFac=p.asymptote./(1+exp(-(max(R3(:))-p.shift)./p.slope));
R3 = R3norm .* scFac ;

%% slow convolution

t3 = 0:STIM.tsample:p.tau3*10;
G3 = Gamma(3,p.tau3,t3);
R4 =  STIM.tsample*conv(R3,G3);
R4 = R4(1:length(STIM.t));

end
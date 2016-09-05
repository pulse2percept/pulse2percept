function [ err] = fitKrishnan2Nanduri( p,STIM)
% finds the parameters for the Krishnan model that most closely matches the
% Nanduri model
 
for freqNum=1:length(STIM.freqList)
        % stimulus
    STIM.freq = STIM.freqList(freqNum);
    sawtooth = STIM.freq*mod(STIM.t,1/STIM.freq);
    on  = sawtooth > STIM.pulsedur*STIM.freq & sawtooth < 2*STIM.pulsedur*STIM.freq;
    off = sawtooth < STIM.pulsedur*STIM.freq;
    STIM.tsform = on-off;
    
    % total charge in the system as a function of time
    chargeacc=STIM.tsample*cumsum(max(STIM.tsform,0));
    tmp = p.e*STIM.tsample*conv(p.G2,chargeacc);
    STIM.chargeacc = tmp(1:length(STIM.t));
    Rn(freqNum,:)=ModelNanduri(p, STIM, freqNum);
    Rk(freqNum,:)=ModelKrishnan(p, STIM, freqNum);
end
err=sum((Rn(:)-Rk(:)).^2);
disp(err)



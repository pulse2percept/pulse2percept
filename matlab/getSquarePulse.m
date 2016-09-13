function pulse = getSquarePulse(t, pulseFreq, pulseAmpl, pulseDur)
% pulse = getSquarePulse(t, pulseFreq, pulseAmpl, pulseDur) generates a
% square pulse with frequency pulseFreq, amplitude pulseAmpl, and duration
% pulseDur, evaluated at times t.
    if ~isvector(t)
        error('t must be a vector')
    end
    if ~isscalar(pulseFreq)
        error('pulseFreq must be a scalar')
    end
    if ~isscalar(pulseDur)
        error('pulseDur must be a scalar')
    end
    sawtooth = pulseFreq * mod(t, 1/pulseFreq);
    on  = sawtooth >= pulseDur * pulseFreq & ...
        sawtooth < 2 * pulseDur * pulseFreq;
    off = sawtooth < pulseDur * pulseFreq;
    pulse = pulseAmpl * (on - off);
end
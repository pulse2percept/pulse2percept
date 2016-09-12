[p,STIM] = setupModel;

freqNum = 1;
R4N = ModelNanduri(p,STIM,freqNum);
R4K = ModelKrishnan(p,STIM,freqNum);

plot(R4N, '-b', 'LineWidth', 6)
hold on
plot(R4K, '-y', 'LineWidth', 2)
xlabel('time (s)')
ylabel('R4')
legend('Nanduri','Krishnan')

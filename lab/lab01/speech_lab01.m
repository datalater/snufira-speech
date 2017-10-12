%##f file read - play sound
[y, Fs] = audioread('she.wav'); % Fs=sampling_frequency
% sound(y,Fs);
plot(y);
% t = linspace(0, length(y)*(1/Fs), length(y));
% plot(t, y);
% xlabel('time(s)');
% ylabel('amplitude');
% samples = [1, 3*Fs]
% [y, Fs] = audioread('she.wav', samples);
% sound(y,Fs);
% plot(y)
% sol sol la la sol sol mi
la_freq = 440; %A4
sol_freq = 392
mi_freq = 329.63

Fs = 8000; % sampling frequency
A4 = [cos(2*pi*la_freq/Fs*[0:8000])];
sol = [cos(2*pi*sol_freq/Fs*[0:8000])];
mi = [cos(2*pi*mi_freq/Fs*[0:8000])];

% sound(sol, Fs);
% pause(1)
% sound(sol, Fs);
% pause(1)
% sound(A4, Fs);
% pause(1)
% sound(A4, Fs);
% pause(1)
% sound(sol, Fs);
% pause(1)
% sound(mi, Fs);
[y,Fs] = audioread('she.wav');
X = y; % 신호의 크기
n = 5; % 차수
Wn = 1000; % 차단주파수
Fn = Fs/2;
ftype = 'high';
[b,a] = butter(n, Wn/Fn, ftype); % Wn을 Fn으로 나눠준 것은 normalizing 목적
y_f = filter(b, a, X);
figure(2), plot(y_f);
sound(y_f, Fs);
[y,Fs] = audioread('she.wav');
X = y; % ��ȣ�� ũ��
n = 5; % ����
Wn = 1000; % �������ļ�
Fn = Fs/2;
ftype = 'high';
[b,a] = butter(n, Wn/Fn, ftype); % Wn�� Fn���� ������ ���� normalizing ����
y_f = filter(b, a, X);
figure(2), plot(y_f);
sound(y_f, Fs);
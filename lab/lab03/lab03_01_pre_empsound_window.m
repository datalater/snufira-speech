[sound, rate] = audioread('wsj.wav'); 
% sound : ���õ� ���� ����
% rate : 1�ʿ� ���� ���� ����
% rate : sampling frequency
% sound = 88737, rate = 16000
% sound/rate = 0.5546
% window size : 25msec�� �ڸ���, ��ġ���� 10msec�� �̵��ذ��鼭 �ݺ�.
% 10msec : 0.01��, �� ���� 5.546��
% 5.546��/0.01 = 554 <= window�� ����
% �׷��� �Ʒ��� windowed_sound �� (400, 553)�̶�� �Ǿ� ����. 554�� �ƴ� ������ ������� ���� ��.

preempsound = filter([1 -0.97], 1, sound);
% audiosound�� �����ļ��� ������Ŵ
% ���� feature�� �̾Ƴ��� ���� �����ļ��� ������ ��

windowed_sound = zeros(400, 553);
% zeros (400, 553)�� �ǹ�
% 553���� �� = 553���� window
% 400 = window ���� 25msec
% 1�� : 16000�� = 25msec(=0.025��) : 400��

i = 1;
for n = 1:553 % window�� 1���� 553���� 
    windowed_sound(:,n) = preempsound(i:i+(rate*0.025)-1); 
    i = i+rate*0.01;
end
figure(1), plot(windowed_sound(:,1)), hold on, plot(sound(1:400), 'r'), hold off;
% windowed_sound : ������ ��ȣ
% sound : original ��ȣ

% for�� ����
% windowed_sound(:,n) : ���� ���, n=1�� ��, ��� �� 400���� �� �����Ѵ�.
% windowed_sound(:,n) = preempsound(i:i+(rate*0.025)-1); : i=1�� ��
% 1:1+(16000 * 0.025) => 1:1+(400) = > ù��° ���� 400���� ���� �ִ´�.
% i=553���� �ݺ��Ѵ�.

% 


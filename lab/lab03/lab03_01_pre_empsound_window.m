[sound, rate] = audioread('wsj.wav'); 
% sound : 샘플된 점의 개수
% rate : 1초에 찍은 점의 개수
% rate : sampling frequency
% sound = 88737, rate = 16000
% sound/rate = 0.5546
% window size : 25msec씩 자른다, 겹치도록 10msec씩 이동해가면서 반복.
% 10msec : 0.01초, 총 길이 5.546초
% 5.546초/0.01 = 554 <= window의 개수
% 그래서 아래에 windowed_sound 에 (400, 553)이라고 되어 있음. 554가 아닌 이유는 오차라고 보면 됨.

preempsound = filter([1 -0.97], 1, sound);
% audiosound의 고주파수를 증폭시킴
% 좋은 feature를 뽑아내기 위해 저주파수를 제거한 것

windowed_sound = zeros(400, 553);
% zeros (400, 553)의 의미
% 553개의 열 = 553개의 window
% 400 = window 구간 25msec
% 1초 : 16000번 = 25msec(=0.025초) : 400번

i = 1;
for n = 1:553 % window를 1부터 553까지 
    windowed_sound(:,n) = preempsound(i:i+(rate*0.025)-1); 
    i = i+rate*0.01;
end
figure(1), plot(windowed_sound(:,1)), hold on, plot(sound(1:400), 'r'), hold off;
% windowed_sound : 증폭된 신호
% sound : original 신호

% for문 설명
% windowed_sound(:,n) : 예를 들어, n=1일 때, 모든 행 400개를 다 포함한다.
% windowed_sound(:,n) = preempsound(i:i+(rate*0.025)-1); : i=1일 때
% 1:1+(16000 * 0.025) => 1:1+(400) = > 첫번째 열에 400개의 점을 넣는다.
% i=553까지 반복한다.

% 


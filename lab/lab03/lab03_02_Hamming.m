wHamm = hamming(length(windowed_sound(:,1)));
% 400의 크기를 가진 window를 만들어준다.
% 이전에 만든 window의 크기가 400이었으므로.

Hammed = zeros(400, 553);
for n = 1:553
    Hammed(:,n) = wHamm.*windowed_sound(:,n);
end
figure(2), plot(Hammed(:,1),'r'), hold on, plot(windowed_sound(:,n)), hold off;
% plot(Hammed(:,1),'r') : 첫번째 성분(칼럼, 윈도우)만 뽑아낸다.

% Hamming Window : 기존 윈도우에 새로운 윈도우를 곱해준다.
% 경계에서 끊어지는 부분을 0으로 줄여주고 가운데 성분만 더 드러내는 연산
% 장점 : 중심으로 집중이 되면 훈련이 더 좋아진다. (실험 결과)

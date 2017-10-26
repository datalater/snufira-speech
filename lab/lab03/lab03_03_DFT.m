%fft
fftsound = zeros(512, 553);
for n = 1:553
    fftsound(:, n) = fft(Hammed(:, n), 512);
end
%f = rate*(1:512);
figure(3), stem(fftsound(:, 1))


ensound = zeros(512, 553);
for n = 1:553
    ensound(:, n) = abs(fftsound(:, n)).^2;
end
figure(4), stem(ensound(:, 1))

% DFT : 시간축을 주파수축으로 바꾸는 것
% fft : dft해주는 함수
% 점(N) 512개 사용

% 전체 과정 속 의미
% pre-emphasis와 window 이후 주파수 처리하기 위한 작업

% 이전 파일에서 window 행 사이즈가 400이었다. (400,553)
% 그런데 (512, 533)으로 바뀌었다.
% DFT의 정의에 따라 400개의 점이 
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

% DFT : �ð����� ���ļ������� �ٲٴ� ��
% fft : dft���ִ� �Լ�
% ��(N) 512�� ���

% ��ü ���� �� �ǹ�
% pre-emphasis�� window ���� ���ļ� ó���ϱ� ���� �۾�

% ���� ���Ͽ��� window �� ����� 400�̾���. (400,553)
% �׷��� (512, 533)���� �ٲ����.
% DFT�� ���ǿ� ���� 400���� ���� 
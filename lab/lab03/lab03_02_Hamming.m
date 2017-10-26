wHamm = hamming(length(windowed_sound(:,1)));
% 400�� ũ�⸦ ���� window�� ������ش�.
% ������ ���� window�� ũ�Ⱑ 400�̾����Ƿ�.

Hammed = zeros(400, 553);
for n = 1:553
    Hammed(:,n) = wHamm.*windowed_sound(:,n);
end
figure(2), plot(Hammed(:,1),'r'), hold on, plot(windowed_sound(:,n)), hold off;
% plot(Hammed(:,1),'r') : ù��° ����(Į��, ������)�� �̾Ƴ���.

% Hamming Window : ���� �����쿡 ���ο� �����츦 �����ش�.
% ��迡�� �������� �κ��� 0���� �ٿ��ְ� ��� ���и� �� �巯���� ����
% ���� : �߽����� ������ �Ǹ� �Ʒ��� �� ��������. (���� ���)

fb = load('fb.mat');
fb = struct2array(fb);
fb_sound = zeros(26, 553);
for n = 1:553
    for i = 1:26
        fb_sound(i,n) = log(sum(ensound(1:257,n)'.*fb(i,:)));
    end
end
figure(5), stem(fb_sound(:,1))

inverse_sound = zeros(26,553);
for n = 1:553
inverse_sound(:,n) = ifft(fb_sound(:,n));
end
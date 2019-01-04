clear,close all,clc

%% Unscented transform plot

mu = [0 0];
Sigma = [2 1.5; 1.5 2];
x1 = -3:.05:3; x2 = -3:.05:3;
[X1,X2] = meshgrid(x1,x2);
F = mvnpdf([X1(:) X2(:)],mu,Sigma);
F = reshape(F,length(x2),length(x1));
X = [mu' mu'+chol(Sigma,'lower') mu'-chol(Sigma,'lower')];

figure
hold on
contour(x1,x2,F,5,'linewidth',2)
plot(X(1,:),X(2,:),'.b','MarkerSize',20)
axis equal

matlab2tikz('filename','utp.tikz',...
            'height','0.4\linewidth',...
            'width','0.7\linewidth')
        
close
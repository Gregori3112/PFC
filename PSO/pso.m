clc;clear all;close all;
xmin = [-10 -10];
xmax = [10 10];
nrvar = length(xmin);
lambda1 = 2.02;
lambda2 = 2.02;
omega = 0.4;
pop = 5;
tol = 1E-7;
itermax=50;
rng(2);

[X1,X2] = meshgrid(xmin(1):0.2:xmax(1),xmin(2):0.2:xmax(2));
for i=1:length(X1)
    for j=1:length(X2)
        Z(i,j)=FCN([X1(i,j);X2(i,j)]);
    end
end
figure(1)

mesh(X1,X2,Z);hold on;
view(140,45)

gbest = 1E30;
k=1;
v = zeros(pop,nrvar);
for i = 1:pop
    for j = 1:nrvar
        x(i,j) = xmin(j) + (xmax(j) - xmin(j))*rand;
        %x(i,j) = -3 + 1*rand;
    end
    y = FCN(x(i,:));
    lbest(i) = y;
    xlbest(i,:) = x(i,:);
    if (y < gbest)
        gbest(k) = y;
        xgbest = x(i,:);
    end
end
%Figura 1 (localizacao do melhor individuo na primeira iteracao)
 figure(1)
 plot3(xgbest(1),xgbest(2),gbest(k),'ro','MarkerSize',13,'MarkerFaceColor','r');hold on

flag = 0;
k = 2;
while(flag==0)
    gbest(k) = gbest(k-1);
    for i = 1:pop
        for j = 1:nrvar
            r1 = rand;
            r2 = rand;
            vnew(i,j) = omega*v(i,j) + lambda1*r1*(xlbest(i,j) - x(i,j)) + lambda2*r2*(xgbest(j) - x(i,j));
            xnew(i,j) = x(i,j) + vnew(i,j);
            if (xnew(i,j) < xmin(j))
                xnew(i,j) = xmin(j);
            elseif (xnew(i,j) > xmax(j))
                xnew(i,j) = xmax(j);
            end
        end
        ynew = FCN(xnew(i,:));

            %Figura 2 (todas particulas por itera��o)
%            figure(2)
%            if (rem(k,5)==0)
%             plot(k,ynew,'b*');hold on
%            end

           %Figura 4 (percurso da part�cula)
           figure(4)
           zz=1;
           if(i==zz||i==zz+1||i==zz+2)
               if (i==zz)
                   z1(1:2,k-1) = [xnew(zz,1),xnew(zz,2)];
                    plot(z1(1,:),z1(2,:),'b');hold on;
               elseif (i==zz+1)
                   z2(1:2,k-1) = [xnew(zz+1,1),xnew(zz+1,2)];
                    plot(z2(1,:),z2(2,:),'r');hold on;
               elseif (i==zz+2)
                   z3(1:2,k-1) = [xnew(zz+2,1),xnew(zz+2,2)];
                   plot(z3(1,:),z3(2,:),'g');hold on;
               end
               axis([xmin(1) xmax(1) xmin(2) xmax(2)])
               axis('equal')
               %plot(xnew(zz,1),xnew(zz,2),'b*');hold on
           end

        if (ynew < lbest(i))
            lbest(i) = ynew;
            xlbest(i,:) = xnew(i,:);
        end
        if (ynew < gbest(k))
            gbest(k) = ynew;
            xgbest = xnew(i,:);
        end
    end

    %figura 3 (plot do melhor resultado por iteracao e localizacao do
    %melhor resultado por iteracao)
    figure(3)
    subplot(2,1,1)
    plot(k,gbest(k),'b*');hold on;
    axis([0 Inf 0 Inf])

    subplot(2,1,2)
    plot(xgbest(1),xgbest(2),'b*');hold on;
    plot(1,1,'ro')
    axis([xmin(1) xmax(1) xmin(2) xmax(2)])


    if (gbest(k)<gbest(k-1))
        %Figura 1 (localizacao do melhor individuo por iteracao, caso encontre um melhor)
        figure(1)
        plot3(xgbest(1),xgbest(2),gbest(k),'ro','MarkerSize',9,'MarkerFaceColor','r');hold on
    end

    if (k >= itermax)
        flag = 1;
    end
    if (k>11)
        norm = sum(gbest(k-9:k-5)) - sum(gbest(k-4:k));
        if (norm < tol)
            %flag = 1;
        end
    end
    k = k+1;
    x = xnew;
    v = vnew;

end
disp('k = ')
disp(k-1)
disp('norm = ')
disp(norm)
disp('gbest = ')
disp(gbest(k-1))
disp('xgbest = ')
disp(xgbest)


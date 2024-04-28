function classification_NN_2ex
%%%%%%% DATA %%%%%%%%%%%
% xcoords, ycoords, targets
x1 = [0.1,0.3,0.1,0.6,0.4,0.6,0.5,0.9,0.4,0.7];
x2 = [0.1,0.4,0.5,0.9,0.2,0.3,0.6,0.2,0.4,0.6];
y = [ones(1,5) zeros(1,5); zeros(1,5) ones(1,5)];

figure(1)
clf
f1 = subplot(1,1,1);
plot(x1(1:5),x2(1:5),'ro','MarkerSize',12,'LineWidth',4)
hold on
plot(x1(6:10),x2(6:10),'bx','MarkerSize',12,'LineWidth',4)
f1.XTick = [0 1];
f1.YTick = [0 1];
f1.FontWeight = 'Bold';
f1.FontSize = 16;
xlim([0,1])
ylim([0,1])

%print -dpng pic_xy.png

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize weights and biases 
rng(5000);
W2 = 0.5*randn(2,2);
W3 = 0.5*randn(3,2);
W4 = 0.5*randn(2,3);
b2 = 0.5*randn(2,1);
b3 = 0.5*randn(3,1);
b4 = 0.5*randn(2,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Forward and Back propagate 
% Pick a training point at random
eta = 0.4;%大小？否则不收敛 0.05
rho=0.9;
epsilon=1e-8;
beta1=0.9;
beta2=0.999;

Niter = 1e6;
savecost = zeros(Niter,1);



    %AdapGrad
    sum_gw2 = zeros(2,2);
    sum_gw3 = zeros(3,2);
    sum_gw4 = zeros(2,3);
    sum_gb2 = zeros(2,1);
    sum_gb3 = zeros(3,1);
    sum_gb4 = zeros(2,1);



for counter = 1:Niter
    k = randi(10);                  %随机梯度
    x = [x1(k); x2(k)];
    % Forward pass
         z2=W2*x+b2;%2nd layer
         a2 = activate(z2);
         z3=W3*a2+b3;%3rd layer
         a3 = activate(z3);
         z4=W4*a3+b4;%4th layer（output）
         a4 = activate(z4);
    % Backward pass
    delta4 = diag(grad_activate(a4))*(a4-y(:,k));
    delta3 = diag(grad_activate(a3))*(W4'*delta4);
    delta2 = diag(grad_activate(a2))*(W3'*delta3);

    % Gradient step
    gw2=delta2*x';
    gw3=delta3*a2';
    gw4=delta4*a3';



      %%%RMSprop
    sum_gw2 = rho*sum_gw2+(1-rho)*gw2.*gw2;
    sum_gw3 = rho*sum_gw3+(1-rho)*gw3.*gw3;
    sum_gw4 = rho*sum_gw4+(1-rho)*gw4.*gw4;
    sum_gb2 = rho*sum_gb2+(1-rho)*delta2.*delta2;
    sum_gb3 = rho*sum_gb3+(1-rho)*delta3.*delta3;
    sum_gb4 = rho*sum_gb4+(1-rho)*delta4.*delta4;


    W2 = W2 - eta./(sqrt(sum_gw2+epsilon)).*gw2;
    W3 = W3 - eta./(sqrt(sum_gw3+epsilon)).*gw3;
    W4 = W4 - eta./(sqrt(sum_gw4+epsilon)).*gw4;
    b2 = b2 - eta./(sqrt(sum_gb2+epsilon)).*delta2;
    b3 = b3 - eta./(sqrt(sum_gb3+epsilon)).*delta3;
    b4 = b4 - eta./(sqrt(sum_gb4+epsilon)).*delta4;




    s_gw2 = zeros(2,2);
    s_gw3 = zeros(3,2);
    s_gw4 = zeros(2,3);
    s_gb2 = zeros(2,1);
    s_gb3 = zeros(3,1);
    s_gb4 = zeros(2,1);

     % g^2
%       %%%ADAM
%     s_gw2 = beta1*s_gw2+(1-beta1)*gw2;
%     s_gw3 = beta1*s_gw3+(1-beta1)*gw3;
%     s_gw4 = beta1*s_gw4+(1-beta1)*gw4;
%     s_gb2 = beta1*s_gb2+(1-beta1)*delta2;
%     s_gb3 = beta1*s_gb3+(1-beta1)*delta3;
%     s_gb4 = beta1*s_gb4+(1-beta1)*delta4;
% 
%     sum_gw2 = beta2*sum_gw2+(1-beta2)*gw2.*gw2;
%     sum_gw3 = beta2*sum_gw3+(1-beta2)*gw3.*gw3;
%     sum_gw4 = beta2*sum_gw4+(1-beta2)*gw4.*gw4;
%     sum_gb2 = beta2*sum_gb2+(1-beta2)*delta2.*delta2;
%     sum_gb3 = beta2*sum_gb3+(1-beta2)*delta3.*delta3;
%     sum_gb4 = beta2*sum_gb4+(1-beta2)*delta4.*delta4;
% 
%     mw2=s_gw2/(1-beta1^counter);
%     mw3=s_gw3/(1-beta1^counter);
%     mw4=s_gw4/(1-beta1^counter);
%     mb2=s_gb2/(1-beta1^counter);
%     mb3=s_gb3/(1-beta1^counter);
%     mb4=s_gb4/(1-beta1^counter);
% 
%     vw2=sum_gw2/(1-beta2^counter);
%     vw3=sum_gw3/(1-beta2^counter);
%     vw4=sum_gw4/(1-beta2^counter);
%     vb2=sum_gb2/(1-beta2^counter);
%     vb3=sum_gb3/(1-beta2^counter);
%     vb4=sum_gb4/(1-beta2^counter);  
% 
%     W2 = W2 - eta./(sqrt(vw2)+epsilon).*mw2;
%     W3 = W3 - eta./(sqrt(vw3)+epsilon).*mw3;
%     W4 = W4 - eta./(sqrt(vw4)+epsilon).*mw4;
%     b2 = b2 - eta./(sqrt(vb2)+epsilon).*mb2;
%     b3 = b3 - eta./(sqrt(vb3)+epsilon).*mb3;
%     b4 = b4 - eta./(sqrt(vb4)+epsilon).*mb4;

% 
%     %%%AdapGrad
%     sum_gw2 = sum_gw2+gw2.*gw2;
%     sum_gw3 = sum_gw3+gw3.*gw3;
%     sum_gw4 = sum_gw4+gw4.*gw4;
%     sum_gb2 = sum_gb2+delta2.*delta2;
%     sum_gb3 = sum_gb3+delta3.*delta3;
%     sum_gb4 = sum_gb4+delta4.*delta4;






    % Monitor progress
    newcost = cost(W2,W3,W4,b2,b3,b4)   % display cost to screen
    savecost(counter) = newcost;
end






figure(2)
clf
semilogy([1:1e4:Niter],savecost(1:1e4:Niter),'b-','LineWidth',2)
xlabel('Iteration Number')
ylabel('Value of cost function')
set(gca,'FontWeight','Bold','FontSize',18)
print -dpng pic_cost.png

%%%%%%%%%%% Display shaded and unshaded regions 
N = 500;
Dx = 1/N;
Dy = 1/N;
xvals = 0:Dx:1;
yvals = 0:Dy:1;
for k1 = 1:N+1
    xk = xvals(k1);
    for k2 = 1:N+1
        yk = yvals(k2);
        xy = [xk;yk];
         z2=W2*xy+b2;
         a2 = activate(z2);
         z3=W3*a2+b3;
         a3 = activate(z3);
         z4=W4*a3+b4;
         a4 = activate(z4);
        Aval(k2,k1) = a4(1);
        Bval(k2,k1) = a4(2);
     end
end
[X,Y] = meshgrid(xvals,yvals);

figure(3)
clf
f3 = subplot(1,1,1);
Mval = Aval>Bval;
contourf(X,Y,Mval)
hold on
colormap([1 1 1; 0.8 0.8 0.8])
plot(x1(1:5),x2(1:5),'ro','MarkerSize',12,'LineWidth',4)
plot(x1(6:10),x2(6:10),'bx','MarkerSize',12,'LineWidth',4)
f3.XTick = [0 1];
f3.YTick = [0 1];
f3.FontWeight = 'Bold';
f3.FontSize = 16;
xlim([0,1])
ylim([0,1])

print -dpng pic_bdy_bp.png

  function costval = cost(W2,W3,W4,b2,b3,b4)

     costvec = zeros(10,1); 
     for i = 1:10
         x =[x1(i);x2(i)];
         z2=W2*x+b2;
         a2 = activate(z2);
         z3=W3*a2+b3;
         a3 = activate(z3);
         z4=W4*a3+b4;
         a4 = activate(z4);
         costvec(i) = norm(y(:,i) - a4,2);
     end
     costval = norm(costvec,2)^2;
   end % of nested function
function y = activate(z)
 y = 1./(1+exp(-z)); 
end
    function y=grad_activate(x)
 y=x.*(1-x);
    end

eta
epsilon
%rho
beta1
beta2

end

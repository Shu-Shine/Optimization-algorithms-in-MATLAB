function [xopt,fopt,niter,gnorm,dx] = grad_descent_1D
% min f(x)=x1^2+x1x2+3x2^2
%%%%%%%%%Initial value%%%%%%
x0=2;
% termination tolerance
tol = 1e-6; %最小1e-16,一般-6或-8
% maximum number of allowed iterations
maxiter = 10; %10000
% minimum allowed perturbation
dxmin = 1e-6;

% initialize gradient norm, optimization vector, iteration counter, perturbation
gnorm = inf; 

niter = 0; 
dx = inf; %正无穷


% define the objective function:
f = @(x) x.^2 + 1;


% plot objective function contours for visualization:
%[X,Y]=meshgrid(-5:0.1:5,-5:0.1:5); %生成网格
x = -15:0.1:15;


figure(1) %用于建立图形窗口
plot(x,f(x))
hold on %点击行号，可以暂停执行到此行？
% figure(1); clf; fcontour(f); axis equal; hold on
% redefine objective function syntax for use with optimization%:
%f2 = @(x) f(x(1),x(2)); %重新定义函数，将变量x表示成vector
x = x0; 
% gradient descent algorithm:
while (gnorm>=tol&&niter <= maxiter&&dx >= dxmin) 
    % calculate gradient:
    g = grad1(x);
    gnorm = norm(g); %欧几里德范数
    %%%%%%%Line search%%%%%%%%
%     t=1; %为啥取1？
%     xnew=x-t*g;
%      while f2(xnew)>f2(x)-0.6*t*gnorm^2 %α=0.6 ？
%          t=0.99*t; %避免过小
%          xnew=x-t*g;
%      end

    %%%Constant stepsize:%%%%%

       alpha = 1.1; %0.1收敛，0.4就发散了
       xnew = x - alpha*g;

    % plot current point
     plot([x,xnew],[f(x),f(xnew)],'ko-')
    
     
    % update termination metrics
    niter = niter + 1;
    dx = norm(xnew-x);
    x = xnew;
    
end
xopt = x
fopt = f(xopt)
gnorm
niter = niter - 1
dx
% define the gradient of the objective
function g = grad1(x)
 g = 2*x ;
end
end
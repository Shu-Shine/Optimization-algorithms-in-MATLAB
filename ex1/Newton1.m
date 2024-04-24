
function Newton1
f=@(x) 2*x(1)^2-1.05*x(1)^4+x(1)^6/6+x(1)*x(2)+x(2)^2;
fp=@(x) [4*x(1)-4.2*x(1)^3+x(1)^5+x(2); x(1)+2*x(2)];
fpp=@(x) [4-12.6*x(1)^2+5*x(1)^4,1;1,2];
[x,xk]=Newton(f,fp,fpp,[0.4;0.4]);
F=@(x1,x2) 2*x1^2-1.05*x1^4+x1^6/6+x1*x2+x2^2;
Fp1=@(x1,x2) 4*x1-4.2*x1^3+x1^5+x2;
Fp2=@(x1,x2) x1+2*x2;
[X,Y]=meshgrid(-1.5:0.1:1.5,-1.5:0.1:1.5);
contour(X,Y,F(X,Y),50,LineWidth=1)
hold on
% quiver(X,Y,Fp1(X,Y),Fp2(X,Y),5)
plot(xk(1,:),xk(2,:),'-o')
hold off
end







function [x,xk]=Newton(f,fp,fpp,x0,tol,maxiter,tau,be,alinit)
% NEWTON Minimization with Newton descent and Armijo line search
% [x,xk]=Newton(f,fp,fpp,x0,tol,maxiter,tau,be,alinit) finds an
% approximate minimum of the function f with gradient fp and Hessian
% fpp, starting at the initial guess x0. The remaining parameters are
% optional and default values are used if they are omited. xk
% contains all the iterates of the method.
if nargin<9, alinit=1; end
if nargin<8, be=0.1; end
if nargin<7, tau=0.5; end
if nargin<6, maxiter=100; end
if nargin<5, tol=1e-8; end
x=x0;
xk=x0;
p=-feval(fpp,x)\feval(fp,x);
k=0;
while norm(feval(fp,x))>tol && k<maxiter
al=alinit;
while feval(f,x+al*p)>feval(f,x)-al*be*p'*p %使得f(xk+1)减小的最大步长
   al=tau*al;
end
x=x+al*p;
% x=x+p;
p=-feval(fpp,x)\feval(fp,x);
k=k+1;
xk(:,k+1)=x;
end
k
x
end
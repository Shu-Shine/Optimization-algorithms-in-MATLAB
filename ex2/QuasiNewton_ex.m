
function QuasiNewton_ex
f=@(x) 100*(x(2)-x(1)^2)^2+(x(1)-1)^2;
% define the gradient and Hessian of the objective
fp=@(x) [2*x(1)-400*x(1)*(-x(1)^2+x(2))-2; -200*x(1)^2+200*x(2)];
%fpp=@(x) [1200*x(1)^2 - 400*x(2)+2,-400*x(1);-400*x(1),200];
x0=[0;0];

F=@(x1,x2) 100*(x2-x1.^2).^2+(x1-1).^2;
Fp1=@(x1,x2) 2*x1-400*x1.*(-x1.^2+x2)-2;
Fp2=@(x1,x2) -200*x1.^2+200*x2;
[X,Y]=meshgrid(-1.5:0.1:1.5,-1.5:0.1:1.5);
contour(X,Y,F(X,Y),50,LineWidth=1)
hold on

fprintf('DFP formula for Hk');
[x,xk]=DFP_H(f,fp,x0);

% fprintf('BFGS formula for Hk');
% [x,xk]=BFGS_H(f,fp,x0);
% 
% fprintf('DFP formula for Bk');
% [x,xk]=DFP_B(f,fp,x0);
% 
% fprintf('BFGS formula for Bk');
% [x,xk]=BFGS_B(f,fp,x0);


quiver(X,Y,Fp1(X,Y),Fp2(X,Y),2)
plot(xk(1,:),xk(2,:),'-o')
hold off



end



function [x,xk]=BFGS_B(f,fp,x0)%？
alinit=1;
be=0.2;
tau=0.999; 
maxiter=50000; 
tol=1e-8;
B=eye(size(x0,1));
x=x0;
xk=x0;
k=0;
while norm(feval(fp,x))>tol && k<maxiter
p=-B\feval(fp,x);
al=alinit;
while feval(f,x+al*p)>feval(f,x)-al*be*p'*p %加line search吗?
   al=tau*al;
end
x_new=x+al*p;
g=feval(fp,x);
g_new=feval(fp,x_new);
yk=g_new-g; sk=x_new-x; yks=yk'*sk;
%  if yks> 1e-6*norm(sk)*norm(yk)
Bs=B*sk;
B=B+(yk*yk')/yks-(Bs*Bs')/(sk'*Bs);
%  end
k=k+1;
x=x_new;
xk(:,k+1)=x;
end
k
x
end


function [x,xk]=BFGS_H(f,fp,x0)
alinit=1;
be=0.2;
tau=0.999; 
maxiter=50000; 
tol=1e-8;
B=eye(size(x0,1));
H=inv(B);
x=x0;
xk=x0;
k=0;
while norm(feval(fp,x))>tol && k<maxiter
    p=-H*feval(fp,x);
    al=alinit;
    while feval(f,x+al*p)>feval(f,x)-al*be*p'*p
       al=tau*al;
    end
    x_new=x+al*p;
    g=feval(fp,x);
    g_new=feval(fp,x_new);
    yk=g_new-g; sk=x_new-x; sky=sk'*yk;
    %  if yks> 1e-6*norm(sk)*norm(yk)
%     Bs=B*sk;
%     B=B+(yk*yk')/yks-(Bs*Bs')/(sk'*Bs);
    Hy=H*yk;
    H=H+(1+yk'*Hy/sky)*(sk*sk')/sky-(sk*yk'*H+Hy*sk')/sky;
    %  end
    k=k+1;
    x=x_new;
    xk(:,k+1)=x;
end
k
x
end



function [x,xk]=DFP_H(f,fp,x0)
alinit=1;
be=0.2;
%tau=0.24; 
tau=0.999; 
maxiter=50000; 
tol=1e-8;
B=eye(size(x0,1));
H=inv(B);

x=x0;
xk=x0;
k=0;
while norm(feval(fp,x))>tol && k<maxiter
    p=-H*feval(fp,x);
    al=alinit;
    while feval(f,x+al*p)>feval(f,x)-al*be*p'*p
       al=tau*al;
    end
    x_new=x+al*p;
    g=feval(fp,x);
    g_new=feval(fp,x_new);
    yk=g_new-g; sk=x_new-x; sky=sk'*yk;
    %  if yks> 1e-6*norm(sk)*norm(yk)
    
    %Bs=B*sk;
    Hy=H*yk;
    %B=B+(yk*yk')/yks-(Bs*Bs')/(sk'*Bs);
    H=H+(sk*sk')/sky-(Hy*Hy')/(yk'*Hy);
    %  end
    k=k+1;
    x=x_new;
    xk(:,k+1)=x;
end
k
x
end


function [x,xk]=DFP_B(f,fp,x0)
alinit=1;
be=0.2;
%tau=0.24; 
tau=0.999; 
maxiter=50000; 
tol=1e-8;
B=eye(size(x0,1));
H=inv(B);
x=x0;
xk=x0;
k=0;
while norm(feval(fp,x))>tol && k<maxiter
    p=-B\feval(fp,x);
    al=alinit;
    while feval(f,x+al*p)>feval(f,x)-al*be*p'*p
       al=tau*al;
    end
    x_new=x+al*p;
    g=feval(fp,x);
    g_new=feval(fp,x_new);
    yk=g_new-g; sk=x_new-x; yks=yk'*sk;
     if yks> 1e-6*norm(sk)*norm(yk)
    
    Bs=B*sk;
    B=B+(1+sk'*Bs/yks)*(yk*yk')/yks-(yk*sk'*B+Bs*yk')/yks;

     end
    k=k+1;
    x=x_new;
    xk(:,k+1)=x;
end
k
x
end

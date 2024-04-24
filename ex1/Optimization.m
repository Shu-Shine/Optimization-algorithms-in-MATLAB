
function Optimization
% define the objective function:
f=@(x) 100*(x(2)-x(1)^2)^2+(x(1)-1)^2;
% define the gradient and Hessian of the objective
fp=@(x) [2*x(1)-400*x(1)*(-x(1)^2+x(2))-2; -200*x(1)^2+200*x(2)];
fpp=@(x) [1200*x(1)^2 - 400*x(2)+2,-400*x(1);-400*x(1),200];

%Initial value
x0=[0,0]';

%%% Gradient descent method with constant step sizes
fprintf('Gradient descent method with constant step sizes');
[xopt,fopt]=graddescent(f,fp,x0,0.1,"Constant stepsize");
[xopt,fopt]=graddescent(f,fp,x0,0.01,"Constant stepsize");
[xopt,fopt]=graddescent(f,fp,x0,0.001,"Constant stepsize");
[xopt,fopt]=graddescent(f,fp,x0,0.0001,"Constant stepsize");

%%% Gradient descent method with backtracking line search
fprintf('Gradient descent method with backtracking line search');
[xopt,fopt]=graddescent(f,fp,x0,1,"Line search");

%%% Classic Newton method
fprintf('Classic Newton method');
[xn]=Newtonmethod(f,fp,fpp,x0,"Classic");

%%% Newton method with backtracking line search
fprintf('Newton method with backtracking line search');
[xn]=Newtonmethod(f,fp,fpp,x0,"Line search");

end


function [xopt,fopt] = graddescent(f,fp,x0,alpha,choose_way)
% termination tolerance
tol = 1e-8; 
% maximum number of allowed iterations
maxiter = 50000; 
% minimum allowed perturbation
dxmin = 1e-8;
% initialize optimization vector, iteration counter, perturbation
x = x0; 
niter = 0; 
dx = inf; 

% gradient descent algorithm:
while (norm(feval(fp,x))>=tol&&niter <= maxiter&&dx >= dxmin) 
    switch choose_way
    case{"Line search"}
    %%%%%%%Line search%%%%%%%%
        t=1; 
        xnew=x-t*feval(fp,x);
         while f(xnew)>f(x)-0.5*t*norm(feval(fp,x))^2 
             t=0.99*t; 
             xnew=x-t*feval(fp,x);
         end
    case{"Constant stepsize"}
    %%%Constant stepsize:%%%%%
        xnew = x - alpha*feval(fp,x);
    end 
    % update termination metrics
    niter = niter + 1;
    dx = norm(xnew-x);
    x = xnew;    
end

xopt = x
fopt = f(xopt);
norm(feval(fp,x));
dx
niter = niter - 1

end



function [x]=Newtonmethod(f,fp,fpp,x0,choose_way)

alinit=1;
be=0.01;
tau=0.9; 
maxiter=50000; 
tol=1e-8; 

x=x0;
p=-feval(fpp,x)\feval(fp,x);
k=0;

while norm(feval(fp,x))>tol && k<maxiter
    al=alinit;
    if choose_way=="Line search"
        while feval(f,x+al*p)>feval(f,x)-al*be*p'*p
           al=tau*al;
        end
        x=x+al*p;
    else
        x=x+p;
    end
    
    p=-feval(fpp,x)\feval(fp,x);
    k=k+1;
end
k
x
end

n = 100;
% generate a well-conditioned positive definite matrix
% (for faster convergence)
P = rand(n);
P = P + P';
[V D] = eig(P);
P = V*diag(1+rand(n,1))*V';
q = randn(n,1);
r = randn(1);
l = randn(n,1);
u = randn(n,1);
lb = min(l,u);
ub = max(l,u);
[x history] = quadprog(P, q, r, lb, ub,1.0);



K = length(history.objval);
h = figure;
plot(1:K, history.objval, 'k', 'MarkerSize', 10, 'LineWidth', 2);
ylabel('f(x^k) + g(z^k)'); xlabel('iter (k)');
g = figure;
subplot(2,1,1);
semilogy(1:K, max(1e-8, history.r_norm), 'k', ...
1:K, history.eps_pri, 'k--', 'LineWidth', 2);
ylabel('||r||_2');
subplot(2,1,2);
semilogy(1:K, max(1e-8, history.s_norm), 'k', ...
1:K, history.eps_dual, 'k--', 'LineWidth', 2);
ylabel('||s||_2'); xlabel('iter (k)');

function [z, history] = quadprog(P, q, r, lb, ub, beta)
% Solves the following problem via ADMM:
%
% minimize (1/2)*x’*P*x + q’*x + r
% subject to lb <= x <= ub

t_start = tic;
QUIET = 0;
MAX_ITER = 1000;
ABSTOL = 1e-4;
RELTOL = 1e-2;
n = size(P,1);
x = zeros(n,1);
z = zeros(n,1);
u = zeros(n,1);

if ~QUIET
fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
      'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
end
for k = 1:MAX_ITER
%%%%ADMM iteration%%%%%%%
%(please complete this missing part, including the updates of x,z and u)%
% diagnostics, reporting, termination checks

% x-update
s=beta*(z-u)-q;
x=(P+beta*eye(n,n))\s;
% z-update
zold = z;
z = max(lb, min(ub, x + u));
% u-update
u = u + (x- z);


history.objval(k) = objective(P, q, r, x);
history.r_norm(k) = norm(x - z);
history.s_norm(k) = norm(-beta*(z - zold));
history.eps_pri(k) = sqrt(n)*ABSTOL + RELTOL*max(norm(x), norm(-z));
history.eps_dual(k)= sqrt(n)*ABSTOL + RELTOL*norm(beta*u);
if ~QUIET
fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
      history.r_norm(k), history.eps_pri(k), ...
      history.s_norm(k), history.eps_dual(k), history.objval(k));
end
if (history.r_norm(k) < history.eps_pri(k) && ...
        history.s_norm(k) < history.eps_dual(k))
        
break;
end
end
if ~QUIET
toc(t_start);
end
end
function obj = objective(P, q, r, x)
obj = 0.5*x'*P*x + q'*x + r;
end



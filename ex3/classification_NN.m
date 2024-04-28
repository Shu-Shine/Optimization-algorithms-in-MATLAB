function classification_NN
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
eta = 0.05;%大小？否则不收敛
Niter = 1e6;
savecost = zeros(Niter,1);
for counter = 1:Niter
    k = randi(10);
    x = [x1(k); x2(k)];
    % Forward pass
         z2=W2*x+b2;
         a2 = activate(z2);
         z3=W3*a2+b3;
         a3 = activate(z3);
         z4=W4*a3+b4;
         a4 = activate(z4);
    % Backward pass
    delta4 = diag(grad_activate(a4))*(a4-y(:,k));
    delta3 = diag(grad_activate(a3))*(W4'*delta4);
    delta2 = diag(grad_activate(a2))*(W3'*delta3);
    % Gradient step
    W2 = W2 - eta*delta2*x';
    W3 = W3 - eta*delta3*a2';
    W4 = W4 - eta*delta4*a3';
    b2 = b2 - eta*delta2;
    b3 = b3 - eta*delta3;
    b4 = b4 - eta*delta4;
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
end

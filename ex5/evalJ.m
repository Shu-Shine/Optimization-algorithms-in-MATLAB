function J = evalJ(Yout, X, V, b, NT, weights, dt)

J = 0;
for kk = 2:NT-1
    J = J + sum((X(:,kk) - Yout).^2);
end
J = dt*weights(1)*J/2;

J = J + sum((X(:,end) - Yout).^2)/2;

J = J + dt*weights(2)/2*(sum(sum(sum(V.^2))) + sum(sum(b.^2)));
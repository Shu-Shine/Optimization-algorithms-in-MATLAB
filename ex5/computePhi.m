function Phi = computePhi(Yout, X, V, b, N, I, NT, dsigma, weights, dt)

Phi = zeros(N*I, NT-1);
Phi(:,end) = X(:,end)-Yout;
for kk = NT-2:-1:1
    Vkk = V(:, :, kk+1);
    for ii = 1:I
        ind = N*(ii-1)+(1:N);
        ds = diag(dsigma(X(ind,kk+1)));
        Phi(ind,kk) = Phi(ind,kk+1) + dt*ds*(Vkk.')*Phi(ind,kk+1) + dt*weights(1)*(X(ind,kk)-Yout(ind));
    end
end
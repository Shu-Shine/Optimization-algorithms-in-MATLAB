function X = computeX(X0, V, b, N, I, NT, sigma, dt)

X = zeros(N*I, NT);
X(:,1) = X0;
for kk = 1:(NT-1)
    for ii = 1:I
        ind = N*(ii-1)+(1:N);
        X(ind,kk+1) = X(ind,kk) + dt*(V(:,:,kk)*sigma(X(ind,kk)) + b(:,kk));
    end
end
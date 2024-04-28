function [gradV, gradb] = computeGradients(Phi, X, V, b, N, I, NT, sigma, weights, dt)

gradV = zeros(size(V));
gradb = zeros(size(b));
for kk = 1:NT-1
    for ii = 1:I
        ind = N*(ii-1)+(1:N);
        s = sigma(X(ind,kk));
        gradV(:,:,kk) = gradV(:,:,kk) + dt*Phi(ind,kk)*(s.');
        gradb(:,kk)   = gradb(:,kk)   + dt*Phi(ind,kk);
    end
end
     

gradV = gradV + dt*weights(2)*V;
gradb = gradb + dt*weights(2)*b;
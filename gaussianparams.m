function [mu, C] = gaussianparams(X, n)
    mu = mean(X, 2);
    Xc = X - mu;
    C = (1/(n-1))*(Xc*Xc');
end
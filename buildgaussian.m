function [dist] = buildgaussian(mu, C)
    iCov = inv(C);
    dist = @(x_d)-0.5*log(det(C)) - 0.5*((x_d - mu)'*iCov*(x_d - mu));
end
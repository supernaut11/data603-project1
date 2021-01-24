function [train, test] = serialsplit(X, boundary)
    train = X(:,1:boundary);
    test = X(:,boundary+1:end);
end
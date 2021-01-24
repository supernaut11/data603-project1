function [Yc, Uc] = mypca(X, num_dimensions)
    % Center all of the data
    Xm = mean(X, 2);
    Xc = X - Xm;

    % Solve for the eigenvalues of the scatter matrix of the centered data
    [Uc,Sigmac,Vc] = svd(Xc, 'econ');

    esort = diag(Sigmac);

    figure;
    plot(esort,'.','Markersize',20);
    grid;

    Yc = Uc(:,1:num_dimensions)'*X;
end
function [classification] = myknn(k, train, test, labels)
    distances = vecnorm(train - test);

    [~,I] = mink(distances, k);
    classification = mode(labels(:,I),2);
end
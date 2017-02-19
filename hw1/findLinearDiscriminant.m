% This function finds a linear discriminant using LP
% The linear discriminant is represented by 
% the weight vector w and the threshold theta.
% YOU NEED TO FINISH IMPLEMENTATION OF THIS FUNCTION.

function [w,theta,delta] = findLinearDiscriminant(data)
%% setup linear program
[m, np1] = size(data);
n = np1-1;

% write your code here
b = ones(m+1, 1);
b(m+1, 1) = 0;

c = zeros(n+2, 1);
c(n+2, 1) = 1;

A = zeros(m+1, n+2);
A(1:m+1, n+2) = 1;
y = data(:, n+1);
A(1:m, n+1) = y;
for i = 1:m
    A(i, 1:n) = y(i) * data(i, 1:n);
end
%% solve the linear program
%adjust for matlab input: A*x <= b
[t, z] = linprog(c, -A, -b);

%% obtain w,theta,delta from t vector
w = t(1:n);
theta = t(n+1);
delta = t(n+2);

end

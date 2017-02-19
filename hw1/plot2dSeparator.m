% This function plots the linear discriminant.
% YOU NEED TO IMPLEMENT THIS FUNCTION

function plot2dSeparator(w, theta)
xdata = -0.1:0.01:1.1;
y = - w(1)/w(2) * xdata - theta/w(2);
plot(xdata, y);
end

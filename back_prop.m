function[theta, delta_weight_prev] = back_prop(a, D, theta, delta_weight_prev, ...
    learning_rate, RP, NL, LS)

N = size(D, 2);
delta = zeros(max(LS), N, NL);

Y = a(1:LS(NL), :, NL);
delta(1:LS(NL), 1:N, NL) = (D-Y).* a(1:LS(NL),:, NL).* (1 - a(1:LS(NL),:, NL));

for j = NL-1:-1:2
    delta(1:LS(j), 1:N, j) = (theta(1:LS(j), 1:LS(j+1), j) * ...
        delta(1:LS(j+1) ,:, j+1)) .* a(1:LS(j),:, j) .* (1 - a(1:LS(j),:, j));
end;

for j = NL-1:-1:1
    %grad is of the form j size by j+1
    grad = delta(1:LS(j + 1),:, j+1) * a(1:LS(j),:, j).';
    delta_weight(1:LS(j),1:LS(j+1),j) = (learning_rate * grad).' + (RP * ...
        delta_weight_prev(1:LS(j),1:LS(j+1),j));
    theta(1:LS(j), 1:LS(j+1), j) = theta(1:LS(j), 1:LS(j+1), j) + ...
        delta_weight(1:LS(j),1:LS(j+1),j);
end;

delta_weight_prev = delta_weight;
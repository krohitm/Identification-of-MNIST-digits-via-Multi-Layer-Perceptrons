function[theta, delta_weight_prev] = bwd_propagation(a, theta, ...
    delta_weight_prev, D, learning_rate, RP, NL, LS)
%NL --> number of layers
%LS --> size of layers

%thetaPrev = theta;
%delta is of the form (layer_size by layer_number)
delta = zeros(max(LS), NL);   %won't be using delta 1 but kept for ease of convention

Y = a(1:LS(NL), NL);
%get delta for last layer
delta(1:LS(NL), NL) = (D - Y).* a(1:LS(NL), NL).* (1 - a(1:LS(NL), NL));

%get delta for all other layers except 1
for j = (NL - 1):-1:2
    delta(1:LS(j), j) = (theta(1:LS(j), 1:LS(j+1), j) * ...
        delta(1:LS(j+1), j+1)) .* a(1:LS(j), j) .* (1 - a(1:LS(j), j));
end;

for j = NL-1:-1:1
    grad = delta(1:LS(j + 1), j+1) * a(1:LS(j), j).';
    delta_weight(1:LS(j),1:LS(j+1),j) = (learning_rate * grad).' + (RP * ...
        delta_weight_prev(1:LS(j),1:LS(j+1),j));
    theta(1:LS(j), 1:LS(j+1), j) = theta(1:LS(j), 1:LS(j+1), j) + ...
        delta_weight(1:LS(j),1:LS(j+1),j);
end;

delta_weight_prev = delta_weight;
%E = delta(1:LS(NL), NL);
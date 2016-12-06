function[theta, thetaPrev] = bwd_propagation(a, theta, ...
    thetaPrev, D, learning_rate, RP, NL, LS)

%thetaPrev = theta;
delta = zeros(max(LS), NL);   %won't be using delta 1 but kept for ease of convention

Y = a(NL, 1:LS(NL));
%get delta for last layer
delta(1:LS(NL),NL) = (D - Y);%.* Y.* (1 - Y);
    
%get delta for all other layers except 1
for j = (NL - 1):-1:2
    delta(1:LS(j), j) = (((theta(2:LS(j) + 1, 1:LS(j+1), j) * ...
        delta(1:LS(j+1), j+1)).') .* a(j,1:LS(j)) .* (1 - a(j,1:LS(j)))).';
end;
    
thetaTemp = theta;
%update weights for all layers
for j = NL:-1:2
    theta(1:LS(j - 1) + 1, 1:LS(j), j - 1) =  ...
        theta(1:LS(j - 1) + 1, 1:LS(j), j - 1) + ...
        (learning_rate * delta(1:LS(j), j) * a(j-1, 1:LS(j-1) + 1)).' + ...
        RP * ([zeros(1, LS(j)); theta(2:LS(j - 1) + 1, 1:LS(j), j - 1)] - ...
        [zeros(1, LS(j)); thetaPrev(2:LS(j - 1) + 1, 1:LS(j), j - 1)]);
end;

thetaPrev = thetaTemp;
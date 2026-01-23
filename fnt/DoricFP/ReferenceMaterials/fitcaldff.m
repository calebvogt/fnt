function [dff]=fitcaldff(trace405,trace473)
 
bls1 = polyfit(trace405,trace473,1);
x2 = lsqnonneg(trace405,trace473);

if bls1(1,1) <= 0
    yfit1 = x2 * trace405;
else
    yfit1 = polyval(bls1,trace405);
end

dff=100*(trace473-yfit1)/mean(trace473);

end
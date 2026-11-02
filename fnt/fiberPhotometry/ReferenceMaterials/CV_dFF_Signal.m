trace405=photomat(:,3);
trace473=photomat(:,2);
b1s1=polyfit(trace405,trace473,1);
x2=lsqnonneg(trace405,trace473);
if b1s1(1,1) <= 0
    yfit1 = x2 * trace405;
else yfit1 = polyval(b1s1,trace405);
end
dff=100*(trace473-yfit1)/mean(trace473);
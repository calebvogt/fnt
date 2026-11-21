function [dff,filt405,filt473]=filteredfitcaldff(trace405,trace473)
% Filter signals
sRate= 250;
Nyquist= sRate/2;
order= 100;

b=fir1(order,20/Nyquist,'low');
filt405=filtfilt(b,1, trace405);
filt473=filtfilt(b,1, trace473);
 
bls1 = polyfit(filt405,filt473,1);
x2 = lsqnonneg(filt405,filt473);

if bls1(1,1) <= 0
    yfit1 = x2 * filt405;
else
    yfit1 = polyval(bls1,filt405);
end

dff=100*(filt473-yfit1)/mean(filt473);

end
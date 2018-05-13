function T = argmin_g(w0, zeta, X, T)
     lhd= 1 ./  (w0 .^2 + zeta); % left hand
     
     % compute T for each channel
     for i = 1:size(X,3)
         T(:,:,i) = lhd .* X(:,:,i);
     end
end



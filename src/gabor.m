function Psi = gabor (w,nu,mu,Kmax,f,sig)

% w  : Window [128 128]
% nu : Scale [0 ...4];
% mu : Orientation [0...7]
% kmax = pi/2
% f = sqrt(2)
% sig = 2*pi

m = w(1);
n = w(2);
K = Kmax/f^nu * exp(i*mu*pi/8);
Kreal = real(K);
Kimag = imag(K);
NK = Kreal^2+Kimag^2;
Psi = zeros(m,n);
for x = 1:m
    for y = 1:n
        Z = [x-m/2;y-n/2];
        Psi(x,y) = (sig^(-2))*exp((-.5)*NK*(Z(1)^2+Z(2)^2)/(sig^2))*...
                   (exp(i*[Kreal Kimag]*Z)-exp(-(sig^2)/2));
    end
end
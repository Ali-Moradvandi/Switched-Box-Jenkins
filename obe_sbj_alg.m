function [theta,nu,flag_est,Ye,v] = obe_sbj_alg(Y,U,ni,delta,theta_ini,nu_ini,order)

na      = order(1);             % order of the system with respect to x
nb      = order(2);             % order of the system with respect to u
nc      = order(3);             % order of the system with respect to w
nd      = order(4);             % order of the system with respect to v

ite     = length(Y);            % number of iterations
Ye      = zeros(ite,ni);        % augmented estimation errors
flag_est= zeros(ite,1);         % initializing estimated modes
theta   = theta_ini;            % initializing estimated \theta
nu      = nu_ini;               % initializing estimated \nu
p0      = 1e1;                  % initializing p0
x       = zeros(ite,1);         % initializing x vector
w       = zeros(ite,1);         % initializing w vector
v       = ones(ite,1)/1e1;      % initializing v vector
P1      = p0*eye(ni*(na+nb));   % P w.r.t. \theta
P2      = p0*eye(ni*(nc+nd));   % P w.r.t. \nu
start   = max([na,nb,nc,nd]);   % iteration counting

PHI = zeros(ni*(na+nb),ni);     % initializing \phi
PSI = zeros(ni*(nc+nd),ni);     % initializing \psi

%% start the identification algorithm | The algorithm can be found in algorithm table 1 @https://doi.org/10.1016/j.jwpe.2024.105202
for i = start+3:ite-1

        PHI = [-x(i-1:-1:i-na+1-1); U(i-1:-1:i-nb+1-1)];
        PSI = [-w(i-1:-1:i-nc+1-1); v(i-1:-1:i-nd+1-1)];


        phi = kron(eye(ni),PHI);
        psi = kron(eye(ni),PSI);

        Ye(i,:)  = phi'*theta + psi'*nu;
        err_i    = Y(i)*ones(ni,1) - Ye(i,:)';  

        aaa       = abs(err_i);
        [val,ind] = min(aaa);    

        A = eye(ni);       
        if val>delta
            A(ind,ind) = val/delta;
            sigma_theta=  (inv(phi'*P1*phi))*(A - eye(ni)); 
            sigma_nu   =  (inv(psi'*P2*psi))*(A - eye(ni)); 
        else
            sigma_theta= zeros(ni);
            sigma_nu   = zeros(ni);
        end 

        L1  = P1*phi*sigma_theta/(eye(ni) + phi'*P1*phi*sigma_theta);
        L2  = P2*psi*sigma_nu   /(eye(ni) + psi'*P2*psi*sigma_nu);
        P1  = (eye(ni*(na+nb)) - L1*phi')*P1;  
        P2  = (eye(ni*(nc+nd)) - L2*psi')*P2;   

        theta = theta + 0.5*L1*(Y(i) - psi'*nu - phi'*theta);
        nu    = nu    + 0.5*L2*(Y(i) - psi'*nu - phi'*theta);

        Ye(i,:) = phi'*theta + psi'*nu;
        err_o   = Y(i)*ones(ni,1) - Ye(i,:)';
        bbb     = abs(err_o);
        [val2,ind2] = min(bbb);
        flag_est(i) = ind2;

        X      = phi'*theta;
        x(i)   = X(flag_est(i));
        w(i)   = Y(i) - x(i);
        V      = psi'*nu;
        v(i)   = w(i) - V(flag_est(i));

        % reseting P
        if mod(i,40) == 0
            P1 = 1e1*eye(ni*(na+nb));
            P2 = 1e1*eye(ni*(nc+nd));
        end
end 





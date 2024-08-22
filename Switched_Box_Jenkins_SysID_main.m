%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Copyright (C) TU Delft All Rights Reserved
% Written by Ali Moradvandi
% For any correspondence: moradvandi@gmail.com

%% Introduction of code (purpose)
% This code is the main code for the identification method based on
% Outer Bounded Ellipsoid (OBE) algorithm for Switched Box-Jenkins (SBJ) systems. 
% The details of the algorithm can be found in https://doi.org/10.1016/j.jwpe.2024.105202

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Defining a simulation model and generating data for indetification
clear all
clc

%----- fix random seed -----%
rng(13);
%---------------------------%

th = 500;                                   % time horizon
rand_switch = randi([1 2],1,th);
num_sys     = 2;
u           = 1*2*(rand(th,1)-0.5);         % input sequences
x           = zeros(th,1);                  % intermediate input sequences
w           = zeros(th,1);                  % intermediate disturbance/noise sequences
v           = zeros(th,1);                  % disturbance/noise sequences
ynl         = zeros(th,1);                  % noiseless output
y           = zeros(th,1);                  % corrupted output
flag        = zeros(th,1);                  % flag of modes
delta_e     = 0.04;                         % bound of disturbance/noise
nm          = 2;                            % number of mode


% data generation | This example can be found on https://doi.org/10.1016/j.jwpe.2024.105202
for k = nm+1:th
    v(k) = delta_e*2*(rand-0.5);            %% -0.04<disturbance/noise<0.04 %%
    
    %mode 1
    if rand_switch(k) == 1 
        x(k)   = -0.45*x(k-1) + 0.2*x(k-2) - 0.4*u(k-1) + 0.95*u(k-2);
        w(k)   = -0.64*w(k-1) + v(k) - 0.32*v(k-1); 
        y(k)   = x(k) + w(k);
        ynl(k) = x(k);
        flag(k)= 1;
    end
    % mode 2
    if rand_switch(k) == 2
        x(k)   = -0.1*x(k-1) - 0.35*x(k-2) - 0.5*u(k-1) + 1.1*u(k-2);
        w(k)   = 0.36*w(k-1) + v(k) - 0.5*v(k-1);
        y(k)   = x(k) + w(k);
        ynl(k) = x(k);
        flag(k)= 2;
    end
end

% data

na  = 2;     % order of the system with respect to x
nb  = 2;     % order of the system with respect to u 
nc  = 1;     % order of the system with respect to w
nd  = 1;     % order of the system with respect to v
order = [na,nb,nc,nd];

U   = u;     % Augmented U
Y   = y;     % Augmented Y
V   = w;     % Augmented V

%% Run the algorithm for one mode for initialization

ni          = 1;                      % number of mode = one for initialization
delta_1     = (max(Y)-min(Y))/2;      % assigning a bound for the algorithm
p0p         = 1e2;                    % initial P0
theta_ini   = ones((na+nb)*1,1)/p0p;  % intial \theta
nu_ini      = ones((nc+nd)*1,1)/p0p;  % intial \nu

% calling the algorithm function
[theta,nu,flag_est,Ye] = obe_sbj_alg(Y,U,ni,delta_1,theta_ini,nu_ini,order);

%% Run the algorithm for n mode based on results of the initialization

ni          = nm;                                       % assigning the number of mode
delta_2     = 0.1;                                      % assigning the error bound. Should be bigger than delta_e
theta_ini   = kron((ones(ni,1)+0*randn(ni,1)),theta);   % augmented intial \theta for n mode
nu_ini      = kron((ones(ni,1)+1*randn(ni,1)),nu);      % augmented intial \nu for n mode

% calling the algorithm function
[theta,nu,flag_est,Ye,Ve] = obe_sbj_alg(Y,U,ni,delta_2,theta_ini,nu_ini,order);

% calculating the output based on the identified clusters | restructuring
ye = zeros(th,1);
for i = 5:th-1
    ye(i) = Ye(i,flag_est(i));
end

% ploting real and estimated outputs

% calculation for the initial FIT
ef = y - ye;
f1 = figure(13);
plot(y,'g','linewidth',1.5)
hold on 
plot(ye,'b-.','linewidth',1.4)
legend('real', 'estimated','Interpreter','Latex','fontsize',10,'Location', 'southeast')
title('Identified SBJ model','Interpreter','Latex','fontsize',12);
grid on
ylabel('$y$,$\hat{y}$','Interpreter','Latex','fontsize',12)
xlabel('Time step','Interpreter','Latex','fontsize',12)
xlim([400 499])
ylim([-2 3])
yticks([-2 -1 0 1 2])
xticks([400 420 440 460 480])

axes('position',[.17 .65 .72 .25])
box on                              % put box around new pair of axes
tt = 0:500;
ta = ones(501,1);
plot(ef,'b','linewidth',1.5)
grid on
hold on
plot(tt, 0.1*ta,'r','linewidth',2)
plot(tt, -0.1*ta,'r','linewidth',2)
xlim([400 499])
xticks([400 420 440 460 480])
ylim([-0.1 0.1])
xticks([])
yticks([-0.1 0 0.1])
ylabel('$e_{k/k}$','Interpreter','Latex','fontsize',12)
f1.Position = [100 100 1100 450];

f2 = figure(16);
plot(flag_est,'b*','linewidth',2)
hold on 
plot(flag,'r+','linewidth',1.6)
legend ('estimated', 'real','Interpreter','Latex','fontsize',10)
title('Identified switch sequences','Interpreter','Latex','fontsize',12);
xlabel('Time step','Interpreter','Latex','fontsize',12)
ylabel('Mode number','Interpreter','Latex','fontsize',12)
yticks([1 2])
xlim([400 499])
xticks([400 420 440 460 480])
grid on
f2.Position = [100 100 1100 200];
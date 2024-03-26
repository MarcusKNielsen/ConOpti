clear
clc


%% Parameters for random QP
n=10; 
alpha=0.1; 
density=0.15; 

%% Initial guess of solution (we know zero is feasible)
x0 = zeros(n,1);

%% Generate random QP
[H,g,bl,A,bu,l,u]=RandomQP(n,alpha,density); 

%% Convert bl <= A' x <= bu, l <= x <= u to c(x) = Abar' x + bbar >= 0
Abar = [full(A) full(-A) eye(n,n) -eye(n,n)]; 
bbar = [-bl; bu; -l; u];

%% Solve QP using primal active-set algorithm
[x,lambda,Wset,it] = qpsolverActiveSet(H,g,Abar,bbar,x0);
x,Wset,it 




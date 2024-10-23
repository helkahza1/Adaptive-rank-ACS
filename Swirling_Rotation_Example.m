% Implementation of the developed integrator with rank-1 advection coefficients
% and multi-rank diffusion coefficients.
% "Reference paper"

% Author: Hamad El Kahza (In collaboration with Jingmei-Qiu, William Taintano, and Luis Chacón.)
% This code discretizes the spatial domain using a tensor-product grid and evolves
% the solution over time using an adaptive-rank implicit DIRK method. The advection and
% diffusion matrices are constructed based on spatially variable coefficients.
%
% The integrator uses Krylov subspace techniques to handle the temporal evolution
% efficiently, adjusting basis dimensions dynamically to maintain
% accuracy. Functions are provided for the construction of diffusion and advection
% matrices, as well as for the initialization and evolution of the solution.
%
%
% Input:
% - Grid dimensions (Nx, Ny), domain limits (L), advective CFL number.
% - Physical parameters including final time (Tf), Reynolds number (Re).
% - Advection and diffusion coefficients defined as function handles.
% - Krylov subspace tolerances.
%
% Output:
% - Evolution of the solution stored in matrices Vx_n, Vy_n, and S_n at each time step.
% - Plots showing the contour evolution of the solution at specified time intervals.
% - Krylov subspace dimensions and iteration counts are stored for analysis.
%
% Sections of the code:
% 1. Define Parameters
% 2. Create Grid
% 3. Define Diffusion and Advection Coefficients
% 4. Compute Diffusion and Advection Matrices
% 5. Set Initial Conditions
% 6. Time Loop with Krylov Subspace Methods and Plotting
% 7. Function Definitions for Diffusion and Advection Constructors


%% Define Parameters
Nx = 2100; Ny = 2100; L = 1;
x1 = -L; x2 = L; y1 = -L; y2 = L;
dx = (2 * L) / Nx; dy = (2 * L) / Ny;
Tf = 10; Cons = 100; lambda_A = 15;
adv_max = 2;  % Max velocity

%% Create Grid
xvals = linspace(x1, x2, Nx);
yvals = linspace(y1, y2, Ny);
xvalsf = linspace(x1 + dx / 2, x2 - dx / 2, Nx - 1);
yvalsf = linspace(y1 + dy / 2, y2 - dy / 2, Ny - 1);
xvalsfplus = xvals + dx / 2;
xvalsfminus = xvals - dx / 2;
yvalsfplus = yvals + dy / 2;
yvalsfminus = yvals - dy / 2;

%% Define Diffusion Coefficients
Re = 1000;
coeff1x = @(x) 1 + x - x;
coeff1y = @(y) 1 + y - y;
mydiffcoeffs = {coeff1x, coeff1y};
%Other choices:
% Re = 1000;
% coeff1x=@(x) exp(-(x-0.3*sin(x)).^2);
% coeff1y=@(y) exp(-(y-0.3*cos(y)).^2);
% coeff2x=@(x) exp(-(x-0.6*sin(pi*x)).^2);
% coeff2y=@(y) exp(-(y-0.6*sin(pi*y)).^2);
% coeff3x=@(x) exp(-(x-0.6*sin(2*pi*x)).^2);
% coeff3y=@(y) exp(-(y-0.6*sin(2*pi*y)).^2);
% mydiffcoeffs={coeff1x,coeff1y,coeff2x,coeff2y,coeff3x,coeff3y};

%% Define Advection Coefficients
advcoeff1x = @(x) -(1 - x.^2);
advcoeff1y = @(y) 2 .* y;
advcoeff2x = @(x) 2 * x;
advcoeff2y = @(y) (1 - y.^2);
myadvcoeffsx = {advcoeff1x, advcoeff1y};
myadvcoeffsy = { advcoeff2x, advcoeff2y};

%% Compute Diffusion Matrices
[Dv_diff, A_diff] = diffusion_constructor(xvals(2:end-1), yvals(2:end-1), xvalsfplus(2:end-1), yvalsfplus(2:end-1), xvalsfminus(2:end-1), yvalsfminus(2:end-1), mydiffcoeffs,Re);

%% Compute Advection Matrices
[Dv, A] = advection_constructor(xvals(2:end-1), yvals(2:end-1), xvalsfminus(2:end-1), xvalsfplus(2:end-1), yvalsfminus(2:end-1), yvalsfplus(2:end-1), myadvcoeffsx,myadvcoeffsy, A_diff, Dv_diff);

%% Time Step Calculations
dt = lambda_A * dx / adv_max;
Diff = 1 / Re;
tvals = 0:dt:Tf;
dt = tvals(2) - tvals(1);
CFL_adv = dt * adv_max / dx;
CFL_diff = dt * Diff / dx^2;


%% Initial Conditions
sigg = 0.15;
Vx_n = [exp(-1 / (2 * sigg^2) * (xvals(2:end-1)' - 0.3).^2), exp(-1 / (2 * sigg^2) * (xvals(2:end-1)' + 0.5).^2)];
Vy_n = [exp(-1 / (2 * sigg^2) * (yvals(2:end-1)' - 0.35).^2), exp(-1 / (2 * sigg^2) * (yvals(2:end-1)' + 0.5).^2)];
S_n = diag([0.5, 0.8]);
[Vx_n, rrx] = qr(Vx_n, 0);
[Vy_n, rry] = qr(Vy_n, 0);
[t1, S_n, t2] = svd(rrx * S_n * rry', 0);
Vx_n = Vx_n * t1;
Vy_n = Vy_n * t2;

%% Krylov Parameters
r = 2; %% rank-2 initial condition
krylov_iter = zeros(size(tvals));
krylov_dim = zeros(size(tvals));
krylov_dim_bef = zeros(size(tvals));
rankk = zeros(size(tvals));
rankk(1) = r;
epsilon_kappa=1e-5;
epsilon=1e-6;
epsilon_GMRES=1e-10;
epsilon_tol=1e-5;
tol={epsilon_kappa, epsilon_GMRES,epsilon,epsilon_tol};

%% Butcher Tableau for DIRK Method
a = [0.4358665215, 0, 0;
    0.5 * (1 - 0.4358665215), 0.4358665215, 0;
    -3 * 0.5 * 0.4358665215^2 + 4 * 0.4358665215 - 0.25, 3 * 0.5 * 0.4358665215^2 - 5 * 0.4358665215 + 5/4, 0.4358665215];

%% Define Operator Matrices
myopx = 0.5 * speye(Nx-2, Ny-2);
myopy = 0.5 * speye(Nx-2, Ny-2);
p=size(A, 2);
for i=1:p
    mymean{i}=mean((diag(Dv{i})));
end
ggg=1;
for i = 1:p
    if mod(i, 2) == 1
        myopx = myopx - (dt * a(1, 1) * mymean{i+1} * A{i});
        mimi{i} = ((1/size(A, 2)) * speye(Nx-2, Ny-2) - dt * mymean{i+1} * A{i});
        [LL{i}, U{i}] = lu(mimi{i});
        Aop{i} = ((1/size(A, 2)) * speye(Nx-2, Ny-2) - dt * a(1, 1) * mymean{i+1} * A{i});
    else
        myopy = myopy - (dt * a(1, 1) * mymean{i-1} * A{i});
        mimi{i} = ((1/size(A, 2)) * speye(Nx-2, Ny-2) - dt * mymean{i-1} * A{i});
        [LL{i}, U{i}] = lu(mimi{i});
        Aop{i} = ((1/size(A, 2)) * speye(Nx-2, Ny-2) - dt * Dv{i});
    end
end

%% Time Loop with Plotting
figure(10);
pp = 3;
n=1;
subplot(3, 2, 3);
contourf(xvals(2:end-1), yvals(2:end-1), Vx_n * S_n * Vy_n', 'LineStyle', 'none');
title(['Contour Plot at $t = ', num2str(tvals(n)), '$'], 'Interpreter', 'latex', 'FontSize', 17);
xlabel('$x$', 'Interpreter', 'latex', 'FontSize', 17);
ylabel('$y$', 'Interpreter', 'latex', 'FontSize', 17);
colorbar;
set(gca, 'FontSize', 17, 'LineWidth', 1.5, 'TickLabelInterpreter', 'latex', 'XMinorTick', 'on', 'YMinorTick', 'on');
pp=pp+1;
for n = 2:numel(tvals)
    length(tvals)-(n)
    dt = tvals(n) - tvals(n-1);
    
    [Vx_nn, Vy_nn, S_nn, size_krylov, size_krylov_bef, iter_krylov] = Adaptive_Rank_ACS_DIRK(Vx_n, S_n, Vy_n, a, tol, A, Dv, dt, LL, U, myopx, myopy);
    Vx_n = Vx_nn;
    Vy_n = Vy_nn;
    S_n = S_nn;
    krylov_iter(n) = iter_krylov;
    krylov_dim(n) = size_krylov;
    krylov_dim_bef(n) = size_krylov_bef;
    rankk(n) = size(Vx_nn, 2);
    
    % Plotting results at certain time steps
    if n == floor(length(tvals) / 3) || n == floor(2 * length(tvals) / 3) || n == numel(tvals)
        subplot(3, 2, pp);
        contourf(xvals(2:end-1), yvals(2:end-1), Vx_n * S_n * Vy_n', 'LineStyle', 'none');
        title(['Contour Plot at $t = ', num2str(tvals(n)), '$'], 'Interpreter', 'latex', 'FontSize', 17);
        xlabel('$x$', 'Interpreter', 'latex', 'FontSize', 17);
        ylabel('$y$', 'Interpreter', 'latex', 'FontSize', 17);
        colorbar;
        set(gca, 'FontSize', 17, 'LineWidth', 1.5, 'TickLabelInterpreter', 'latex', 'XMinorTick', 'on', 'YMinorTick', 'on');
        pp = pp + 1;
    end
end

subplot(3, 2, [1, 2]);  % Combine position 1 and 2 into a single plot
plot(tvals, rankk);
xlabel('time', 'Interpreter', 'latex', 'FontSize', 15); % Modify as needed
ylabel('rank', 'Interpreter', 'latex', 'FontSize', 15); % Modify as needed

%% Diffusion Constructor Function

function [Dv,A] = diffusion_constructor(xvals,yvals,xvalsfplus,yvalsfplus,xvalsfminus,yvalsfminus,mycoeffs,Re)

Nx=length(xvalsfplus);
Ny=Nx;
p=size(mycoeffs,2);
dx=xvals(2)-xvals(1);
dy=yvals(2)-yvals(1);
%faces discretization
ef = ones(Nx,1); % vector of ones of same size x
Dxplus  =(spdiags([-ef ef],[0 1],Nx,Nx)); % 1st order matrix
Dxplus= (1/dx^2 )*Dxplus;


%cells discretization
ec = ones(Nx,1); % vector of ones of same size x
Dxminus  =(spdiags([-ec ec],[-1 0],Nx,Nx)); % 1st order matrix
Dxminus= (1/dy^2 )*Dxminus;


for i=1:p
    if mod(i, 2) == 1
        v{i}=mycoeffs{i}(xvals');
        Dv{i}= spdiags(v{i},0,Nx,Ny )  ;
        vfplus{i}=mycoeffs{i}(xvalsfplus');
        Dvfplus{i}=spdiags(vfplus{i},0,Nx,Ny);
        vfminus{i}=mycoeffs{i}(xvalsfminus');
        Dvfminus{i}=spdiags(vfminus{i},0,Nx,Ny);
        %         A{i}=(1/(Ren)).*sparse(   Dxc*Dvf{i}*Dxf(:,2:end-1)   );
        
        A{i}=(1/(Re)).*(Dvfplus{i}*Dxplus-Dvfminus{i}*Dxminus);
        %     hm
    else
        v{i}=mycoeffs{i}(yvals');
        Dv{i}=spdiags(v{i},0,Nx,Ny);
        vfplus{i}=mycoeffs{i}(yvalsfplus');
        Dvfplus{i}=spdiags(vfplus{i},0,Nx,Ny);
        vfminus{i}=mycoeffs{i}(yvalsfminus');
        Dvfminus{i}=spdiags(vfminus{i},0,Nx,Ny);
        
        A{i}=(1/(Re)).*(Dvfplus{i}*Dxplus-Dvfminus{i}*Dxminus);
        %         A{i}=(1/(Ren)).*sparse(Dxc*Dvf{i}*Dxf(:,2:end-1));
    end
    
end

end


%% Advection Constructor Function

function [Dv, A] = advection_constructor(xvals, yvals, xvalsfminus, xvalsfplus, yvalsfminus, yvalsfplus, myadvcoeffsx,myadvcoeffsy, A, Dv)
p=length(A);
Nx=size(A{1},1);
Ny=size(A{1},2);
dx=xvals(end)-xvals(end-1);
ef = (1/dx)*ones(Nx,1);
Splus  =(spdiags([ef ef],[0 1],Nx,Nx)); % 1st order matrix
Sminus=(spdiags([ef ef],[-1 0],Nx,Nx)); % 1st order matrix

Sigma1minus=spdiags(0.5*myadvcoeffsx{1}(xvalsfminus'),0,Ny,Ny);
Sigma1plus=spdiags(0.5*myadvcoeffsx{1}(xvalsfplus'), 0, Nx,Ny);
Sigma1= spdiags(myadvcoeffsx{1+1}(yvals' ),0,Nx,Ny );

op1=(Sigma1plus*Splus-Sigma1minus*Sminus );


A{p+1}=sparse(op1);
Dv{p+2}=sparse(Sigma1);


% for i=1:length(myadvcoeffsx)
Sigma2=spdiags(myadvcoeffsy{1}(xvals'),0,Nx,Ny);
Sigma2minus=spdiags((1/2).*myadvcoeffsy{2}(yvalsfminus'),0,Nx,Ny);
Sigma2plus=spdiags((1/2).*myadvcoeffsy{2}(yvalsfplus'),0,Nx,Ny);

op3=(Sigma2plus*Splus- Sigma2minus*Sminus);
A{p+2}=sparse(op3);
Dv{p+1}=sparse(Sigma2);
end


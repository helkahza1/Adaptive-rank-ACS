function [Vx_nn, Vy_nn, S_nn, rankdata, rankdatabefore, iterdata] = Adaptive_Rank_ACS_DIRK(Vx_n, S_n, Vy_n, a, tol, A, Dv, dt, L, U, P_1, P_2)
% This function solves the generalized Sylvester equation arising from
% implicit discretization of advection-diffusion equations of the form:
% A{1}X_1Dv{2}'+Dv{1}X_1A{2}' + ... + A{p}X_1Dv{p-1}' = X_0
% where A  are difference operators and Dv are diagonal matrices.
% The function employs Diagonally Implicit
% Runge-Kutta (DIRK) integration schemes to advance the solution in time from t(n) to t(n+1).
%
% Inputs:
% Vx_n, Vy_n: Input matrices (basis vectors)
% S_n: Input singular values matrix
% a: Coefficient matrix for DIRK method
% tol: Cell array of tolerances [epsilon_kappa, epsilon_GMRES, epsilon, epsilon_tol]
% A, Dv: Cell arrays of matrices
% dt: Time step
% L, U: LU decomposition matrices (cell arrays) of operators A.
% P_1, P_2: Averaged operators
% mymean: Cell array of mean values
% Outputs:
% Vx_n, Vy_n: Updated basis vectors
% S_n: Updated singular values matrix
% rankdata: Final rank of Vx_nn
% rankdatabefore: Rank before truncation
% iterdata: Number of iterations taken to converge


% Extract tolerances
epsilon_kappa = tol{1};
epsilon_GMRES = tol{2};
epsilon = tol{3};
epsilon_tol = tol{4};

% Number of terms in A
p = size(A, 2);

% Inverse of first coefficient in 'a' matrix
oneovera = 1 / a(1, 1);

% Dimensions
[Nx, ~] = size(Vx_n);
[Ny, ~] = size(Vy_n);

% Initialize cell arrays
AV = cell(1, p);
A11 = cell(1, p);
Diag = cell(1, p);
DV11 = cell(1, p);

% Set initial rank (Optional: truncate)
r = size(Vx_n, 2);

% Initialize variables
Dxinv = repmat({Vx_n(:,1:r)}, 1, p);
Dyinv = repmat({Vy_n(:,1:r)}, 1, p);
DiagVxn = repmat({Vx_n(:,1:r)}, 1, p);
DiagVyn = repmat({Vy_n(:,1:r)}, 1, p);

% Compute norm of S_n
normb = norm(S_n, 'fro');
onenormb = 1 / normb;

% Initialize variables for augmentation
DxVxn2 = Vx_n(:,1:r);
DyVyn2 = Vy_n(:,1:r);
DxVxninv = Vx_n(:,1:r);
DyVyninv = Vy_n(:,1:r);

% Number of stages
stages = size(a, 2);

% Initialize RHS and TERM
RHS = cell(1, stages);
TERM = cell(1, stages - 1);

% Initialize convergence flag and iteration counter
acc = false;
s = 1;

% Main loop until convergence
while ~acc
    if s == 1
        % First iteration
        r0 = r;
        r1x = r0;
        r1y = r0;
        Vx_aug = zeros(Nx, (p + 3) * r0);
        Vy_aug = zeros(Ny, (p + 3) * r0);
        Vx_aug(:, 1:r0) = Vx_n(:,1:r);
        Vy_aug(:, 1:r0) = Vy_n(:,1:r);
    else
        % Subsequent iterations
        r1x = size(Vx_nn, 2);
        r1y = size(Vy_nn, 2);
        pp = (p + 2) * r0;
        Vx_aug = zeros(Nx, r1x + pp);
        Vy_aug = zeros(Ny, r1y + pp);
        Vx_aug(:, 1:r1x) = Vx_nn;
        Vy_aug(:, 1:r1y) = Vy_nn;
    end
    
    % Update variables using operators
    DxVxn2 = P_1 * DxVxn2;
    DyVyn2 = P_2 * DyVyn2;
    DxVxninv = P_1 \ DxVxninv;
    DyVyninv = P_2 \ DyVyninv;
    
    % Augment Vx_aug and Vy_aug
    Vx_aug(:, r1x + 1 : r1x + 2 * r0) = [DxVxninv, DxVxn2];
    Vy_aug(:, r1y + 1 : r1y + 2 * r0) = [DyVyninv, DyVyn2];
    
    % Initialize indices for augmentation
    myindodd = r1x + 2 * r0;
    myindeven = r1y + 2 * r0;
    
    % Loop over p (number of operator in generalized Sylvester equation)
    for i = 1:p
        if mod(i,2) == 1
            % For odd i, process Vx variables
            DiagVxn{i} = Dv{i} * DiagVxn{i};
            Dxinv{i} = U{i} \ (L{i} \ Dxinv{i});
            Vx_aug(:, myindodd + 1 : myindodd + 2*r0) = [Dxinv{i}, DiagVxn{i}];
            myindodd = myindodd + 2 * r0;
        else
            % For even i, process Vy variables
            DiagVyn{i} = Dv{i} * DiagVyn{i};
            Dyinv{i} = U{i} \ (L{i} \ Dyinv{i});
            Vy_aug(:, myindeven + 1 : myindeven + 2*r0) = [Dyinv{i}, DiagVyn{i}];
            myindeven = myindeven + 2 * r0;
        end
    end
    
    % QR decomposition and truncation for basis in x direction
    [Vx_nn, Rx] = qr(Vx_aug, 0);
    [t1x, Rx_svd, ~] = svd(Rx, 0);
    r11 = find(diag(Rx_svd) > epsilon_kappa, 1, 'last');
    Vx_nn = Vx_nn * t1x(:, 1:r11);
    
    % QR decomposition and truncation for basis in y direction
    [Vy_nn, Ry] = qr(Vy_aug, 0);
    [t1y, Ry_svd, ~] = svd(Ry, 0);
    r22 = find(diag(Ry_svd) > epsilon_kappa, 1, 'last');
    Vy_nn = Vy_nn * t1y(:, 1:r22);
    
  
    % Compute AV, A11, Diag, DV11 for each term
    for i = 1:p
        if mod(i,2) == 1
            % For odd i, process Vx variables
            AV{i} = A{i} * Vx_nn;
            A11{i} = Vx_nn' * AV{i};
            Diag{i} = Dv{i} * Vx_nn;
            DV11{i} = Vx_nn' * Diag{i};
        else
            % For even i, process Vy variables
            AV{i} = A{i} * Vy_nn;
            A11{i} = Vy_nn' * AV{i};
            Diag{i} = Dv{i} * Vy_nn;
            DV11{i} = Vy_nn' * Diag{i};
        end
    end
    
    % Initialize operators for Sylvester equation
    P1_tilde=Vx_nn'*P_1*Vx_nn;
    P2_tilde=Vy_nn'*P_2*Vy_nn;
    
    % Loop over stages
    for ss = 1:stages
        if ss == 1
            RHS{ss} = (Vx_nn' * Vx_n) * S_n * (Vy_n' * Vy_nn);
            TERM{ss} = 0;
        else
            RHS{ss} = RHS{1};
            for j = 1:ss-1
                RHS{ss} = RHS{ss} + a(ss, j) * TERM{j};
            end
        end
        
        % Solve the linear system using GMRES with ACS preconditioner
        [S1_vec, ~, ~, ~, ~] = gmres(@(x) afun(x, r11, r22, A11, DV11, dt, a), RHS{ss}(:), [], epsilon_GMRES, r11, @(x) mfun(x, P1_tilde, P2_tilde));
        S1 = reshape(S1_vec, r11, r22);
        
        if ss < stages
            TERM{ss} = oneovera * (S1 - RHS{ss});
        end
        
        if ss == stages % If DIRK stages are achieved
            % Compute SVD of first diagonal block and truncate
            mo = (S1 - RHS{ss});
            [t1ast, S1nast, t2ast] = svd(mo, 0);
            
            % Compute SVD of S1 and S1ast and truncate
            [t1, S1n, t2] = svd(S1, 0);
            r = find(diag(S1n) > S1n(1,1) * epsilon, 1, 'last');
            rast = find(diag(S1nast) > S1nast(1,1) * epsilon, 1, 'last');
            S1ast = S1nast(1:rast,1:rast);
            S1 = S1n(1:r,1:r);
            
            
            % Build and perform QR decompositions of residual factors 
            mymat = zeros(Nx, rast + p * (r));
            mymat2 = zeros(Ny, rast + p * (r));
            mymat(:, 1:rast) = Vx_nn*t1ast(:,1:rast);
            mymat2(:, 1:rast) = Vy_nn*t2ast(:,1:rast);
            myindx = rast;
            myindy = rast;
            
            for i = 1:p
                if mod(i,2) == 1
                    mymat(:, myindx + 1 : myindx + 2*r) = [AV{i} * t1(:,1:r), Diag{i} * t1(:,1:r)];
                    myindx = myindx + 2 * r;
                else
                    mymat2(:, myindy + 1 : myindy + 2*r) = [Diag{i} * t2(:,1:r), AV{i} * t2(:,1:r)];
                    myindy = myindy + 2 * r;
                end
            end
            
            
            [~, Rv] = qr(mymat2, 0);
            [~, Ru] = qr(mymat, 0);
            
            % Middle residual factor
            mymato = (-a(1,1) * dt * S1);
            mo = blkdiag(S1ast, kron(eye(p), mymato) );
            
            % Compute norm to check convergence
            norm1 = norm(Ru * mo * Rv','fro');
            if abs(norm1) * onenormb < epsilon_tol
                rankdatabefore = size(Vx_aug, 2);
                rankdata = size(Vx_nn, 2);
                iterdata = s;
                
                % Update outputs
                Vx_nn = Vx_nn * t1(:,1:r);
                Vy_nn = Vy_nn * t2(:,1:r);
                S_nn = S1;
                
                % Set convergence flag
                acc = true;
            end
        end
    end
    
    % If not converged, increment iteration counter
    if ~acc
        s = s + 1;
    end
end
end

% Function to compute matrix-vector product for GMRES
function y = afun(x, r1, r2, A, Dv, dt, a)
p = size(A, 2);
X = reshape(x, r1, r2);
XX = X;
for i = 1:p
    if mod(i, 2) == 1
        XX = XX - (dt * a(1,1)) * A{i} * X * Dv{i + 1}';
    else
        XX = XX - (dt * a(1,1)) * Dv{i - 1} * X * A{i}';
    end
end
y = XX(:);
end

% Function to solve Sylvester equation in GMRES preconditioner
function y = mfun(x, P1_tilde, P2_tilde)
X = reshape(x, size(P1_tilde, 1), size(P2_tilde, 2));
Y = sylvester(P1_tilde, P2_tilde', X);
y = Y(:);
end

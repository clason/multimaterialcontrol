function multimat_pot
% This function solves the nonlinear multibang control problem
%  min 1/2 \|y-yd\|^2 +
%    alpha/2\sum_i\int((u_i+u_{i+1})u(x)-u_iu_{i+1})\chi_{[u_i,u_{i+1}]}
%      s.t. -\Delta y + u = f, \partial_\nu y = 0,  u_1 <= u(x) <= u_d
% using the approach described in the paper
%    "A convex analysis approach to multi-material topology optimization"
% by Christian Clason and Karl Kunisch, see
% http://math.uni-graz.at/mobis/publications/SFB-Report-2015-011.pdf.
%
% June 1, 2015               Christian Clason <christian.clason@uni-due.de>
%                                                     http://udue.de/clason

%% setup
% problem parameters
N     = 128;            % number of nodes per dimension
maxit = 300;            % max number of Newton steps
alpha = 1e-6;           % control cost parameter (L^2)
tmin  = 1e-6;           % minimal step length
ub = [1 1.5 2 2.5]';    % vector of control states
d  = length(ub);        % number of control states

% uniform grid
x       = linspace(-1,1,N)';     % spatial grid points (uniform in x,y)
[xx,yy] = meshgrid(x);           % coordinates of nodes
h2      = (x(2)-x(1))^2;         % grid size (squared)
N2      = N*N;
tplot   = @(n,f,s) tplot_(n,f,s,N,xx,yy);

% differential operator (second order finite differences for Poisson)
D2 = spdiags(ones(N,1)*[-1 2 -1]/h2,-1:1,N,N); 
D2(1,1) = 1/h2; D2(end,end) = 1/h2;   % modify for Neumann b.c.
A = kron(speye(N,N),D2)+kron(D2,speye(N,N));

% right-hand side
f = sin(pi*xx(:)).*sin(pi*yy(:));

% target obtained from reference coefficient
ue = 1.5+(xx.^2+yy.^2 < 3/4).*(xx.^2+yy.^2 > 1/4).*(xx>1/10)...
    +(xx.^2+yy.^2 < 3/4).*(xx.^2+yy.^2 > 1/4).*(xx<-1/10);
z = (A+spdiags(ue(:),0,N2,N2))\f;
tplot(1,ue,'reference');
tplot(2,z,'target');

%% compute control
% initialize iterates
y  = zeros(N2,1);                 % state variable
p  = zeros(N2,1);                 % dual variable
as = zeros(2*d-1,N2);             % active sets

% continuation strategy
gamma = 1;
while gamma > 1e-12
    fprintf('\nCompute solution for gamma = %1.3e:\n',gamma);
    fprintf('It\tupdate\t\tresidual\tstep size\n');
       
    % semismooth Newton iteration
    it = 1;    nold = 1e99;    tau = 1;    tflag = '';
    ga = 1+2*gamma/alpha;
    while true
        % compute active sets
        as_old = as;
        q = -p(:).*y(:);
        % Q_i^gamma
        as(1,:) = (q < alpha/2*(ga*ub(1)+ub(2)));
        for i = 2:d-1
            as(i,:) = (q > alpha/2*(ub(i-1)+ga*ub(i))) & ...
                (q < alpha/2*(ga*ub(i)+ub(i+1)));
        end
        as(d,:) =  (q > alpha/2*(ub(d-1)+ga*ub(d)));
        He = as(1:d,:)'*ub;
        % Q_i,i+1^gamma
        for i = 1:d-1
            as(d+i,:) = (q >= alpha/2*(ga*ub(i)+ub(i+1))) & ...
                (q <= alpha/2*(ga*ub(i+1)+ub(i)));
            He  = He + (q-alpha/2*(ub(i)+ub(i+1))).*as(d+i,:)'/gamma;
        end
        DHe = sum(as(d+1:end,:),1)'/gamma;
        
        % gradient
        rhs = -[A*p+He.*p+(y-z(:)); A*y+He.*y-f];
        nr  = norm(rhs(:));
        
        % line search
        if nr >= nold        % if no decrease: backtrack (never on first iteration)
            tau = tau/2;
            y = y - tau*dx(1:N2);
            p = p - tau*dx(1+N2:end);
            if tau < tmin    % accept non-monotone step
                tflag = 'n';
            else             % bypass rest of while loop; compute new gradient
                continue;
            end
        end
        
        % terminate Newton?
        update = nnz((as-as_old));
        fprintf('%i\t%d\t\t%1.3e\t%1.3e%c\n',...
            it,update,nr,tau,tflag);
        if update == 0 && nr < 1e-6  % success, solution found
            break;
        elseif it == maxit           % failure, too many iterations
            break;
        end
        
        % otherwise update information, continue
        it = it+1;   nold = nr;   tau = 1;   tflag = '';
        
        % Newton matrix
        Ae  = A+spdiags(He-DHe.*p.*y,0,N2,N2);
        C   = [spdiags(1-DHe.*p.^2,0,N2,N2)   Ae; ...
               Ae'                            spdiags(-DHe.*y.^2,0,N2,N2)];

        % semismooth Newton step
        dx = C\rhs;
        y = y + dx(1:N2);
        p = p + dx(1+N2:end);
    end % Newton
    
    % check convergence
    if it < maxit                      % converged: accept iterate, continue
        u = He;                        % compute control from dual iterate
        tplot(3,u,'control');          % plot control
        regnodes = nnz(as(d+1:end,:)); % number of nodes in Q_i,i+1^gamma
        fprintf('Solution has %i node(s) in regularized active sets\n',regnodes);
        gamma = gamma/2;               % reduce gamma
    else                               % not converged: reject, terminate
        fprintf('Iterate rejected, returning u_gamma for gamma = %1.3e\n',gamma*2);
        break;
    end
    
end % continuation

%% plot results
tplot(3,u,'control');
tplot(4,y,'state');
red_cost  = (norm(ue(:))-norm(u(:)))/norm(ue(:)); % relative reduction in control cost
red_track = norm(y(:)-z(:))/norm(z(:));           % relative reduction in tracking cost
fprintf('Relative reduction in control cost: %1.3e, tracking: %1.3e\n',red_cost,red_track);
 
end % main function


function tplot_(n,f,s,N,x,y)
figure(n);
surf(x,y,reshape(f,N,N));
shading interp; lighting phong; camlight headlight; alpha(0.8);
title(s); xlabel('x_1'); ylabel('x_2');
drawnow;
end % tplot_ function


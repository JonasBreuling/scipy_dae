###############################################################################
# converted using https://www.codeconvert.ai/app
###############################################################################
"""
function varargout = ode15i(ode,tspan,y0,yp0,options,varargin)
solver_name = 'ode15i';

import matlab.internal.math.nowarn.mldivide

% Check inputs
if nargin < 5
  options = [];
  if nargin < 4
    error(message('MATLAB:ode15i:NotEnoughInputs'));
  end            
endarargin)
solver_name = 'ode15i';

import matlab.internal.math.nowarn.mldivide

% Check inputs
if nargin < 5
  options = [];
  if nargin < 4
    error(message('MATLAB:ode15i:NotEnoughInputs'));
  end            
end

% Stats
nsteps   = 0;
nfailed  = 0;
nfevals  = 0; 
npds     = 0;
ndecomps = 0;
nsolves  = 0;

ode = packageAsFuncHandle(ode);

% Output
FcnHandlesUsed  = true;   % No MATLAB v. 5 legacy. 
output_sol = (FcnHandlesUsed && (nargout==1));      % sol = odeXX(...)
output_ty  = (~output_sol && (nargout > 0));  % [t,y,...] = odeXX(...)
% There might be no output requested...

sol = []; kvec = [];  
if output_sol
  sol.solver = solver_name;
  sol.extdata.odefun = ode;
  sol.extdata.options = options;                       
  sol.extdata.varargin = varargin;  
end  
odeArgs = varargin;   

% Handle solver arguments -- pass yp0 as first extra parameter.
[neq, tspan, ntspan, next, t0, tfinal, tdir, y0, f0, ~, ...
 options, threshold, rtol, ~, ~, hmax, htry, htspan] = ...
    odearguments(FcnHandlesUsed, false, solver_name, ode, tspan, y0,  ...
                 options, [{yp0(:)} varargin]);
nfevals = nfevals + 1;

% Non-negative solution components
nonNegative = ~isempty(odeget(options,'NonNegative',[]));
if nonNegative
  warning(message('MATLAB:ode15i:NonNegativeIgnored'));   
end

% Handle the output
if nargout > 0
  outputFcn = odeget(options,'OutputFcn',[]);
else
  outputFcn = odeget(options,'OutputFcn',@odeplot);
end
outputArgs  = {};      
if isempty(outputFcn)
  haveOutputFcn = false;
else
  haveOutputFcn = true;
  outputs = odeget(options,'OutputSel',1:neq);
  outputArgs = varargin;
end
refine = max(1,odeget(options,'Refine',1));
if ntspan > 2
  outputAt = 'RequestedPoints';         % output only at tspan points
elseif refine <= 1
  outputAt = 'SolverSteps';             % computed points, no refinement
else
  outputAt = 'RefinedSteps';            % computed points, with refinement
  S = (1:refine-1) / refine;
end
printstats = strcmp(odeget(options,'Stats','off'),'on');

% Handle the event function 
[haveEventFcn,eventFcn,eventArgs,valt,teout,yeout,ieout] = ...
    odeevents(FcnHandlesUsed,ode,t0,y0,options,[{yp0(:)},varargin]);
if haveEventFcn
  eventArgs = eventArgs(2:end);  % remove yp0
end  

t = t0;
y = y0;
yp = yp0;  % Assumes consistent initial slope is supplied.

% Initialize the partial derivatives
[Jac,dfdy,dfdyp,Jconstant,dfdy_options,dfdyp_options,nfcn] = ...
    ode15ipdinit(ode,t0,y0,yp0,f0,options,varargin);

npds = npds + 1;  
nfevals = nfevals + nfcn;

PDscurrent = true;    

maxk = odeget(options,'MaxOrder',5);

% Initialize method parameters for BDFs in Lagrangian form
% and constant step size.  Column k corresponds to the formula 
% of order k.  lcf holds the leading coefficient, cf holds 
% the rest.
lcf = [  1  3/2 11/6  25/12 137/60 ];  
cf =  [ -1  -2   -3    -4     -5
         0  1/2  3/2    3      5
         0   0  -1/3  -4/3  -10/3
         0   0    0    1/4    5/4
         0   0    0     0    -1/5 ];
     
% derM(:,k) contains coefficients for calculating scaled
% derivative of order k using equally spaced mesh.       
derM = [  1  1  1  1   1   1
         -1 -2 -3 -4  -5  -6
          0  1  3  6  10  15
          0  0 -1 -4 -10 -20
          0  0  0  1   5  15
          0  0  0  0  -1  -6
          0  0  0  0   0   1 ];

maxit = 4;

% hmin is a small number such that t + hmin is clearly different from t in
% the working precision, but with this definition, it is 0 if t = 0.
hmin = 16*eps*abs(t);

if isempty(htry)
  % Compute an initial step size h using yp = y'(t0).
  wt = max(abs(y),threshold);
  rh = 1.25 * norm(yp ./ wt,inf) / sqrt(rtol);
  absh = min(hmax, htspan);
  if absh * rh > 1
    absh = 1 / rh;
  end
  absh = max(absh, hmin); 
else
  absh = min(hmax, max(hmin, htry));
end
h = tdir * absh;

% Initialize.  Set dummy value for klast to force the
% formation of iteration matrix.
k = 1;               
klast = 0;
abshlast = absh;
raised_order = false;

% For j = 1:6
%   t_{n+1-j} are stored in mesh(j)
%   y_{n+1-j} are stored in meshsol(:,j)
mesh = zeros(1,maxk+2);
mesh(1) = t0;
meshsol = zeros(neq,maxk+2);
meshsol(:,1) = y0;
% Using the initial slope, create fictitious solution at t - h for 
% starting the integration.
mesh(2) = t0 - h;
meshsol(:,2) = y0 - h*yp0;
nconh = 1;

% Allocate memory if we're generating output.
nout = 0;
tout = []; yout = [];
if nargout > 0
  if output_sol
    chunk = min(max(100,50*refine), refine+floor((2^11)/neq));      
    tout = zeros(1,chunk);
    yout = zeros(neq,chunk);
    kvec = zeros(1,chunk);
  else      
    if ntspan > 2                         % output only at tspan points
      tout = zeros(1,ntspan);
      yout = zeros(neq,ntspan);
    else                                  % alloc in chunks
      chunk = min(max(100,50*refine), refine+floor((2^13)/neq));
      tout = zeros(1,chunk);
      yout = zeros(neq,chunk);
    end
  end  
  nout = 1;
  tout(nout) = t;
  yout(:,nout) = y;  
end

% Initialize the output function.
if haveOutputFcn
  feval(outputFcn,[t tfinal],y(outputs),'init',outputArgs{:});
end

if ~isempty(odeArgs)
  ode = @(t,y,yp) ode(t,y,yp,odeArgs{:});
end

% THE MAIN LOOP

done = false;
while ~done
  
  hmin = 16*eps*abs(t);
  absh = min(hmax, max(hmin, absh));
  
  % Stretch the step if within 10% of tfinal-t.
  if 1.1*absh >= abs(tfinal - t)
    h = tfinal - t;
    absh = abs(h);
    done = true;
  end 

  % LOOP FOR ADVANCING ONE STEP.
  nfails = 0;
  while true                            % Evaluate the formula.
     
    gotynew = false;                    % is ynew evaluated yet?
    invwt = 1 ./ max(abs(y),threshold);
    while ~gotynew
      
      h = tdir * absh;
      tnew = t + h;
      if done
        tnew = tfinal;   % Hit end point exactly.
      end
      h = tnew - t;      % Purify h. 
      
      if (absh ~= abshlast) || (k ~= klast)
        if absh ~= abshlast
          nconh = 0;
        end
        Miter = dfdy + (lcf(k)/h)*dfdyp;
        if issparse(Miter)
          [L,U,P,Q,R] = lu(Miter);
        else  
          [L,U,p] = lu(Miter,'vector');
        end  
        ndecomps = ndecomps + 1;            
        havrate = false;
        rate = 1;   % Dummy value for test.
      end
      
      % Predict the solution and its derivative at tnew.
      c = weights(mesh(1:k+1),tnew,1);
      ynew  = meshsol(:,1:k+1) * c(:,1);
      ypnew = meshsol(:,1:k+1) * c(:,2);
      ypred = ynew;
      minnrm = 100*eps*norm(ypred .* invwt,inf);

      % Compute local truncation error constant.
      erconst = - 1/(k+1);
      for j = 2:k
        erconst = erconst - ...
                cf(j,k)*prod(((t - (j-1)*h) - mesh(1:k+1)) ./ (h * (1:k+1)));
      end
      erconst = abs(erconst);        

      % Iterate with simplified Newton method.
      tooslow = false;
      for iter = 1:maxit
        rhs = - ode(tnew,ynew,ypnew);
        
        % use overloaded subfunction mldivide which throws no warning.
        if issparse(Miter)
          del = Q * (U \ (L \ (P * (R \ rhs))));  
        else
          del = U \ (L \ rhs(p));
        end        
        
        newnrm = norm(del .* invwt,inf);
        ynew  = ynew + del;
        ypnew = ypnew + (lcf(k)/h) * del;
        
        if iter == 1
          if newnrm <= minnrm
            gotynew = true;
            break;
          end
          savnrm = newnrm;
        else
          rate = (newnrm/savnrm)^(1/(iter-1));
          havrate = true;
          if rate > 0.9
            tooslow = true;
            break;
          end
        end
        if havrate && ((newnrm * rate/(1 - rate)) <= 0.33*rtol)
          gotynew = true;
          break;
        elseif iter == maxit
          tooslow = true;
          break;
        end
        
      end                               % end of Newton loop
      nfevals = nfevals + iter;         
      nsolves = nsolves + iter;  

      if tooslow
        nfailed = nfailed + 1;
        abshlast = absh;
        klast = k;
        % Speed up the iteration by forming new linearization or reducing h.
        if ~PDscurrent   % always current if Jconstant                  
          if isempty(dfdy_options) && isempty(dfdyp_options)
            f = [];   % No numerical approximations formed
          else 
            f = ode(t,y,yp);
            nfevals = nfevals + 1;
          end
          [dfdy,dfdyp,dfdy_options,dfdyp_options,NF] = ...
              ode15ipdupdate(Jac,ode,t,y,yp,f,dfdy,dfdyp,dfdy_options,dfdyp_options,{});
          
          npds = npds + 1;            
          nfevals = nfevals + NF;
          PDscurrent = true;
          
          % Set a dummy value of klast to force formation of iteration matrix.
          klast = 0;
          
        elseif absh <= hmin
          warning(message('MATLAB:ode15i:IntegrationTolNotMet', sprintf( '%e', t ), sprintf( '%e', hmin )));
          solver_output = odefinalize(solver_name, sol,...
                                      outputFcn, outputArgs,...
                                      printstats, [nsteps, nfailed, nfevals,...
                                                   npds, ndecomps, nsolves],...
                                      nout, tout, yout,...
                                      haveEventFcn, teout, yeout, ieout,...
                                      {kvec,yp});
          if nargout > 0
            varargout = solver_output;
          end  
          return;
          
        else
          absh = 0.25 * absh;
          done = false; 
        end
      end   
    end     % end of while loop for getting ynew
    
    % Using the tentative solution, approximate scaled derivative 
    % used to estimate the error of the step.
    sderkp1 = norm((ynew - ypred) .* invwt,inf) * ...
              abs(prod((absh * (1:k+1)) ./ (tnew - mesh(1:k+1))));
    erropt = sderkp1 / (k+1);    % Error assuming constant step size.    
    err = sderkp1 * erconst;     % Error accounting for irregular mesh.

    % Approximate directly derivatives needed to consider lowering the
    % order.  Multiply by a power of h to get scaled derivative.
    kopt = k;
    if k > 1
      if nconh >= k
        sderk = norm(([ynew,meshsol(:,1:k)] * derM(1:k+1,k)) .* invwt,inf);
      else
        c = weights([tnew,mesh(1:k)],tnew,k);
        sderk = norm(([ynew,meshsol(:,1:k)] * c(:,k+1)) .* invwt,inf) * absh^k;
      end
      if k == 2
        if sderk <= 0.5*sderkp1
          kopt = k - 1;
          erropt = sderk / k;
        end
      else
        if nconh >= k-1
          sderkm1 = norm(([ynew,meshsol(:,1:k-1)] * derM(1:k,k-1)) .* invwt,inf);
        else
          c = weights([tnew mesh(1:k-1)],tnew,k-1);
          sderkm1 = norm(([ynew,meshsol(:,1:k-1)] * c(:,k)) .* invwt,inf) * absh^(k-1);
        end
        if max(sderkm1,sderk) <= sderkp1
          kopt = k - 1;
          erropt = sderk / k;
        end
      end
    end

    if err > rtol                       % Failed step
      nfailed = nfailed + 1;
      if absh <= hmin
        warning(message('MATLAB:ode15i:IntegrationTolNotMet', sprintf( '%e', t ), sprintf( '%e', hmin )));
        solver_output = odefinalize(solver_name, sol,...
                                    outputFcn, outputArgs,...
                                    printstats, [nsteps, nfailed, nfevals,...
                                                 npds, ndecomps, nsolves],...
                                    nout, tout, yout,...
                                    haveEventFcn, teout, yeout, ieout,...
                                    {kvec,yp});
        if nargout > 0
          varargout = solver_output;
        end  
        return;
      end
            
      abshlast = absh;
      klast = k;
      nfails = nfails + 1;
      switch nfails
      case 1
        absh = absh * min(0.9,max(0.25, 0.9*(0.5*rtol/erropt)^(1/(kopt+1)))); 
      case 2
        absh = absh * 0.25;
      otherwise
        kopt = 1;
        absh = absh * 0.25;
      end
      absh = max(absh,hmin);
      if absh < abshlast
        done = false;
      end
      k = kopt;      
      
    else                                % Successful step
      break;
      
    end
  end % while true
  nsteps = nsteps + 1;   
    
  if haveEventFcn
    events_args = {eventFcn,tnew,ynew,ypnew,mesh(1:k),meshsol(:,1:k),eventArgs{:}}; %#ok<CCAT>
    [te,ye,ie,valt,stop] = odezero(@ntrp15i,@events_aux,events_args,valt,...
                                   t,y,tnew,ynew,t0,mesh(1:k),meshsol(:,1:k));
    if ~isempty(te)
      if output_sol || (nargout > 2)
        teout = [teout, te]; %#ok<AGROW>
        yeout = [yeout, ye]; %#ok<AGROW>
        ieout = [ieout, ie]; %#ok<AGROW>
      end
      if stop               % Stop on a terminal event.               
        % Adjust the interpolation data to [t te(end)].                 
        tnew = te(end);
        ynew = ye(:,end);
        done = true;
      end
    end
  end
  
  if output_sol
    nout = nout + 1;
    if nout > length(tout)
      tout = [tout, zeros(1,chunk)]; %#ok<AGROW> requires chunk >= refine
      yout = [yout, zeros(neq,chunk)]; %#ok<AGROW>
      kvec = [kvec, zeros(1,chunk)]; %#ok<AGROW>
    end
    tout(nout) = tnew; %#ok<AGROW>
    yout(:,nout) = ynew; %#ok<AGROW>
    kvec(nout) = k; %#ok<AGROW>
  end  

  if output_ty || haveOutputFcn 
    switch outputAt
     case 'SolverSteps'        % computed points, no refinement
      nout_new = 1;
      tout_new = tnew;
      yout_new = ynew;
     case 'RefinedSteps'       % computed points, with refinement
      tref = t + (tnew-t)*S;
      nout_new = refine;
      tout_new = [tref, tnew];
      yout_new = [ntrp15i(tref,[],[],tnew,ynew,mesh(1:k),meshsol(:,1:k)), ynew];
     case 'RequestedPoints'    % output only at tspan points
      nout_new =  0;
      tout_new = [];
      yout_new = [];
      while next <= ntspan  
        if tdir * (tnew - tspan(next)) < 0
          if haveEventFcn && stop     % output tstop,ystop
            nout_new = nout_new + 1;
            tout_new = [tout_new, tnew]; %#ok<AGROW>
            yout_new = [yout_new, ynew]; %#ok<AGROW>
          end
          break;
        end
        nout_new = nout_new + 1;              
        tout_new = [tout_new, tspan(next)]; %#ok<AGROW>
        if tspan(next) == tnew
          yout_new = [yout_new, ynew]; %#ok<AGROW>
        else  
          yout_new = [yout_new, ntrp15i(tspan(next),[],[],tnew,ynew,...
                                        mesh(1:k),meshsol(:,1:k))]; %#ok<AGROW>
        end  
        next = next + 1;
      end
    end
    
    if nout_new > 0
      if output_ty
        oldnout = nout;
        nout = nout + nout_new;
        if nout > length(tout)
          tout = [tout, zeros(1,chunk)]; %#ok<AGROW> requires chunk >= refine
          yout = [yout, zeros(neq,chunk)]; %#ok<AGROW>
        end
        idx = oldnout+1:nout;        
        tout(idx) = tout_new; %#ok<AGROW>
        yout(:,idx) = yout_new; %#ok<AGROW>
      end
      if haveOutputFcn
        stop = feval(outputFcn,tout_new,yout_new(outputs,:),'',outputArgs{:});
        if stop
          done = true;
        end  
      end     
    end  
  end
  
  if done
    break
  end
    
  % Advance the integration one step.
  t = tnew;
  y = ynew;
  yp = ypnew;
  mesh = [t mesh(1:end-1)];
  meshsol = [y meshsol(:,1:end-1)]; 
  PDscurrent = Jconstant;
  
  klast = k;
  abshlast = absh;
  nconh = min(nconh+1,maxk+2);

  % Estimate the scaled derivative of order k+2 if 
  %  *  at constant step size, 
  %  *  have not already decided to reduce the order, 
  %  *  not already at the maximum order, and
  %  *  did not raise order on last step. 
  if (nconh >= k + 2) && ~(kopt < k) && ~(k == maxk) && ~raised_order
    sderkp2 = norm((meshsol(:,1:k+3) * derM(1:k+3,k+2)) .* invwt,inf);
    if (k > 1) && (sderk <= min(sderkp1,sderkp2))
      kopt = k - 1;
      erropt = sderk / k;
    elseif ((k == 1) && (sderkp2 < 0.5*sderkp1)) || ...
           ((k > 1) && (sderkp2 < sderkp1))
      kopt = k + 1;
      erropt = sderkp2 / (k + 2);
    end      
  end
  temp = (erropt/(0.5*rtol))^(1/(kopt+1));  % hopt = absh/temp
  if temp <= 1/2 
    absh = absh * 2;
  elseif temp > 1
    absh = absh * max(0.5,min(0.9,1/temp));
  end
  raised_order = kopt > k;
  k = kopt;
    
end % while ~done

solver_output = odefinalize(solver_name, sol,...
                            outputFcn, outputArgs,...
                            printstats, [nsteps, nfailed, nfevals,...
                                         npds, ndecomps, nsolves],...
                            nout, tout, yout,...
                            haveEventFcn, teout, yeout, ieout,...
                            {kvec,yp});
if nargout > 0
  varargout = solver_output;
end  

% -----------------------------
function c = weights(x,xi,maxder)
% Compute Lagrangian interpolation coeffients c for the value at xi 
% of a polynomial interpolating at distinct nodes x(1),...,x(N) and
% derivatives of the polynomial of orders 0,...,maxder.  c(j,d+1) 
% is the coefficient of the function value corresponding to x(j) when
% computing the derivative of order d.  Note that maxder <= N-1.
%
% This program is based on the Fortran code WEIGHTS1 of B. Fornberg, 
% A Practical Guide to to Pseudospectral Methods, Cambridge University
% Press, 1995.

n = length(x) - 1;
c = zeros(n+1,maxder+1);
c(1,1) = 1;
tmp1 = 1;
tmp4 = x(1) - xi;
for i = 1:n
    mn = min(i,maxder);
    tmp2 = 1;
    tmp5 = tmp4;
    tmp4 = x(i+1) - xi;
    for j = 0:i-1
        tmp3 = x(i+1) - x(j+1);
        tmp2 = tmp2*tmp3;
        if j == i-1
          for k = mn:-1:1
            c(i+1,k+1) = tmp1*(k*c(i,k) - tmp5*c(i,k+1))/tmp2;
          end
          c(i+1,1) = - tmp1*tmp5*c(i,1)/tmp2;
        end
        for k = mn:-1:1
          c(j+1,k+1) = (tmp4*c(j+1,k+1) - k*c(j+1,k))/tmp3;
        end
        c(j+1,1) = tmp4*c(j+1,1)/tmp3;
    end
    tmp1 = tmp2;
end

sol = []; kvec = [];  
if output_sol
  sol.solver = solver_name;
  sol.extdata.odefun = ode;
  sol.extdata.options = options;                       
  sol.extdata.varargin = varargin;  
end  
odeArgs = varargin;   

% Handle solver arguments -- pass yp0 as first extra parameter.
[neq, tspan, ntspan, next, t0, tfinal, tdir, y0, f0, ~, ...
 options, threshold, rtol, ~, ~, hmax, htry, htspan] = ...
    odearguments(FcnHandlesUsed, false, solver_name, ode, tspan, y0,  ...
                 options, [{yp0(:)} varargin]);
nfevals = nfevals + 1;

% Non-negative solution components
nonNegative = ~isempty(odeget(options,'NonNegative',[]));
if nonNegative
  warning(message('MATLAB:ode15i:NonNegativeIgnored'));   
end

% Handle the output
if nargout > 0
  outputFcn = odeget(options,'OutputFcn',[]);
else
  outputFcn = odeget(options,'OutputFcn',@odeplot);
end
outputArgs  = {};      
if isempty(outputFcn)
  haveOutputFcn = false;
else
  haveOutputFcn = true;
  outputs = odeget(options,'OutputSel',1:neq);
  outputArgs = varargin;
end
refine = max(1,odeget(options,'Refine',1));
if ntspan > 2
  outputAt = 'RequestedPoints';         % output only at tspan points
elseif refine <= 1
  outputAt = 'SolverSteps';             % computed points, no refinement
else
  outputAt = 'RefinedSteps';            % computed points, with refinement
  S = (1:refine-1) / refine;
end
printstats = strcmp(odeget(options,'Stats','off'),'on');

% Handle the event function 
[haveEventFcn,eventFcn,eventArgs,valt,teout,yeout,ieout] = ...
    odeevents(FcnHandlesUsed,ode,t0,y0,options,[{yp0(:)},varargin]);
if haveEventFcn
  eventArgs = eventArgs(2:end);  % remove yp0
end  

t = t0;
y = y0;
yp = yp0;  % Assumes consistent initial slope is supplied.

% Initialize the partial derivatives
[Jac,dfdy,dfdyp,Jconstant,dfdy_options,dfdyp_options,nfcn] = ...
    ode15ipdinit(ode,t0,y0,yp0,f0,options,varargin);

npds = npds + 1;  
nfevals = nfevals + nfcn;

PDscurrent = true;    

maxk = odeget(options,'MaxOrder',5);

% Initialize method parameters for BDFs in Lagrangian form
% and constant step size.  Column k corresponds to the formula 
% of order k.  lcf holds the leading coefficient, cf holds 
% the rest.
lcf = [  1  3/2 11/6  25/12 137/60 ];  
cf =  [ -1  -2   -3    -4     -5
         0  1/2  3/2    3      5
         0   0  -1/3  -4/3  -10/3
         0   0    0    1/4    5/4
         0   0    0     0    -1/5 ];
     
% derM(:,k) contains coefficients for calculating scaled
% derivative of order k using equally spaced mesh.       
derM = [  1  1  1  1   1   1
         -1 -2 -3 -4  -5  -6
          0  1  3  6  10  15
          0  0 -1 -4 -10 -20
          0  0  0  1   5  15
          0  0  0  0  -1  -6
          0  0  0  0   0   1 ];

maxit = 4;

% hmin is a small number such that t + hmin is clearly different from t in
% the working precision, but with this definition, it is 0 if t = 0.
hmin = 16*eps*abs(t);

if isempty(htry)
  % Compute an initial step size h using yp = y'(t0).
  wt = max(abs(y),threshold);
  rh = 1.25 * norm(yp ./ wt,inf) / sqrt(rtol);
  absh = min(hmax, htspan);
  if absh * rh > 1
    absh = 1 / rh;
  end
  absh = max(absh, hmin); 
else
  absh = min(hmax, max(hmin, htry));
end
h = tdir * absh;

% Initialize.  Set dummy value for klast to force the
% formation of iteration matrix.
k = 1;               
klast = 0;
abshlast = absh;
raised_order = false;

% For j = 1:6
%   t_{n+1-j} are stored in mesh(j)
%   y_{n+1-j} are stored in meshsol(:,j)
mesh = zeros(1,maxk+2);
mesh(1) = t0;
meshsol = zeros(neq,maxk+2);
meshsol(:,1) = y0;
% Using the initial slope, create fictitious solution at t - h for 
% starting the integration.
mesh(2) = t0 - h;
meshsol(:,2) = y0 - h*yp0;
nconh = 1;

% Allocate memory if we're generating output.
nout = 0;
tout = []; yout = [];
if nargout > 0
  if output_sol
    chunk = min(max(100,50*refine), refine+floor((2^11)/neq));      
    tout = zeros(1,chunk);
    yout = zeros(neq,chunk);
    kvec = zeros(1,chunk);
  else      
    if ntspan > 2                         % output only at tspan points
      tout = zeros(1,ntspan);
      yout = zeros(neq,ntspan);
    else                                  % alloc in chunks
      chunk = min(max(100,50*refine), refine+floor((2^13)/neq));
      tout = zeros(1,chunk);
      yout = zeros(neq,chunk);
    end
  end  
  nout = 1;
  tout(nout) = t;
  yout(:,nout) = y;  
end

% Initialize the output function.
if haveOutputFcn
  feval(outputFcn,[t tfinal],y(outputs),'init',outputArgs{:});
end

if ~isempty(odeArgs)
  ode = @(t,y,yp) ode(t,y,yp,odeArgs{:});
end

% THE MAIN LOOP

done = false;
while ~done
  
  hmin = 16*eps*abs(t);
  absh = min(hmax, max(hmin, absh));
  
  % Stretch the step if within 10% of tfinal-t.
  if 1.1*absh >= abs(tfinal - t)
    h = tfinal - t;
    absh = abs(h);
    done = true;
  end 

  % LOOP FOR ADVANCING ONE STEP.
  nfails = 0;
  while true                            % Evaluate the formula.
     
    gotynew = false;                    % is ynew evaluated yet?
    invwt = 1 ./ max(abs(y),threshold);
    while ~gotynew
      
      h = tdir * absh;
      tnew = t + h;
      if done
        tnew = tfinal;   % Hit end point exactly.
      end
      h = tnew - t;      % Purify h. 
      
      if (absh ~= abshlast) || (k ~= klast)
        if absh ~= abshlast
          nconh = 0;
        end
        Miter = dfdy + (lcf(k)/h)*dfdyp;
        if issparse(Miter)
          [L,U,P,Q,R] = lu(Miter);
        else  
          [L,U,p] = lu(Miter,'vector');
        end  
        ndecomps = ndecomps + 1;            
        havrate = false;
        rate = 1;   % Dummy value for test.
      end
      
      % Predict the solution and its derivative at tnew.
      c = weights(mesh(1:k+1),tnew,1);
      ynew  = meshsol(:,1:k+1) * c(:,1);
      ypnew = meshsol(:,1:k+1) * c(:,2);
      ypred = ynew;
      minnrm = 100*eps*norm(ypred .* invwt,inf);

      % Compute local truncation error constant.
      erconst = - 1/(k+1);
      for j = 2:k
        erconst = erconst - ...
                cf(j,k)*prod(((t - (j-1)*h) - mesh(1:k+1)) ./ (h * (1:k+1)));
      end
      erconst = abs(erconst);        

      % Iterate with simplified Newton method.
      tooslow = false;
      for iter = 1:maxit
        rhs = - ode(tnew,ynew,ypnew);
        
        % use overloaded subfunction mldivide which throws no warning.
        if issparse(Miter)
          del = Q * (U \ (L \ (P * (R \ rhs))));  
        else
          del = U \ (L \ rhs(p));
        end        
        
        newnrm = norm(del .* invwt,inf);
        ynew  = ynew + del;
        ypnew = ypnew + (lcf(k)/h) * del;
        
        if iter == 1
          if newnrm <= minnrm
            gotynew = true;
            break;
          end
          savnrm = newnrm;
        else
          rate = (newnrm/savnrm)^(1/(iter-1));
          havrate = true;
          if rate > 0.9
            tooslow = true;
            break;
          end
        end
        if havrate && ((newnrm * rate/(1 - rate)) <= 0.33*rtol)
          gotynew = true;
          break;
        elseif iter == maxit
          tooslow = true;
          break;
        end
        
      end                               % end of Newton loop
      nfevals = nfevals + iter;         
      nsolves = nsolves + iter;  

      if tooslow
        nfailed = nfailed + 1;
        abshlast = absh;
        klast = k;
        % Speed up the iteration by forming new linearization or reducing h.
        if ~PDscurrent   % always current if Jconstant                  
          if isempty(dfdy_options) && isempty(dfdyp_options)
            f = [];   % No numerical approximations formed
          else 
            f = ode(t,y,yp);
            nfevals = nfevals + 1;
          end
          [dfdy,dfdyp,dfdy_options,dfdyp_options,NF] = ...
              ode15ipdupdate(Jac,ode,t,y,yp,f,dfdy,dfdyp,dfdy_options,dfdyp_options,{});
          
          npds = npds + 1;            
          nfevals = nfevals + NF;
          PDscurrent = true;
          
          % Set a dummy value of klast to force formation of iteration matrix.
          klast = 0;
          
        elseif absh <= hmin
          warning(message('MATLAB:ode15i:IntegrationTolNotMet', sprintf( '%e', t ), sprintf( '%e', hmin )));
          solver_output = odefinalize(solver_name, sol,...
                                      outputFcn, outputArgs,...
                                      printstats, [nsteps, nfailed, nfevals,...
                                                   npds, ndecomps, nsolves],...
                                      nout, tout, yout,...
                                      haveEventFcn, teout, yeout, ieout,...
                                      {kvec,yp});
          if nargout > 0
            varargout = solver_output;
          end  
          return;
          
        else
          absh = 0.25 * absh;
          done = false; 
        end
      end   
    end     % end of while loop for getting ynew
    
    % Using the tentative solution, approximate scaled derivative 
    % used to estimate the error of the step.
    sderkp1 = norm((ynew - ypred) .* invwt,inf) * ...
              abs(prod((absh * (1:k+1)) ./ (tnew - mesh(1:k+1))));
    erropt = sderkp1 / (k+1);    % Error assuming constant step size.    
    err = sderkp1 * erconst;     % Error accounting for irregular mesh.

    % Approximate directly derivatives needed to consider lowering the
    % order.  Multiply by a power of h to get scaled derivative.
    kopt = k;
    if k > 1
      if nconh >= k
        sderk = norm(([ynew,meshsol(:,1:k)] * derM(1:k+1,k)) .* invwt,inf);
      else
        c = weights([tnew,mesh(1:k)],tnew,k);
        sderk = norm(([ynew,meshsol(:,1:k)] * c(:,k+1)) .* invwt,inf) * absh^k;
      end
      if k == 2
        if sderk <= 0.5*sderkp1
          kopt = k - 1;
          erropt = sderk / k;
        end
      else
        if nconh >= k-1
          sderkm1 = norm(([ynew,meshsol(:,1:k-1)] * derM(1:k,k-1)) .* invwt,inf);
        else
          c = weights([tnew mesh(1:k-1)],tnew,k-1);
          sderkm1 = norm(([ynew,meshsol(:,1:k-1)] * c(:,k)) .* invwt,inf) * absh^(k-1);
        end
        if max(sderkm1,sderk) <= sderkp1
          kopt = k - 1;
          erropt = sderk / k;
        end
      end
    end

    if err > rtol                       % Failed step
      nfailed = nfailed + 1;
      if absh <= hmin
        warning(message('MATLAB:ode15i:IntegrationTolNotMet', sprintf( '%e', t ), sprintf( '%e', hmin )));
        solver_output = odefinalize(solver_name, sol,...
                                    outputFcn, outputArgs,...
                                    printstats, [nsteps, nfailed, nfevals,...
                                                 npds, ndecomps, nsolves],...
                                    nout, tout, yout,...
                                    haveEventFcn, teout, yeout, ieout,...
                                    {kvec,yp});
        if nargout > 0
          varargout = solver_output;
        end  
        return;
      end
            
      abshlast = absh;
      klast = k;
      nfails = nfails + 1;
      switch nfails
      case 1
        absh = absh * min(0.9,max(0.25, 0.9*(0.5*rtol/erropt)^(1/(kopt+1)))); 
      case 2
        absh = absh * 0.25;
      otherwise
        kopt = 1;
        absh = absh * 0.25;
      end
      absh = max(absh,hmin);
      if absh < abshlast
        done = false;
      end
      k = kopt;      
      
    else                                % Successful step
      break;
      
    end
  end % while true
  nsteps = nsteps + 1;   
    
  if haveEventFcn
    events_args = {eventFcn,tnew,ynew,ypnew,mesh(1:k),meshsol(:,1:k),eventArgs{:}}; %#ok<CCAT>
    [te,ye,ie,valt,stop] = odezero(@ntrp15i,@events_aux,events_args,valt,...
                                   t,y,tnew,ynew,t0,mesh(1:k),meshsol(:,1:k));
    if ~isempty(te)
      if output_sol || (nargout > 2)
        teout = [teout, te]; %#ok<AGROW>
        yeout = [yeout, ye]; %#ok<AGROW>
        ieout = [ieout, ie]; %#ok<AGROW>
      end
      if stop               % Stop on a terminal event.               
        % Adjust the interpolation data to [t te(end)].                 
        tnew = te(end);
        ynew = ye(:,end);
        done = true;
      end
    end
  end
  
  if output_sol
    nout = nout + 1;
    if nout > length(tout)
      tout = [tout, zeros(1,chunk)]; %#ok<AGROW> requires chunk >= refine
      yout = [yout, zeros(neq,chunk)]; %#ok<AGROW>
      kvec = [kvec, zeros(1,chunk)]; %#ok<AGROW>
    end
    tout(nout) = tnew; %#ok<AGROW>
    yout(:,nout) = ynew; %#ok<AGROW>
    kvec(nout) = k; %#ok<AGROW>
  end  

  if output_ty || haveOutputFcn 
    switch outputAt
     case 'SolverSteps'        % computed points, no refinement
      nout_new = 1;
      tout_new = tnew;
      yout_new = ynew;
     case 'RefinedSteps'       % computed points, with refinement
      tref = t + (tnew-t)*S;
      nout_new = refine;
      tout_new = [tref, tnew];
      yout_new = [ntrp15i(tref,[],[],tnew,ynew,mesh(1:k),meshsol(:,1:k)), ynew];
     case 'RequestedPoints'    % output only at tspan points
      nout_new =  0;
      tout_new = [];
      yout_new = [];
      while next <= ntspan  
        if tdir * (tnew - tspan(next)) < 0
          if haveEventFcn && stop     % output tstop,ystop
            nout_new = nout_new + 1;
            tout_new = [tout_new, tnew]; %#ok<AGROW>
            yout_new = [yout_new, ynew]; %#ok<AGROW>
          end
          break;
        end
        nout_new = nout_new + 1;              
        tout_new = [tout_new, tspan(next)]; %#ok<AGROW>
        if tspan(next) == tnew
          yout_new = [yout_new, ynew]; %#ok<AGROW>
        else  
          yout_new = [yout_new, ntrp15i(tspan(next),[],[],tnew,ynew,...
                                        mesh(1:k),meshsol(:,1:k))]; %#ok<AGROW>
        end  
        next = next + 1;
      end
    end
    
    if nout_new > 0
      if output_ty
        oldnout = nout;
        nout = nout + nout_new;
        if nout > length(tout)
          tout = [tout, zeros(1,chunk)]; %#ok<AGROW> requires chunk >= refine
          yout = [yout, zeros(neq,chunk)]; %#ok<AGROW>
        end
        idx = oldnout+1:nout;        
        tout(idx) = tout_new; %#ok<AGROW>
        yout(:,idx) = yout_new; %#ok<AGROW>
      end
      if haveOutputFcn
        stop = feval(outputFcn,tout_new,yout_new(outputs,:),'',outputArgs{:});
        if stop
          done = true;
        end  
      end     
    end  
  end
  
  if done
    break
  end
    
  % Advance the integration one step.
  t = tnew;
  y = ynew;
  yp = ypnew;
  mesh = [t mesh(1:end-1)];
  meshsol = [y meshsol(:,1:end-1)]; 
  PDscurrent = Jconstant;
  
  klast = k;
  abshlast = absh;
  nconh = min(nconh+1,maxk+2);

  % Estimate the scaled derivative of order k+2 if 
  %  *  at constant step size, 
  %  *  have not already decided to reduce the order, 
  %  *  not already at the maximum order, and
  %  *  did not raise order on last step. 
  if (nconh >= k + 2) && ~(kopt < k) && ~(k == maxk) && ~raised_order
    sderkp2 = norm((meshsol(:,1:k+3) * derM(1:k+3,k+2)) .* invwt,inf);
    if (k > 1) && (sderk <= min(sderkp1,sderkp2))
      kopt = k - 1;
      erropt = sderk / k;
    elseif ((k == 1) && (sderkp2 < 0.5*sderkp1)) || ...
           ((k > 1) && (sderkp2 < sderkp1))
      kopt = k + 1;
      erropt = sderkp2 / (k + 2);
    end      
  end
  temp = (erropt/(0.5*rtol))^(1/(kopt+1));  % hopt = absh/temp
  if temp <= 1/2 
    absh = absh * 2;
  elseif temp > 1
    absh = absh * max(0.5,min(0.9,1/temp));
  end
  raised_order = kopt > k;
  k = kopt;
    
end % while ~done

solver_output = odefinalize(solver_name, sol,...
                            outputFcn, outputArgs,...
                            printstats, [nsteps, nfailed, nfevals,...
                                         npds, ndecomps, nsolves],...
                            nout, tout, yout,...
                            haveEventFcn, teout, yeout, ieout,...
                            {kvec,yp});
if nargout > 0
  varargout = solver_output;
end  

% -----------------------------
function c = weights(x,xi,maxder)
% Compute Lagrangian interpolation coeffients c for the value at xi 
% of a polynomial interpolating at distinct nodes x(1),...,x(N) and
% derivatives of the polynomial of orders 0,...,maxder.  c(j,d+1) 
% is the coefficient of the function value corresponding to x(j) when
% computing the derivative of order d.  Note that maxder <= N-1.
%
% This program is based on the Fortran code WEIGHTS1 of B. Fornberg, 
% A Practical Guide to to Pseudospectral Methods, Cambridge University
% Press, 1995.

n = length(x) - 1;
c = zeros(n+1,maxder+1);
c(1,1) = 1;
tmp1 = 1;
tmp4 = x(1) - xi;
for i = 1:n
    mn = min(i,maxder);
    tmp2 = 1;
    tmp5 = tmp4;
    tmp4 = x(i+1) - xi;
    for j = 0:i-1
        tmp3 = x(i+1) - x(j+1);
        tmp2 = tmp2*tmp3;
        if j == i-1
          for k = mn:-1:1
            c(i+1,k+1) = tmp1*(k*c(i,k) - tmp5*c(i,k+1))/tmp2;
          end
          c(i+1,1) = - tmp1*tmp5*c(i,1)/tmp2;
        end
        for k = mn:-1:1
          c(j+1,k+1) = (tmp4*c(j+1,k+1) - k*c(j+1,k))/tmp3;
        end
        c(j+1,1) = tmp4*c(j+1,1)/tmp3;
    end
    tmp1 = tmp2;
end
"""
###############################################################################
###############################################################################

###############################################################################
# yielding 
###############################################################################
"""
def ode15i(ode, tspan, y0, yp0, options, *varargin):
    solver_name = 'ode15i'

    from scipy.sparse import issparse
    from scipy.sparse.linalg import splu
    import numpy as np

    # Check inputs
    if len(options) == 0:
        options = []
        if len(options) < 4:
            raise Exception('MATLAB:ode15i:NotEnoughInputs')

    # Stats
    nsteps = 0
    nfailed = 0
    nfevals = 0
    npds = 0
    ndecomps = 0
    nsolves = 0

    # Output
    FcnHandlesUsed = True  # No MATLAB v. 5 legacy.
    output_sol = (FcnHandlesUsed and (len(varargout) == 1))  # sol = odeXX(...)
    output_ty = (not output_sol and (len(varargout) > 0))  # [t,y,...] = odeXX(...)
    # There might be no output requested...

    sol = []
    kvec = []
    if output_sol:
        sol['solver'] = solver_name
        sol['extdata']['odefun'] = ode
        sol['extdata']['options'] = options
        sol['extdata']['varargin'] = varargin
    odeArgs = varargin

    # Handle solver arguments -- pass yp0 as first extra parameter.
    neq, tspan, ntspan, next, t0, tfinal, tdir, y0, f0, options, threshold, rtol, hmax, htry, htspan = \
        odearguments(FcnHandlesUsed, False, solver_name, ode, tspan, y0, options, [yp0, *varargin])
    nfevals = nfevals + 1

    nonNegative = (len(odeget(options, 'NonNegative', [])) != 0)
    if nonNegative:
        print('MATLAB:ode15i:NonNegativeIgnored')

    if len(odeArgs) != 0:
        ode = lambda t, y, yp: ode(t, y, yp, *odeArgs)

    # THE MAIN LOOP

    done = False
    while not done:

        hmin = 16 * np.finfo(float).eps * abs(t)
        absh = min(hmax, max(hmin, absh))

        # Stretch the step if within 10% of tfinal-t.
        if 1.1 * absh >= abs(tfinal - t):
            h = tfinal - t
            absh = abs(h)
            done = True

        # LOOP FOR ADVANCING ONE STEP.
        nfails = 0
        while True:  # Evaluate the formula.

            gotynew = False  # is ynew evaluated yet?
            invwt = 1 / max(abs(y), threshold)
            while not gotynew:

                h = tdir * absh
                tnew = t + h
                if done:
                    tnew = tfinal  # Hit end point exactly.
                h = tnew - t  # Purify h.

                if (absh != abshlast) or (k != klast):
                    if absh != abshlast:
                        nconh = 0
                    Miter = dfdy + (lcf[k] / h) * dfdyp
                    if issparse(Miter):
                        LU = splu(Miter.tocsc())
                    else:
                        LU = splu(Miter)
                    ndecomps = ndecomps + 1
                    havrate = False
                    rate = 1  # Dummy value for test.

                # Predict the solution and its derivative at tnew.
                c = weights(mesh[0:k + 1], tnew, 1)
                ynew = np.dot(meshsol[:, 0:k + 1], c[:, 0])
                ypnew = np.dot(meshsol[:, 0:k + 1], c[:, 1])
                ypred = ynew
                minnrm = 100 * np.finfo(float).eps * np.linalg.norm(ypred * invwt, np.inf)

                # Compute local truncation error constant.
                erconst = -1 / (k + 1)
                for j in range(1, k + 1):
                    erconst = erconst - cf[j, k] * np.prod(
                        ((t - (j - 1) * h) - mesh[0:k + 1]) / (h * np.arange(1, k + 2)))
                erconst = abs(erconst)

                # Iterate with simplified Newton method.
                tooslow = False
                for iter in range(1, maxit + 1):
                    rhs = -ode(tnew, ynew, ypnew)

                    # use overloaded subfunction mldivide which throws no warning.
                    if issparse(Miter):
                        del_ = LU.solve(rhs)
                    else:
                        del_ = LU.solve(rhs)

                    newnrm = np.linalg.norm(del_ * invwt, np.inf)
                    ynew = ynew + del_
                    ypnew = ypnew + (lcf[k] / h) * del_

                    if iter == 1:
                        if newnrm <= minnrm:
                            gotynew = True
                            break
                        savnrm = newnrm
                    else:
                        rate = (newnrm / savnrm) ** (1 / (iter - 1))
                        havrate = True
                        if rate > 0.9:
                            tooslow = True
                            break
                    if havrate and ((newnrm * rate / (1 - rate)) <= 0.33 * rtol):
                        gotynew = True
                        break
                    elif iter == maxit:
                        tooslow = True
                        break

                nfevals = nfevals + iter
                nsolves = nsolves + iter

                if tooslow:
                    nfailed = nfailed + 1
                    abshlast = absh
                    klast = k
                    # Speed up the iteration by forming new linearization or reducing h.
                    if not PDscurrent:  # always current if Jconstant
                        if (len(dfdy_options) == 0) and (len(dfdyp_options) == 0):
                            f = []  # No numerical approximations formed
                        else:
                            f = ode(t, y, yp)
                            nfevals = nfevals + 1
                        dfdy, dfdyp, dfdy_options, dfdyp_options, NF = \
                            ode15ipdupdate(Jac, ode, t, y, yp, f, dfdy, dfdyp, dfdy_options, dfdyp_options, [])
                        npds = npds + 1
                        nfevals = nfevals + NF
                        PDscurrent = True

                        # Set a dummy value of klast to force formation of iteration matrix.
                        klast = 0

                    elif absh <= hmin:
                        print('MATLAB:ode15i:IntegrationTolNotMet', t, hmin)
                        solver_output = odefinalize(solver_name, sol,
                                                    outputFcn, outputArgs,
                                                    printstats, [nsteps, nfailed, nfevals,
                                                                npds, ndecomps, nsolves],
                                                    nout, tout, yout,
                                                    haveEventFcn, teout, yeout, ieout,
                                                    [kvec, yp])
                        if len(varargout) > 0:
                            return solver_output

                    else:
                        absh = 0.25 * absh
                        done = False

            # Using the tentative solution, approximate scaled derivative
            # used to estimate the error of the step.
            sderkp1 = np.linalg.norm((ynew - ypred) * invwt, np.inf) * \
                      abs(np.prod((absh * np.arange(1, k + 2)) / (tnew - mesh[0:k + 1])))

            erropt = sderkp1 / (k + 1)  # Error assuming constant step size.
            err = sderkp1 * erconst  # Error accounting for irregular mesh.

            # Approximate directly derivatives needed to consider lowering the
            # order.  Multiply by a power of h to get scaled derivative.
            kopt = k
            if k > 1:
                if nconh >= k:
                    sderk = np.linalg.norm(
                        (np.dot(np.concatenate((ynew, meshsol[:, 0:k]), axis=1), derM[0:k + 1, k])) * invwt, np.inf)
                else:
                    c = weights(np.concatenate((tnew, mesh[0:k]), axis=1), tnew, k)
                    sderk = np.linalg.norm(
                        (np.dot(np.concatenate((ynew, meshsol[:, 0:k]), axis=1), c[:, k])) * invwt, np.inf) * absh ** k
                if k == 2:
                    if sderk <= 0.5 * sderkp1:
                        kopt = k - 1
                        erropt = sderk / k
                else:
                    if nconh >= k - 1:
                        sderkm1 = np.linalg.norm(
                            (np.dot(np.concatenate((ynew, meshsol[:, 0:k - 1]), axis=1), derM[0:k, k - 1])) * invwt,
                            np.inf)
                    else:
                        c = weights(np.concatenate((tnew, mesh[0:k - 1]), axis=1), tnew, k - 1)
                        sderkm1 = np.linalg.norm(
                            (np.dot(np.concatenate((ynew, meshsol[:, 0:k - 1]), axis=1), c[:, k])) * invwt,
                            np.inf) * absh ** (k - 1)
                    if max(sderkm1, sderk) <= sderkp1:
                        kopt = k - 1
                        erropt = sderk / k

            if err > rtol:  # Failed step
                nfailed = nfailed + 1
                if absh <= hmin:
                    print('MATLAB:ode15i:IntegrationTolNotMet', t, hmin)
                    solver_output = odefinalize(solver_name, sol,
                                                outputFcn, outputArgs,
                                                printstats, [nsteps, nfailed, nfevals,
                                                            npds, ndecomps, nsolves],
                                                nout, tout, yout,
                                                haveEventFcn, teout, yeout, ieout,
                                                [kvec, yp])
                    if len(varargout) > 0:
                        return solver_output

                abshlast = absh
                klast = k
                nfails = nfails + 1
                if nfails == 1:
                    absh = absh * min(0.9, max(0.25, 0.9 * (0.5 * rtol / erropt) ** (1 / (kopt + 1))))
                elif nfails == 2:
                    absh = absh * 0.25
                else:
                    kopt = 1
                    absh = absh * 0.25
                absh = max(absh, hmin)
                if absh < abshlast:
                    done = False
                k = kopt

            else:  # Successful step
                break

        nsteps = nsteps + 1

        if haveEventFcn:
            events_args = [eventFcn, tnew, ynew, ypnew, mesh[0:k], meshsol[:, 0:k], *eventArgs]
            te, ye, ie, valt, stop = odezero(ntrp15i, events_aux, events_args, valt,
                                             t, y, tnew, ynew, t0, mesh[0:k], meshsol[:, 0:k])
            if len(te) != 0:
                if output_sol or (len(varargout) > 2):
                    teout = np.concatenate((teout, te), axis=1)
                    yeout = np.concatenate((yeout, ye), axis=1)
                    ieout = np.concatenate((ieout, ie), axis=1)
                if stop:  # Stop on a terminal event.
                    # Adjust the interpolation data to [t te(end)].
                    tnew = te[-1]
                    ynew = ye[:, -1]
                    done = True

        if output_sol:
            nout = nout + 1
            if nout > len(tout):
                tout = np.concatenate((tout, np.zeros(chunk)), axis=1)
                yout = np.concatenate((yout, np.zeros((neq, chunk))), axis=1)
                kvec = np.concatenate((kvec, np.zeros(chunk)), axis=1)
            tout[nout] = tnew
            yout[:, nout] = ynew
            kvec[nout] = k

        if output_ty or haveOutputFcn:
            if outputAt == 'SolverSteps':  # computed points, no refinement
                nout_new = 1
                tout_new = tnew
                yout_new = ynew
            elif outputAt == 'RefinedSteps':  # computed points, with refinement
                tref = t + (tnew - t) * S
                nout_new = refine
                tout_new = np.concatenate((tref, tnew), axis=1)
                yout_new = np.concatenate((ntrp15i(tref, [], [], tnew, ynew, mesh[0:k], meshsol[:, 0:k]), ynew),
                                           axis=1)
            elif outputAt == 'RequestedPoints':  # output only at tspan points
                nout_new = 0
                tout_new = []
                yout_new = []
                while next <= ntspan:
                    if tdir * (tnew - tspan[next]) < 0:
                        if haveEventFcn and stop:  # output tstop,ystop
                            nout_new = nout_new + 1
                            tout_new = np.concatenate((tout_new, tnew), axis=1)
                            yout_new = np.concatenate((yout_new, ynew), axis=1)
                        break
                    nout_new = nout_new + 1
                    tout_new = np.concatenate((tout_new, tspan[next]), axis=1)
                    if tspan[next] == tnew:
                        yout_new = np.concatenate((yout_new, ynew), axis=1)
                    else:
                        yout_new = np.concatenate(
                            (yout_new, ntrp15i(tspan[next], [], [], tnew, ynew, mesh[0:k], meshsol[:, 0:k])), axis=1)
                    next = next + 1

            if nout_new > 0:
                if output_ty:
                    oldnout = nout
                    nout = nout + nout_new
                    if nout > len(tout):
                        tout = np.concatenate((tout, np.zeros(chunk)), axis=1)
                        yout = np.concatenate((yout, np.zeros((neq, chunk))), axis=1)
                    idx = np.arange(oldnout + 1, nout)
                    tout[idx] = tout_new
                    yout[:, idx] = yout_new
                if haveOutputFcn:
                    stop = outputFcn(tout_new, yout_new[outputs, :], '', *outputArgs)
                    if stop:
                        done = True

        if done:
            break

        # Advance the integration one step.
        t = tnew
        y = ynew
        yp = ypnew
        mesh = np.concatenate((t, mesh[0:-1]), axis=1)
        meshsol = np.concatenate((y, meshsol[:, 0:-1]), axis=1)
        PDscurrent = Jconstant

        klast = k
        abshlast = absh
        nconh = min(nconh + 1, maxk + 2)

        # Estimate the scaled derivative of order k+2 if
        #  *  at constant step size,
        #  *  have not already decided to reduce the order,
        #  *  not already at the maximum order, and
        #  *  did not raise order on last step.
        if (nconh >= k + 2) and not (kopt < k) and not (k == maxk) and not raised_order:
            sderkp2 = np.linalg.norm(
                (np.dot(np.concatenate((meshsol[:, 0:k + 3], derM[0:k + 3, k + 2]), axis=1), invwt)), np.inf)
            if (k > 1) and (sderk <= min(sderkp1, sderkp2)):
                kopt = k - 1
                erropt = sderk / k
            elif ((k == 1) and (sderkp2 < 0.5 * sderkp1)) or \
                    ((k > 1) and (sderkp2 < sderkp1)):
                kopt = k + 1
                erropt = sderkp2 / (k + 2)

        temp = (erropt / (0.5 * rtol)) ** (1 / (kopt + 1))  # hopt = absh/temp
        if temp <= 1 / 2:
            absh = absh * 2
        elif temp > 1:
            absh = absh * max(0.5, min(0.9, 1 / temp))
        raised_order = kopt > k
        k = kopt

    solver_output = odefinalize(solver_name, sol,
                                outputFcn, outputArgs,
                                printstats, [nsteps, nfailed, nfevals,
                                             npds, ndecomps, nsolves],
                                nout, tout, yout,
                                haveEventFcn, teout, yeout, ieout,
                                [kvec, yp])
    if len(varargout) > 0:
        return solver_output
"""

###############################################################################
# improved version
###############################################################################
def ode15i(ode, tspan, y0, yp0, options, *varargin):
    solver_name = 'ode15i'

    from scipy.sparse import issparse
    from scipy.sparse.linalg import splu
    import numpy as np

    # Check inputs
    if len(options) == 0:
        options = []
        if len(options) < 4:
            raise Exception('MATLAB:ode15i:NotEnoughInputs')

    # Stats
    nsteps = 0
    nfailed = 0
    nfevals = 0
    npds = 0
    ndecomps = 0
    nsolves = 0

    # Output
    FcnHandlesUsed = True  # No MATLAB v. 5 legacy.
    output_sol = (FcnHandlesUsed and (len(varargout) == 1))  # sol = odeXX(...)
    output_ty = (not output_sol and (len(varargout) > 0))  # [t,y,...] = odeXX(...)
    # There might be no output requested...

    sol = []
    kvec = []
    if output_sol:
        sol['solver'] = solver_name
        sol['extdata']['odefun'] = ode
        sol['extdata']['options'] = options
        sol['extdata']['varargin'] = varargin
    odeArgs = varargin

    # Handle solver arguments -- pass yp0 as first extra parameter.
    neq, tspan, ntspan, next, t0, tfinal, tdir, y0, f0, options, threshold, rtol, hmax, htry, htspan = \
        odearguments(FcnHandlesUsed, False, solver_name, ode, tspan, y0, options, [yp0, *varargin])
    nfevals = nfevals + 1

    nonNegative = (len(odeget(options, 'NonNegative', [])) != 0)
    if nonNegative:
        print('MATLAB:ode15i:NonNegativeIgnored')

    if len(odeArgs) != 0:
        ode = lambda t, y, yp: ode(t, y, yp, *odeArgs)

    # THE MAIN LOOP

    done = False
    while not done:

        hmin = 16 * np.finfo(float).eps * abs(t)
        absh = min(hmax, max(hmin, absh))

        # Stretch the step if within 10% of tfinal-t.
        if 1.1 * absh >= abs(tfinal - t):
            h = tfinal - t
            absh = abs(h)
            done = True

        # LOOP FOR ADVANCING ONE STEP.
        nfails = 0
        while True:  # Evaluate the formula.

            gotynew = False  # is ynew evaluated yet?
            invwt = 1 / max(abs(y), threshold)
            while not gotynew:

                h = tdir * absh
                tnew = t + h
                if done:
                    tnew = tfinal  # Hit end point exactly.
                h = tnew - t  # Purify h.

                if (absh != abshlast) or (k != klast):
                    if absh != abshlast:
                        nconh = 0
                    Miter = dfdy + (lcf[k] / h) * dfdyp
                    if issparse(Miter):
                        LU = splu(Miter.tocsc())
                    else:
                        LU = splu(Miter)
                    ndecomps = ndecomps + 1
                    havrate = False
                    rate = 1  # Dummy value for test.

                # Predict the solution and its derivative at tnew.
                c = weights(mesh[0:k + 1], tnew, 1)
                ynew = np.dot(meshsol[:, 0:k + 1], c[:, 0])
                ypnew = np.dot(meshsol[:, 0:k + 1], c[:, 1])
                ypred = ynew
                minnrm = 100 * np.finfo(float).eps * np.linalg.norm(ypred * invwt, np.inf)

                # Compute local truncation error constant.
                erconst = -1 / (k + 1)
                for j in range(1, k + 1):
                    erconst = erconst - cf[j, k] * np.prod(
                        ((t - (j - 1) * h) - mesh[0:k + 1]) / (h * np.arange(1, k + 2)))
                erconst = abs(erconst)

                # Iterate with simplified Newton method.
                tooslow = False
                for iter in range(1, maxit + 1):
                    rhs = -ode(tnew, ynew, ypnew)

                    # use overloaded subfunction mldivide which throws no warning.
                    if issparse(Miter):
                        del_ = LU.solve(rhs)
                    else:
                        del_ = LU.solve(rhs)

                    newnrm = np.linalg.norm(del_ * invwt, np.inf)
                    ynew = ynew + del_
                    ypnew = ypnew + (lcf[k] / h) * del_

                    if iter == 1:
                        if newnrm <= minnrm:
                            gotynew = True
                            break
                        savnrm = newnrm
                    else:
                        rate = (newnrm / savnrm) ** (1 / (iter - 1))
                        havrate = True
                        if rate > 0.9:
                            tooslow = True
                            break
                    if havrate and ((newnrm * rate / (1 - rate)) <= 0.33 * rtol):
                        gotynew = True
                        break
                    elif iter == maxit:
                        tooslow = True
                        break

                nfevals = nfevals + iter
                nsolves = nsolves + iter

                if tooslow:
                    nfailed = nfailed + 1
                    abshlast = absh
                    klast = k
                    # Speed up the iteration by forming new linearization or reducing h.
                    if not PDscurrent:  # always current if Jconstant
                        if (len(dfdy_options) == 0) and (len(dfdyp_options) == 0):
                            f = []  # No numerical approximations formed
                        else:
                            f = ode(t, y, yp)
                            nfevals = nfevals + 1
                        dfdy, dfdyp, dfdy_options, dfdyp_options, NF = \
                            ode15ipdupdate(Jac, ode, t, y, yp, f, dfdy, dfdyp, dfdy_options, dfdyp_options, [])
                        npds = npds + 1
                        nfevals = nfevals + NF
                        PDscurrent = True

                        # Set a dummy value of klast to force formation of iteration matrix.
                        klast = 0

                    elif absh <= hmin:
                        print('MATLAB:ode15i:IntegrationTolNotMet', t, hmin)
                        solver_output = odefinalize(solver_name, sol,
                                                    outputFcn, outputArgs,
                                                    printstats, [nsteps, nfailed, nfevals,
                                                                npds, ndecomps, nsolves],
                                                    nout, tout, yout,
                                                    haveEventFcn, teout, yeout, ieout,
                                                    [kvec, yp])
                        if len(varargout) > 0:
                            return solver_output

                    else:
                        absh = 0.25 * absh
                        done = False

            # Using the tentative solution, approximate scaled derivative
            # used to estimate the error of the step.
            sderkp1 = np.linalg.norm((ynew - ypred) * invwt, np.inf) * \
                      abs(np.prod((absh * np.arange(1, k + 2)) / (tnew - mesh[0:k + 1])))

            erropt = sderkp1 / (k + 1)  # Error assuming constant step size.
            err = sderkp1 * erconst  # Error accounting for irregular mesh.

            # Approximate directly derivatives needed to consider lowering the
            # order.  Multiply by a power of h to get scaled derivative.
            kopt = k
            if k > 1:
                if nconh >= k:
                    sderk = np.linalg.norm(
                        (np.dot(np.concatenate((ynew, meshsol[:, 0:k]), axis=1), derM[0:k + 1, k])) * invwt, np.inf)
                else:
                    c = weights(np.concatenate((tnew, mesh[0:k]), axis=1), tnew, k)
                    sderk = np.linalg.norm(
                        (np.dot(np.concatenate((ynew, meshsol[:, 0:k]), axis=1), c[:, k])) * invwt, np.inf) * absh ** k
                if k == 2:
                    if sderk <= 0.5 * sderkp1:
                        kopt = k - 1
                        erropt = sderk / k
                else:
                    if nconh >= k - 1:
                        sderkm1 = np.linalg.norm(
                            (np.dot(np.concatenate((ynew, meshsol[:, 0:k - 1]), axis=1), derM[0:k, k - 1])) * invwt,
                            np.inf)
                    else:
                        c = weights(np.concatenate((tnew, mesh[0:k - 1]), axis=1), tnew, k - 1)
                        sderkm1 = np.linalg.norm(
                            (np.dot(np.concatenate((ynew, meshsol[:, 0:k - 1]), axis=1), c[:, k])) * invwt,
                            np.inf) * absh ** (k - 1)
                    if max(sderkm1, sderk) <= sderkp1:
                        kopt = k - 1
                        erropt = sderk / k

            if err > rtol:  # Failed step
                nfailed = nfailed + 1
                if absh <= hmin:
                    print('MATLAB:ode15i:IntegrationTolNotMet', t, hmin)
                    solver_output = odefinalize(solver_name, sol,
                                                outputFcn, outputArgs,
                                                printstats, [nsteps, nfailed, nfevals,
                                                            npds, ndecomps, nsolves],
                                                nout, tout, yout,
                                                haveEventFcn, teout, yeout, ieout,
                                                [kvec, yp])
                    if len(varargout) > 0:
                        return solver_output

                abshlast = absh
                klast = k
                nfails = nfails + 1
                if nfails == 1:
                    absh = absh * min(0.9, max(0.25, 0.9 * (0.5 * rtol / erropt) ** (1 / (kopt + 1))))
                elif nfails == 2:
                    absh = absh * 0.25
                else:
                    kopt = 1
                    absh = absh * 0.25
                absh = max(absh, hmin)
                if absh < abshlast:
                    done = False
                k = kopt

            else:  # Successful step
                break

        nsteps = nsteps + 1

        if haveEventFcn:
            events_args = [eventFcn, tnew, ynew, ypnew, mesh[0:k], meshsol[:, 0:k], *eventArgs]
            te, ye, ie, valt, stop = odezero(ntrp15i, events_aux, events_args, valt,
                                             t, y, tnew, ynew, t0, mesh[0:k], meshsol[:, 0:k])
            if len(te) != 0:
                if output_sol or (len(varargout) > 2):
                    teout = np.concatenate((teout, te), axis=1)
                    yeout = np.concatenate((yeout, ye), axis=1)
                    ieout = np.concatenate((ieout, ie), axis=1)
                if stop:  # Stop on a terminal event.
                    # Adjust the interpolation data to [t te(end)].
                    tnew = te[-1]
                    ynew = ye[:, -1]
                    done = True

        if output_sol:
            nout = nout + 1
            if nout > len(tout):
                tout = np.concatenate((tout, np.zeros(chunk)), axis=1)
                yout = np.concatenate((yout, np.zeros((neq, chunk))), axis=1)
                kvec = np.concatenate((kvec, np.zeros(chunk)), axis=1)
            tout[nout] = tnew
            yout[:, nout] = ynew
            kvec[nout] = k

        if output_ty or haveOutputFcn:
            if outputAt == 'SolverSteps':  # computed points, no refinement
                nout_new = 1
                tout_new = tnew
                yout_new = ynew
            elif outputAt == 'RefinedSteps':  # computed points, with refinement
                tref = t + (tnew - t) * S
                nout_new = refine
                tout_new = np.concatenate((tref, tnew), axis=1)
                yout_new = np.concatenate((ntrp15i(tref, [], [], tnew, ynew, mesh[0:k], meshsol[:, 0:k]), ynew),
                                           axis=1)
            elif outputAt == 'RequestedPoints':  # output only at tspan points
                nout_new = 0
                tout_new = []
                yout_new = []
                while next <= ntspan:
                    if tdir * (tnew - tspan[next]) < 0:
                        if haveEventFcn and stop:  # output tstop,ystop
                            nout_new = nout_new + 1
                            tout_new = np.concatenate((tout_new, tnew), axis=1)
                            yout_new = np.concatenate((yout_new, ynew), axis=1)
                        break
                    nout_new = nout_new + 1
                    tout_new = np.concatenate((tout_new, tspan[next]), axis=1)
                    if tspan[next] == tnew:
                        yout_new = np.concatenate((yout_new, ynew), axis=1)
                    else:
                        yout_new = np.concatenate(
                            (yout_new, ntrp15i(tspan[next], [], [], tnew, ynew, mesh[0:k], meshsol[:, 0:k])), axis=1)
                    next = next + 1

            if nout_new > 0:
                if output_ty:
                    oldnout = nout
                    nout = nout + nout_new
                    if nout > len(tout):
                        tout = np.concatenate((tout, np.zeros(chunk)), axis=1)
                        yout = np.concatenate((yout, np.zeros((neq, chunk))), axis=1)
                    idx = np.arange(oldnout + 1, nout)
                    tout[idx] = tout_new
                    yout[:, idx] = yout_new
                if haveOutputFcn:
                    stop = outputFcn(tout_new, yout_new[outputs, :], '', *outputArgs)
                    if stop:
                        done = True

        if done:
            break

        # Advance the integration one step.
        t = tnew
        y = ynew
        yp = ypnew
        mesh = np.concatenate((t, mesh[0:-1]), axis=1)
        meshsol = np.concatenate((y, meshsol[:, 0:-1]), axis=1)
        PDscurrent = Jconstant

        klast = k
        abshlast = absh
        nconh = min(nconh + 1, maxk + 2)

        # Estimate the scaled derivative of order k+2 if
        #  *  at constant step size,
        #  *  have not already decided to reduce the order,
        #  *  not already at the maximum order, and
        #  *  did not raise order on last step.
        if (nconh >= k + 2) and not (kopt < k) and not (k == maxk) and not raised_order:
            sderkp2 = np.linalg.norm(
                (np.dot(np.concatenate((meshsol[:, 0:k + 3], derM[0:k + 3, k + 2]), axis=1), invwt)), np.inf)
            if (k > 1) and (sderk <= min(sderkp1, sderkp2)):
                kopt = k - 1
                erropt = sderk / k
            elif ((k == 1) and (sderkp2 < 0.5 * sderkp1)) or \
                    ((k > 1) and (sderkp2 < sderkp1)):
                kopt = k + 1
                erropt = sderkp2 / (k + 2)

        temp = (erropt / (0.5 * rtol)) ** (1 / (kopt + 1))  # hopt = absh/temp
        if temp <= 1 / 2:
            absh = absh * 2
        elif temp > 1:
            absh = absh * max(0.5, min(0.9, 1 / temp))
        raised_order = kopt > k
        k = kopt

    solver_output = odefinalize(solver_name, sol,
                                outputFcn, outputArgs,
                                printstats, [nsteps, nfailed, nfevals,
                                             npds, ndecomps, nsolves],
                                nout, tout, yout,
                                haveEventFcn, teout, yeout, ieout,
                                [kvec, yp])
    if len(varargout) > 0:
        return solver_output
    
###############################################################################
###############################################################################
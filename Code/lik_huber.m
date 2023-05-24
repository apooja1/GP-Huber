function lik = lik_huber(varargin)
%LIK_GAUSSIAN  Create a Huber likelihood structure
%


  ip=inputParser;
  ip.FunctionName = 'LIK_HUBER';
  ip.addOptional('lik', [], @(x) isstruct(x) || isempty(x));
  ip.addParamValue('sigma2',0.1, @(x) isscalar(x) && x>=0);
  ip.addParamValue('sigma2_prior',prior_logunif(), @(x) isstruct(x) || isempty(x));
%   ip.addParamValue('n',[], @(x) isreal(x) && all(x>0));
  ip.addParamValue('weights',[], @(x) isreal(x) && all(x>0));
%   ip.addParamValue('b',[], @(x) isreal(x) && all(x>0));
%   ip.addParamValue('epsilon',[], @(x) isreal(x) && all(x>0));
  ip.parse(varargin{:});
  lik=ip.Results.lik;

  if isempty(lik)
    init=true;
    lik.type = 'Huber';
  else
    if ~isfield(lik,'type') || ~isequal(lik.type,'Huber')
      error('First argument does not seem to be a valid likelihood function structure')
    end
    init=false;
  end
  
  % Initialize parameters
  if init || ~ismember('sigma2',ip.UsingDefaults)
    lik.sigma2 = ip.Results.sigma2;
  end
%   if init || ~ismember('n',ip.UsingDefaults)
%     lik.n = ip.Results.n;
%   end
  if init ||  ~ismember('weights',ip.UsingDefaults)
     lik.weights = ip.Results.weights; 
  end
% 
%   if init ||  ~ismember('b',ip.UsingDefaults)
%      lik.b = ip.Results.b; 
%   end
% 
%   if init ||  ~ismember('epsilon',ip.UsingDefaults)
%      lik.epsilon = ip.Results.epsilon; 
%   end
  % Initialize prior structure
  if init
    lik.p=[];
  end
  if init || ~ismember('sigma2_prior',ip.UsingDefaults)
    lik.p.sigma2=ip.Results.sigma2_prior;
  end
  if init
    % Set the function handles to the subfunctions
    lik.fh.pak = @lik_huber_pak;
    lik.fh.unpak = @lik_huber_unpak;
    lik.fh.ll = @lik_huber_ll;
    lik.fh.llg = @lik_huber_llg;    
    lik.fh.llg2 = @lik_huber_llg2;
    lik.fh.llg3 = @lik_huber_llg3;
    lik.fh.lp = @lik_huber_lp;
    lik.fh.lpg = @lik_huber_lpg;
%     lik.fh.cfg = @lik_huber_cfg;
    lik.fh.tiltedMoments = @lik_huber_tiltedMoments;
%     lik.fh.trcov  = @lik_huber_trcov;
%     lik.fh.trvar  = @lik_huber_trvar;
    lik.fh.predy = @lik_huber_predy;    
    lik.fh.siteDeriv = @lik_huber_siteDeriv;
     lik.fh.invlink = @lik_huber_invlink;
%       lik.fh.optimizef=@lik_huber_optimizef;
    lik.fh.recappend = @lik_huber_recappend;
  end

end

function [w, s, h] = lik_huber_pak(lik)
%LIK_GAUSSIAN_PAK  Combine likelihood parameters into one vector.
%
%  Description
%    W = LIK_GAUSSIAN_PAK(LIK) takes a likelihood structure LIK
%    and combines the parameters into a single row vector W.
%    This is a mandatory subfunction used for example in energy 
%    and gradient computations.
%
%       w = [ log(lik.sigma2)
%             (hyperparameters of lik.magnSigma2)]'
%     
%  See also
%    LIK_GAUSSIAN_UNPAK

  w = []; s = {}; h=[];
  if ~isempty(lik.p.sigma2)
    w = [w log(lik.sigma2)];
    s = [s; 'log(gaussian.sigma2)'];
    h = [h 0];
    % Hyperparameters of sigma2
    [wh, sh, hh] = lik.p.sigma2.fh.pak(lik.p.sigma2);
    w = [w wh];
    s = [s; sh];
    h = [h hh];
  end    

end

function [lik, w] = lik_huber_unpak(lik, w)

  
  if ~isempty(lik.p.sigma2)
    lik.sigma2 = exp(w(1));
    w = w(2:end);
    % Hyperparameters of sigma2
    [p, w] = lik.p.sigma2.fh.unpak(lik.p.sigma2, w);
    lik.p.sigma2 = p;
  end
end


function logLik = lik_huber_ll(lik, y, f, ~) 


  epsilon=0.45;
  b=0.5;
  s2 = lik.sigma2;
  sigma=sqrt(s2);
  r = (f-y);
 weights=lik.weights;
 rs=r./(weights.*sigma);
rho=@(rs) b.^2*(sqrt(1+(rs./b).^(2))-1);
logLik1= -rho(rs) -log(sigma) - 0.5.*log(2.*pi)+log(1-epsilon);
logLik=sum(logLik1);
end

function llg = lik_huber_llg(lik, y, f, param, ~)

 
switch param
    case 'param'
          s2 = lik.sigma2;
          sigma=sqrt(s2);
          epsilon=0.45;
          b=0.5;
          r = (y-f);
%           c1=1+5/(length(f(:,1))-length(f(1,:)));
%           s1 = 1.4826*(c1)*median(abs(r));
%           rs=r./(weights.*sigma);
%             weights=ones(length(r),1); 
            weights=lik.weights;
%           llg=sum((-1./sigma)+b.^(2).*r.^(2)./((weights.^(2).*sigma.^(3)).*(sqrt((r.^2./weights.^(2).*s2))+1)));
% llg=sum((b.^3.*f.*(y - (b.*f.*sigma)./weights))./(weights.*((y - (b.*f.*sigma)./weights).^2 + 1).^(1/2))) - 1/sigma;

 llg=sum((f - y).^2./(sigma^3.*weights.^2.*((f - y).^2./(b^2*sigma^2.*weights.^2) + 1).^(1/2))) - 1/sigma;


    case 'latent'
         epsilon=0.45;
         b=0.5;
        s2 = lik.sigma2;
        sigma=sqrt(s2);
         r = (y-f);
         c1=1+5/(length(f(:,1))-length(f(1,:)));
         s1 = 1.4826*(c1)*median(abs(r));

weights=lik.weights;
 llg=-(f - y)./(sigma^2.*weights.^2.*((f - y).^2./(b^2*sigma^2.*weights.^2) + 1).^(1/2));
 
end
end


function llg2 = lik_huber_llg2(lik, y, f, param, ~)

   
switch param
    case 'latent'
      s2 = lik.sigma2;
      sigma=sqrt(s2);
      epsilon=0.45;
      b=0.5;
      r = (y-f);
%       c1=1+5/(length(f(:,1))-length(f(1,:)));
%       s1 = 1.4826*(c1)*median(abs(r));
%       rs=r./(weights.*sigma);
%       weights=ones(length(r),1);
       weights=lik.weights;
%       llg2=-b.^2.*s2.*weights.^(2)./(s2.*weights.^(2)+f-2.*f.*y+y.^(2)).^(3/2);
     llg2=-1./(sigma.^2.*weights.^2.*((f - y).^2./(b^2*sigma^2.*weights.^2) + 1).^(3/2));

    case 'latent+param'
        % there is also correction due to the log transformation
          s2 = lik.sigma2;
          sigma=sqrt(s2);
          epsilon=0.45;
          b=0.5;
          r = (y-f);

 weights=lik.weights;
llg2=((f - y).*(2*b^2*sigma^2.*weights.^2 + f.^2 - 2.*f.*y + y.^2))./(b^2*sigma^5.*weights.^4.*((f - y).^2./(b^2.*sigma^2.*weights.^2) + 1).^(3/2));

end
end    

function llg3 = lik_huber_llg3(lik, y, f, param, z)


switch param
    case 'latent'
        s2 = lik.sigma2;
        sigma=sqrt(s2); 
        epsilon=0.45;
        b=0.5;
        weights=lik.weights;
 llg3=(3.*f - 3.*y)./(b.^2.*sigma.^4.*weights.^4.*((b.^2.*sigma.^2.*weights.^2 + f.^2 - 2*f.*y + y.^2)./(b^2*sigma^2.*weights.^2)).^(5/2));

    case 'latent2+param'
        s2 = lik.sigma2;
        sigma=sqrt(s2);
         epsilon=0.45;
         b=0.5;
         r=y-f;
         weights=lik.weights;
         llg3=(2*b^2*sigma^2.*weights.^2 - f.^2 + 2.*f.*y - y.^2)./(b.^2.*sigma^5.*weights.^4.*((f - y).^2./(b^2.*sigma^2.*weights.^2) + 1).^(5/2));

end
end


function lp = lik_huber_lp(lik)


  lp = 0;

  if ~isempty(lik.p.sigma2)
    likp=lik.p;
    lp = likp.sigma2.fh.lp(lik.sigma2, likp.sigma2) + log(lik.sigma2);
  end
end

function lpg = lik_huber_lpg(lik)


  lpg = [];

  if ~isempty(lik.p.sigma2)
    likp=lik.p;
    
    lpgs = likp.sigma2.fh.lpg(lik.sigma2, likp.sigma2);
    lpg = lpgs(1).*lik.sigma2 + 1;
    if length(lpgs) > 1
      lpg = [lpg lpgs(2:end)];
    end            
  end
end






function [lpy, Ey, Vary] = lik_huber_predy(lik, Ef, Varf, yt, zt)   %requires attention

  sigma2 = lik.sigma2;
  sigma = sqrt(sigma2);

     epsilon=0.45;
     b=0.5;
  EVary = zeros(size(Ef));
  VarEy = zeros(size(Ef)); 
   Ey = Ef;
   Vary = EVary + VarEy;
  lpy = zeros(length(yt),1);

  if (size(Ef,2) > 1) && size(yt,2) == 1
   
    for i2=1:length(yt)
      py = h_pdf(yt(i2), Ef(i2,:), sigma);
      pf = Varf(i2,:)./sum(Varf(i2,:));
      lpy(i2) = log(sum(py.*pf));
    end
  else
    for i2 = 1:length(yt)
      [pdf, minf,maxf]=init_huber_norm(yt(i2),Ef(i2),Varf(i2),sigma);
      lpy(i2) = log(quadgk(pdf, minf, maxf));
    end
  end
  
end



function [logM_0, m_1, sigm2hati1] = lik_huber_tiltedMoments(lik, y, i1, sigma2_i, myy_i, z)

  
  yy = y(i1);
  sigma2 = lik.sigma2;
  sigma=sqrt(sigma2);

  logM_0=zeros(size(yy));
  m_1=zeros(size(yy));
  sigm2hati1=zeros(size(yy));
  
  for i=1:length(i1)
    if isscalar(sigma2_i)
      sigma2ii = sigma2_i;
    else
      sigma2ii = sigma2_i(i);
    end
  
    [tf,minf,maxf]=init_huber_norm(yy(i),myy_i(i),sigma2ii,sigma);
    
    % Integrate with quadrature
    RTOL = 1.e-6;
    ATOL = 1.e-10;
    [m_0, m_1(i), m_2] = quad_moments(tf, minf, maxf, RTOL, ATOL);
    if isnan(m_0)
      logM_0=NaN;
      return
    end
    sigm2hati1(i) = m_2 - m_1(i).^2;
    
 
    if sigm2hati1(i) >= sigma2ii
      ATOL = ATOL.^2;
      RTOL = RTOL.^2;
      [m_0, m_1(i), m_2] = quad_moments(tf, minf, maxf, RTOL, ATOL);
      sigm2hati1(i) = m_2 - m_1(i).^2;
      if sigm2hati1(i) >= sigma2ii
        warning('lik_huber_tilted_moments: sigm2hati1 >= sigm2_i');
      end
    end
    logM_0(i) = log(m_0);
  end
end












function [g_i] = lik_huber_siteDeriv(lik, y, i1, sigm2_i, myy_i, z)   %not sure 


sigma=sqrt(lik.sigma2);
b=0.5;
yy = y(i1);

  [tf,minf,maxf]=init_huber_norm(yy,myy_i,sigm2_i,sigma);
  td = @deriv;
  
  [m_0, fhncnt] = quadgk(tf, minf, maxf);
  [g_i, fhncnt] = quadgk(@(f) td(f).*tf(f)./m_0, minf, maxf);
  

  function g = deriv(f)
   epsilon=0.45;
   weights=ones(length(y),1);
   weights=lik.weights;

    g = -sqrt(2*pi)./(sigma*(1-epsilon)) +(1./(2* sqrt(((yy-f).^2./(weights.^2.*sigma^2))+1))).*((yy-f)./sigma).^2;
    
  end
end


function [df,minf,maxf] = init_huber_norm(yy,myy_i,sigm2_i,sigma)


epsilon=0.45;
b=0.5;
  ldconst = -log(2*sigma)+log(1-epsilon) ...
            - log(sigm2_i)/2 - log(2*pi);
  df = @huber_norm;
  ld = @log_huber_norm;
  ldg = @log_huber_norm_g;

  if yy==0
    % with yy==0, the mode of the likelihood is not defined
    % use the mode of the Gaussian (cavity or posterior) as a first guess
    modef = myy_i;
  else
    % use precision weighted mean of the Gaussian approximation
    % of the Quantile-GP likelihood and Gaussian
    modef = (myy_i/sigm2_i + yy/sigma)/(1/sigm2_i + 1/sigma);
  end
  % find the mode of the integrand using Newton iterations
  % few iterations is enough, since the first guess in the right direction
  niter=8;       % number of Newton iterations 
  
  minf=modef-6*sigm2_i;
  while ldg(minf) < 0
    minf=minf-2*sigm2_i;
  end
  maxf=modef+6*sigm2_i;
  while ldg(maxf) > 0
    maxf=maxf+2*sigm2_i;
  end
  for ni=1:niter
%     h=ldg2(modef);
    modef=0.5*(minf+maxf);
    if ldg(modef) < 0
      maxf=modef;
    else
      minf=modef;
    end
  end
  % integrand limits based on Gaussian approximation at mode
  minf=modef-6*sqrt(sigm2_i);
  maxf=modef+6*sqrt(sigm2_i);
  modeld=ld(modef);
  iter=0;
  % check that density at end points is low enough
  lddiff=20; % min difference in log-density between mode and end-points
  minld=ld(minf);
  step=1;
  while minld>(modeld-lddiff)
    minf=minf-step*sqrt(sigm2_i);
    minld=ld(minf);
    iter=iter+1;
    step=step*2;
    if iter>100
      error(['lik_huber -> init_huber_norm: ' ...
             'integration interval minimun not found ' ...
             'even after looking hard!'])
    end
  end
  maxld=ld(maxf);
  iter=0;
  step=1;
  while maxld>(modeld-lddiff)
    maxf=maxf+step*sqrt(sigm2_i);
    maxld=ld(maxf);
    iter=iter+1;
    step=step*2;
    if iter>100
      error(['lik_huber -> init_huber_norm: ' ...
             'integration interval maximun not found ' ...
             'even after looking hard!'])
    end
  end
  
  function integrand = huber_norm(f)


      b=0.5;
      weights=ones(length(yy),1);
      epsilon=0.45;
    integrand = exp(ldconst ...
                   - b^2.*(sqrt(1+((yy-f)./(weights.*b.*sigma)).^2)-1) ...
                    -0.5*(f-myy_i).^2./sigm2_i);
  end
  
  function log_int = log_huber_norm(f)

      b=0.5;
      weights=ones(length(yy),1);
      epsilon=0.45;

 
    log_int = ldconst...
             -b^2.*(sqrt(1+((yy-f)./(weights.*sigma.*b)).^2)-1) ...
                    -0.5*(f-myy_i).^2./sigm2_i;
  end
  
  function g = log_huber_norm_g(f)

      b=0.5;
      weights=ones(length(yy),1);


     g = -(f - yy)./(sigma^2.*weights.^2.*((f - yy).^2./(b^2*sigma^2.*weights.^2) + 1).^(1/2)) + (myy_i - f)./sigm2_i;


  end
  
  
end


function mu = lik_huber_invlink(lik, f, z)
%LIK_LAPLACE_INVLINK  Returns values of inverse link function
%             
%  Description 
%    MU = LIK_LAPLACE_INVLINK(LIK, F) takes a likelihood structure LIK and
%    latent values F and returns the values MU of inverse link function.
%    This subfunction is needed when using function gp_predprctmu.
%
%     See also
%     LIK_LAPLACE_LL, LIK_LAPLACE_PREDY
  
  mu = f;
end



function reclik = lik_huber_recappend(reclik, ri, lik)


  if nargin == 2
    % Initialize the record
    reclik.type = 'Huber';
    
    % Initialize the parameters
    reclik.sigma2 = []; 
    reclik.n = []; 
    
    % Set the function handles
    reclik.fh.pak = @lik_huber_pak;
    reclik.fh.unpak = @lik_huber_unpak;
    reclik.fh.lp = @lik_huber_lp;
    reclik.fh.lpg = @lik_huber_lpg;
    reclik.fh.ll= @lik_huber_ll;
    reclik.fh.llg = @lik_huber_llg;
    reclik.fh.llg2 = @lik_huber_llg2;
    reclik.fh.llg3 = @lik_huber_llg3;
    reclik.fh.tiltedMoments = @lik_huber_tiltedMoments;
    reclik.fh.siteDeriv = @lik_huber_siteDeriv;
    reclik.fh.lik_huber_invlink=@lik_huber_invlink;
    reclik.fh.predy = @lik_huber_predy;
    reclik.fh.recappend = @lik_huber_recappend;     
    reclik.p=[];
    reclik.p.sigma2=[];
    if ~isempty(ri.p.sigma2)
      reclik.p.sigma2 = ri.p.sigma2;
    end
  else
    % Append to the record
    likp = lik.p;

    % record sigma2
    reclik.sigma2(ri,:)=lik.sigma2;
    if isfield(likp,'sigma2') && ~isempty(likp.sigma2)
      reclik.p.sigma2 = likp.sigma2.fh.recappend(reclik.p.sigma2, ri, likp.sigma2);
    end
    % record n if given
    if isfield(lik,'n') && ~isempty(lik.n)
      reclik.n(ri,:)=lik.n(:)';
    end
  end
end

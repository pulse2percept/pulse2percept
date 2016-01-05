function y=Gamma(n,theta,t);
% GAMMA
%	y=Gamma(n,theta,t)
%	returns a gamma function from [0:t];
%	y=(t/theta).^(n-1).*exp(-t/theta)/(theta*factorial(n-1));
%	which is the result of an n stage leaky integrator.
%
%	6/27/95 gmb


flag=0;

if t(1)==0
	t=t(2:length(t));
	flag=1;
end
id=find(t<=0);
t(id)=ones(size(id));
y = (  (theta'*(1./t)).^(1-n).*exp(-(1./(theta'*(1./t)))))./(theta'*ones(size(t))*factorial(n-1));
y(id)=zeros(size(id));
if flag==1
	y=[0;y']';
end

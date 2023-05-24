%Function to calculate the projection statistics (PS)

function [P,PS] = projectionstatistics(H)
[m,n]=size(H);
M=median(H);                                            
u=zeros(m,n);
v=zeros(m,n);
z=zeros(m,1);
P=zeros(m,m);
for kk=1:m
    u(kk,:)=H(kk,:)-M;                                  
    v(kk,:)=u(kk,:)/norm(u(kk,:));                      
    for ii=1:m
        z(ii,:)=dot(H(ii,:)',v(kk,:));                  
    end
    zmed=median(z);                                     
    MAD=1.4826*(1+(15/(m)))*median(abs(z-zmed));         
    for ii=1:m                                          
        P(kk,ii)=abs(z(ii)-zmed)/MAD;
    end
end
PS=max(P);                                             
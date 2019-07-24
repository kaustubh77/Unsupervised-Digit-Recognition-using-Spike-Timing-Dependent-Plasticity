%Initialize parameters
dt=0.5;
d=8;
a=0.02;
c=-65;
b=0.2;
T=ceil(1000/dt);
%%
%Reserve memory
v=zeros(T,1);
u=zeros(T,1);
v(1)=-70;
u(1)=-14;
%%
%for loop over time
for t=1:T-1;
    if(t*dt>200 && t*dt<700)
        Iapp =7;
    else
        Iapp=0;
end

if v(t)<35
        %update ODE
    dv=(0.04*v(t)+5)*v(t)+140-u(t);
    v(t+1)=v(t)+(dv+Iapp)*dt;
    du=a*(b*v(t)-u(t));
    u(t+1)=u(t)+dt*du;
else
        %spike!
    v(t)=35;
    v(t+1)=c;
    u(t+1)=u(t)+d;
    end
end
%%
%plot voltage trace
plot((0:T-1)*dt,v,'b');
xlabel('Time [ms]');
ylabel('Membrane voltage [mV]');
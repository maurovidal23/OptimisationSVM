#EJEMPLO TUTORIAL
import Pkg
Pkg.add("JuMP")
using JuMP
Pkg.add("HiGHS")
using HiGHS

model=Model(HiGHS.Optimizer)

@variable(model,x>=0)
@variable(model,0<= y<=3)

@objective(model,Min,12x+20y)
@constraint(model,c1,6x+8y>=100)
@constraint(model,c2,7x+12y>=120)
print(model)
optimize!(model)

#EJMEPLO DE CLASE

#Leemos el archivo SVM_simulado_dim_4_obs_15_semilla_5.dat

nobs=15
dim=4
d=
#[] columnas
data=[30 30 6 -5
-19 11 44 -26
-42 -39 22 43
22 -5 47 -29
40 -34 49 -13
-20 -39 1 12
-3 -22 -10 -7
-31 29 -10 -42
-46 27 10 -48
39 36 10 -43
23 2 -32 12
1 6 26 24
18 -47 -33 24
-6 36 -3 30
24 -45 -43 36;]

y=[1, 1,  -1 , -1 , -1,  -1  ,-1 , 1  ,1,  1 , 1 , -1,  -1 , -1,  -1]

model_primal1=Model(HiGHS.Optimizer)

@variable(model_primal1,w[1:dim])
@variable(model_primal1,b)
@objective(model_primal1,Min,0.5*sum(w.^2))
@constraint(model_primal1,primal_c[i=1:nobs],y[i]*(b+sum(w.*data[i,:]))-1>=0)
print(model_primal1)
optimize!(model_primal1)
for i in 1:dim
print(value(w[i]), " ")
end

model_dual1=Model(HiGHS.Optimizer)
@variable(model_dual1,u[1:nobs]>=0)
@objective(model_dual1,Max,sum(u)-0.5*sum(kron(u,transpose(u)).*kron(y,transpose(y)).*(data*transpose(data))))
@constraint(model_dual1,sum(y.*u)==0)
print(model_dual1)
optimize!(model_dual1)

w=zeros(dim)
for d in 1:dim
    w_d=0
    for i in 1:nobs
       global  w_d=w_d+y[i]*value(u[i])*data[i,d]
        print(w_d)
    end
    w[d]=w_d 
    
end





import Pkg
Pkg.add("JuMP")
using JuMP
Pkg.add("HiGHS")
using HiGHS
Pkg.add("Plots")
using Plots
Pkg.add("AmplNLWriter")
using AmplNLWriter

using Base.Threads

Pkg.add("ThreadsX")
using ThreadsX
using Random

# Fijar la semilla
seed = 42
rng = Random.MersenneTwister(seed)




#función clasificación de los datos
function entrenamiento(data,vect_hiperplano,term_ind_hiperplano)
    nobs=length(data[:,1])
    dim=length(data[1,:])
    #comprobamos que tienen dimensiones compatible 
    if length(vect_hiperplano)!=dim
        print("Error en la dimensión de los datos")
    end

    #definimos la condición 
    #si w^T*x^i-r>0 =>true
    #caso contrario =>false
    condicion(x)= transpose(vect_hiperplano)*x>0

    #define el vector y con los 1,-1 de entrenamiento
    y=zeros(nobs)
    for i in 1:nobs
        y[i]=ifelse.(condicion(data[i,:]),1,-1)
    end

    return y
end


#Funcion que resuelve el modelo primal
function  resolver_primal(data,y)
    nobs=length(data[:,1])
    dim=length(data[1,:])
    #model_primal1=Model(HiGHS.Optimizer)
    println("resolviendo primal")
     model_primal1=Model(() -> AmplNLWriter.Optimizer("bonmin"))
    @variable(model_primal1,w[1:dim])
    @variable(model_primal1,b)
    @objective(model_primal1,Min,0.5*sum(w.^2))
    @constraint(model_primal1,primal_c[i=1:nobs],y[i]*(b+sum(w.*data[i,:]))-1>=0)
    #print(model_primal1)
    time_primal=@elapsed optimize!(model_primal1)
    return time_primal
    #println("timePrimal=",time_primal)
   
end

#Funcion que resuelve el modelo dual
function  resolver_dual(data,y)
    nobs=length(data[:,1])
    dim=length(data[1,:])
    println("resolviendo dual")
     #model_dual1=Model(HiGHS.Optimizer)
     model_dual1=Model(() -> AmplNLWriter.Optimizer("bonmin"))
     @variable(model_dual1,u[1:nobs]>=0)
     @objective(model_dual1,Max,sum(u)-0.5*sum(kron(u,transpose(u)).*kron(y,transpose(y)).*(data*transpose(data))))
     @constraint(model_dual1,sum(y.*u)==0)
     #print(model_dual1)
     time_dual=@elapsed optimize!(model_dual1)
     return time_dual
     #println("timeDual=",time_dual)
     
end


#definición matriz de salidas
data_suavizados=zeros(20*20,2)
#clasificacion entrenamiento
y_entre_suavizado=zeros(20*20)
i=0;
j=0;
k=0;
for dim in 2:2:40
    #para una dimensión fijada, seleccionamos un hiperplano que nos permita clasificar los datos
    #de entrenamiento que vamos a generar en el siguiente bucle
    vect_hiperplano=rand(rng,-50:50,dim) 
    term_ind_hiperplano=rand(rng,-50:50,1)
    i=i+1
    j=0
    for nobs in 1:20
        
        j=j+1
        k=k+1
        data=rand(rng,-30:30,nobs,dim)
        #Necesito función para clasificar y
        y=entrenamiento(data,vect_hiperplano,term_ind_hiperplano)
        
        
        # Crear hilos para los modelos primal y dual
        hilo_primal = Threads.@spawn resolver_primal(data, y)
        hilo_dual = Threads.@spawn resolver_dual(data, y)
        
       
        
        # Esperar a que uno de los hilos termine

        wait(hilo_primal)
        wait(hilo_dual)
        
        time_primal=hilo_primal.result
        time_dual=hilo_dual.result
        data_suavizados[k,:]=[dim,nobs]
        #decimos que el dual gana al primal si tarda menos del 90% del primal
        if time_dual<time_primal*0.9
            y_entre_suavizado[k]=1
        else
            y_entre_suavizado[k]=-1
        end
    end
end

# Encuentra las coordenadas de los puntos 0 y 1 en la matriz
coordenadas_0 = findall(x -> x == 0, salidas)
coordenadas_1 = findall(x -> x == 1, salidas)

# Convierte las coordenadas de CartesianIndex a tuplas de Int
coordenadas_0 = Tuple.(coordenadas_0)
coordenadas_1 = Tuple.(coordenadas_1)

# Crea un nuevo gráfico y agrega puntos azules y rojos
scatter([c[2] for c in coordenadas_0], [c[1] for c in coordenadas_0], color=:blue, legend=false, aspect_ratio=1)
scatter!([c[2] for c in coordenadas_1], [c[1] for c in coordenadas_1] , color=:red)


# Muestra el gráfico
display(plot)


#CLASIFICACIÓN SUAVE PROBLEMA primal


# Abre un archivo en modo escritura
filename = "datos.txt"
file = open(filename, "w")

# Escribe cada elemento del vector en una línea del archivo

for i in 1:nobs
    println(file," ",i," ",data_suavizados[i,1]," ",data_suavizados[i,2])
end


# Cierra el archivo
close(file)

#Abre un archivo en modo escritura
# Abre un archivo en modo escritura
filename = "clasificación_datos_entrenamiento.txt"
file = open(filename, "w")

# Escribe cada elemento del vector en una línea del archivo
for i in 1:nobs
    print(file,i," ",y_entre_suavizado[i], " ")
end

# Cierra el archivo
close(file)


#constantes
M=1
l=1
nobs=length(data_suavizados[:,1])
dim=length(data_suavizados[1,:])
#entrenamiento
data=data_suavizados
y=y_entre_suavizado

#modelo suavizado
model_primal_SVM_suavizado=Model(() -> AmplNLWriter.Optimizer("bonmin"))
@variable(model_primal_SVM_suavizado,w[1:dim])
@variable(model_primal_SVM_suavizado,b)
@variable(model_primal_SVM_suavizado,s[1:nobs])
@objective(model_primal_SVM_suavizado,Min,0.5*sum(w.^2)+M/l*sum(s))
@constraint(model_primal_SVM_suavizado,primal_c[i=1:nobs],y[i]*(b+sum(w.*data[i,:]))-1>=-s[i])
@constraint(model_primal_SVM_suavizado,constrain_s[i=1:nobs],s[i]>=0)
print(model_primal_SVM_suavizado)
optimize!(model_primal_SVM_suavizado)
termination_status(model_primal_SVM_suavizado)
#definimos el w de la resolucion
for i in 1:dim
    print(value(w[i]), " ")
end





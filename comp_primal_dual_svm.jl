import Pkg
Pkg.add("JuMP")
using JuMP
Pkg.add("HiGHS")
using HiGHS
Pkg.add("Plots")
using Plots
Pkg.add("AmplNLWriter")
using AmplNLWriter
import Random
# Fijar la semilla
seed = 42
rng = Random.MersenneTwister(seed)

#dimension de la matriz de salidas

dim_matriz=10

#definición matriz de salidas
salidas=zeros(dim_matriz,dim_ma)


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


#definición matriz de salidas
data_suavizados=zeros(dim_matriz^2,2)
#clasificacion entrenamiento
y_entre_suavizado=zeros(dim_matriz^2)
i=0;
j=0;
k=0;
for dim in 1:dim_matriz
    #para una dimensión fijada, seleccionamos un hiperplano que nos permita clasificar los datos
    #de entrenamiento que vamos a generar en el siguiente bucle
    vect_hiperplano=rand(rng,-50:50,dim) 
    term_ind_hiperplano=rand(rng,-50:50,1)
    i=i+1
    j=0
    for nobs in 1:dim_matriz
        j=j+1
        data=rand(rng,-30:30,nobs,dim)

        #Necesito función para clasificar y
        y=entrenamiento(data,vect_hiperplano,term_ind_hiperplano)
        
        
        #Modelo_primal
        
        #model_primal1=Model(HiGHS.Optimizer)
        model_primal1=Model(() -> AmplNLWriter.Optimizer("bonmin"))


        @variable(model_primal1,w[1:dim])
        @variable(model_primal1,b)
        @objective(model_primal1,Min,0.5*sum(w.^2))
        @constraint(model_primal1,primal_c[i=1:nobs],y[i]*(b+sum(w.*data[i,:]))-1>=0)
        #print(model_primal1)
        time_primal=@elapsed optimize!(model_primal1)
        
        #Modelo_dual
        
        #model_dual1=Model(HiGHS.Optimizer)
        model_dual1=Model(() -> AmplNLWriter.Optimizer("bonmin"))
        @variable(model_dual1,u[1:nobs]>=0)
        @objective(model_dual1,Max,sum(u)-0.5*sum(kron(u,transpose(u)).*kron(y,transpose(y)).*(data*transpose(data))))
        @constraint(model_dual1,sum(y.*u)==0)
        #print(model_dual1)
        time_dual=@elapsed optimize!(model_dual1)

        print("Time primal:", time_primal, " Time dual:", time_dual)

        
        #Consideramos que gana el dual si esta por debajo de un 90% del tiempo del primal
        if time_dual<time_primal*0.9
            salidas[i,j]=1
        else
            salidas[i,j]=0
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


#escritura en texto de los datos para aplicar en AMPL

#constantes
M=1
l=1
nobs=length(data_suavizados[:,1])
dim=length(data_suavizados[1,:])

# Abre un archivo en modo escritura
filename = "datos.txt"
file = open(filename, "w")

# Escribe cada elemento del vector en una línea del archivo
println(file,"param nobs:= ", nobs,";")
println(file, "param dim:= ", 4,";")
println(file,"param data: 1 2:=")
for i in 1:nobs-1
    println(file,i," ",data_suavizados[i,1]," ",data_suavizados[i,2])
end
println(file, nobs," ",data_suavizados[nobs,1], " ", data_suavizados[i,2])

println(file)
println(file,"y:= ",y_entre_suavizado)



# Cierra el archivo
close(file)

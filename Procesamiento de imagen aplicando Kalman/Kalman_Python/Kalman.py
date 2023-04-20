import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dt=0.001 #Periodo de muestreo del sensor (segundos)

pos_0=0 #Posición inicial del sensor (m)
vel_final=0 #Velocidad final del sensor (m/s)
acel_0=0 #Aceleracion inicial del sensor

Q=np.array([[2.7473e+05,4.0006e+04,9.0248e-12],[4.0006e+04,8.0010e+03,1.7881e-08],[9.0248e-12,1.7881e-08,0.0038]])

R=0.0038

F=np.array([[1,dt,((dt*dt)/2)],[0,1,dt],[0,0,1]])

F_trans=np.transpose(F)#FAlta transpuesta

#Modelo de observación
H=np.array([[0,0,1]])

H_trans=np.array([[0],[0],[1]])

#Vector de estados inicial
x_nn=np.array([[pos_0],[vel_final],[acel_0]])

#Matriz de covarianza a-posteriori
P=np.array([[0.8,0.0,0.0],[0.0,0.8,0.0],[0.0,0.0,0.8]])

eye_3=np.array([[1,0,0],[0,1,0],[0,0,1]])

#arduinoSerial = serial.Serial('COM11', 115200)

val = 'Datos_de_prueba_4.xlsx'
dato=pd.read_excel(val,sheet_name='Sheet1')
dato.describe()
#arduinoSerial = np.array(["50","20","-582.26","10","582.26","582.26","582.26","582.26","582.26","582.26","-582.26","0","0","20","-582.26","10","-582.26","0","0","0","0","0","0","50","20","-582.26","10","-582.26","0","0","0","0","0","0","50","20","-582.26","10","-582.26","0","0","0","0","0","0","50","20","-582.26","10","-582.26","0","0","0","0","0","0","50","20","-582.26","10","-582.26","0","0","0","0","0","0","50","20","-582.26","10","-582.26","0","0","0","0","0","0","50","20","582.26","582.26","582.26","582.26","582.26","582.26","582.26","582.26","582.26","582.26","582.26","582.26","582.26","582.26","582.26","582.26","582.26"])

arduino=np.array(dato)
arduino_val=arduino[0]
print(arduino_val)
print(range(np.size(arduino)))

max=range(np.size(arduino))
val1=np.zeros(10001)
val2=np.zeros(10001)
val3=np.zeros(10001)
time.sleep (3)

#Aqui empieza el filtrado de Kalmancito
#while(arduinoSerial):
#Filtro de Kalman
for i in max:
    #Recibiendo señal del sensor
    #z=float(arduinoSerial.readline().strip())
    z=float(arduino_val[i])
    #print(z)

    #PREDICCION
    x_nn=np.dot(F,x_nn) #Listo
    #print(x_nn)

    P=np.add(np.dot(np.dot(F,P),F_trans),Q) #Listo
    #print(P)

    #CORRECCIÓN
    y_m=np.subtract(z,(np.dot(H,x_nn)))
    S=np.add(np.dot(np.dot(H,P),H_trans),R)
    K=np.dot(np.dot(P,H_trans),1/S)

    x_nn=np.add(x_nn,np.dot(K,y_m))
    P=np.dot(np.subtract(eye_3,np.dot(K,H)),P)

    #guardando valores para graficar
    #print(x_nn[2])
    l=x_nn[1]
    val1[i]=x_nn[0]
    val2[i]=x_nn[1]
    val3[i]=x_nn[2]
    #print(val)
    #print(" ")
print(x_nn)
print(val1)
print(val2)
print(val3)
print(" ")

fig = plt.figure()
x=np.arange(0,10001,1)
#plt.subplot(411)
#plt.plot(x,arduino_val)

#plt.subplot(412)
#plt.plot(x,val1)

plt.subplot(111)
plt.title("Acelerometro")
plt.plot(x,arduino_val)
plt.plot(x,val1)
plt.plot(x,val3)
plt.plot(x,val2)
plt.legend(['Aceleracion Sensor','Posicion','Aceleracion','Velocidad'], loc=3)
plt.grid()
plt.show()
import math as mt 
import random as rd 

rd.seed(1)

def RN(m1,m2):
    t=m1*w1+m2*w2+b
    return sigmoide(t)


#dataset 
dataset=[[91,73,0]]

#definisco la derivata della funzione sigmoide
def sigmoide(t):
  try:
    return 1/(1+mt.exp(-0.0001*t))
  except OverflowError:
    return float('inf')
      
def sigmoide_p(t):
    return sigmoide(t)*(1 - sigmoide(t))


def train():

    #pesi inizializzati inizialmente in modo casuale
    w1 = rd.randint(0,9)
    w2 = rd.randint(0,9)
    b = rd.randint(0,9)
    iterazioni =1000#numero di iterazioni 10000
    learning_rate = 0.1 #imposto il learning rate 0.1
    
    for i in range(iterazioni):
       
        ri = rd.randint(0,len(dataset)-1)
      # genero un indice casuale
        point = dataset[ri] # prendo un gatto casuale dal dataset
        
        z = point[0] * w1 + point[1] * w2 + b
        pred = sigmoide(z) # previsione della rete
        
        target = point[2] #il mio valore obiettivo
        
        # costo del punto casuale attuale
        cost = (pred - target)**2
        
        #CALCOLO DELLE DERIVATE PARZIALI
        dcost_dpred = 2 * (pred - target) #derivata parziale del costo rispetto alla previsione
        dpred_dz = sigmoide_p(z) #derivata parziale della previsione rispetto a z
        
        dz_dw1 = point[0] #derivata parziale di z rispetto a w1
        dz_dw2 = point[1] #derivata parziale di z rispetto a w2
        dz_db = 1         #derivata parziale di z rispetto a b
        
        dcost_dz = dcost_dpred * dpred_dz #derivata parziale di z rispetto alla previsione (uso  regola della catena)

        #REGOLA DELLA CATENA
        dcost_dw1 = dcost_dz * dz_dw1 #derivata parziale del costo rispetto a w1
        dcost_dw2 = dcost_dz * dz_dw2 #derivata parziale del costo rispetto a w2
        dcost_db = dcost_dz * dz_db #derivata parziale del costo rispetto a b
        
        #aggiornamento dei pesi e del bias
        w1 = w1 - learning_rate * dcost_dw1
        w2 = w2 - learning_rate * dcost_dw2
        b = b - learning_rate * dcost_db
        
    return w1, w2, b

i=0
for n in range(100):
  w1, w2, b = train()
  x =rd.randint(0,999)#int(input("inserisci primo numero"))
  y =rd.randint(0,999)#int(input("inserisci secodno numero"))
  print(x,y)

  prev=RN(x,y)
  
  if prev > 0.5:
    print(f"{x}>{y}")
    
  else:
    print(f"{y}>{x}")
    
  
  if len(dataset):
    if x>y:
      dataset=dataset+[[x,y,1]]
    else:
      dataset=dataset+[[x,y,0]]
  else:
    if x>y:
      dataset[i]=[x,y,1]
    else:
      dataset[i]=[x,y,0]
      
    i=i+1
 # print(dataset)
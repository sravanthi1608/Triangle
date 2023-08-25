import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
seed_value = 24
np.random.seed(seed_value)
A = np.array([np.random.randint(-6, 6) for _ in range(2)])
B = np.array([np.random.randint(-6, 6) for _ in range(2)])
C = np.array([np.random.randint(-6, 6) for _ in range(2)])
 
print("Assigned points to variables:")
print("A:", A)
print("B:", B)
print("C:", C)

omat = np.array([[0,1],[-1,0]]) 

def line_gen(A, B):
    len = 10
    dim = A.shape[0]
    x_AB = np.zeros((dim, len))
    lam_1 = np.linspace(0, 1, len)
    for i in range(len):
        temp1 = A + lam_1[i] * (B - A)
        x_AB[:, i] = temp1.T
    return x_AB

def dir_vec(A, B):
    return B - A
                                                  #1.1.1
print("direction vector of AB:",dir_vec(A, B))
print("direction vector of BC:",dir_vec(B, C))
print("direction vector of CA:",dir_vec(C, A))
                                                   #1.1.2
def len(A,B):
 return np.linalg.norm(dir_vec(A, B))
 
print("length of AB:",len(A,B)) 
print("length of BC:",len(B,C)) 
print("length of CA:",len(C,A)) 

                                                    #1.1.3
np.set_printoptions(precision=2)
A= np.array(A)
B= np.array(B)
C= np.array(C)

Mat = np.array([[1,1,1],[A[0],B[0],C[0]],[A[1],B[1],C[1]]])

rank = np.linalg.matrix_rank(Mat)

if (rank<=2):
	print("Hence proved that points A,B,C in a triangle are collinear")
else:
	print("The given points are not collinear")
	
                                                       #1.1.4
print("parametric of AB form is x:",A,"+ k",dir_vec(A, B))
print("parametric of BC form is x:",B,"+ k",dir_vec(B, C))
print("parametric of CA form is x:",C,"+ k",dir_vec(C, A))


                                                        #1.1.5
def norm_vec(A, B):
    return np.matmul(omat, dir_vec(A,B))

def line_gen(A, B):
    len = 10
    dim = A.shape[0]
    x_AB = np.zeros((dim, len))
    lam_1 = np.linspace(0, 1, len)
    for i in range(len):
        temp1 = A + lam_1[i] * (B - A)
        x_AB[:, i] = temp1.T
    return x_AB

def side(B,C):
 n=norm_vec(C,B)
 pro=n@B
 return print(n,"x=",pro)
 
side(B,C)
side(C,A)
side(A,B)

x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)
 

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')

#Labeling the coordinates
A1 = A.reshape(-1,1)
B1 = B.reshape(-1,1)
C1 = C.reshape(-1,1)

tri_coords = np.block([[A1, B1, C1]])
plt.scatter(tri_coords[0, :], tri_coords[1, :])
vert_labels = ['A', 'B', 'C']
for i, txt in enumerate(vert_labels):
    offset = 10 if txt == 'C' else -10
    plt.annotate(txt,
                 (tri_coords[0, i], tri_coords[1, i]),
                 textcoords="offset points",
                 xytext=(0, offset),
                 ha='center')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

plt.show()


                                                  #1.1.6

def Area(point1, point2, point3):
    area = 0.5 * np.abs(A[0] * (B[1] - C[1]) +
                        B[0] * (C[1] - A[1]) +
                        C[0] * (A[1] - B[1]))
    return area


area_ABC = Area(A, B, C)
print("Area of triangle ABC:", area_ABC)

                                                 #1.1.7
angle_C = np.arccos((len(B,C)**2 +len(C,A)**2 - len(A,B)**2) / (2 * len(B,C) * len(C,A)))
angle_A = np.arccos((len(C,A)**2 + len(A,B)**2 - len(B,C)**2) / (2 * len(C,A) * len(A,B)))
angle_B = np.arccos((len(A,B)**2 + len(B,C)**2 - len(C,A)**2) / (2 * len(A,B) * len(B,C)))

# Convert angles from radians to degrees
angle_A_deg = np.degrees(angle_A)
angle_B_deg = np.degrees(angle_B)
angle_C_deg = np.degrees(angle_C)

# Print the calculated angles
print("Angle A:", angle_A_deg, "degrees")
print("Angle B:", angle_B_deg, "degrees")
print("Angle C:", angle_C_deg, "degrees")

                                                #1.2.1

def midpoint(P, Q):
    return (P + Q) / 2  

D=midpoint(B,C)
E=midpoint(C,A)
F=midpoint(A,B)

print("D:", list(D))
print("E:", list(E))
print("F:", list(F))

x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)


#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')


#Labeling the coordinates
A1 = A.reshape(-1,1)
B1 = B.reshape(-1,1)
C1 = C.reshape(-1,1)
D1 = D.reshape(-1,1)
E1 = E.reshape(-1,1)
F1 = F.reshape(-1,1)
tri_coords = np.block([[A1,B1,C1,D1,E1,F1]])
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','D','E','F']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.show()

                                              #1.2.2 median equation
def median_equation(A,D):
   n=norm_vec(A,D)
   pro=n@A
   equation=print(n,"x=",pro)
   return equation
   
median_equation(A,D)
median_equation(B,E)
median_equation(C,F)
   
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)

plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')

x_AD = line_gen(A, D)
plt.plot(x_AD[0, :], x_AD[1, :], label='$AD$')

x_BE = line_gen(B, E)
plt.plot(x_BE[0, :], x_BE[1, :], label='$BE$')

x_CF = line_gen(C, F)
plt.plot(x_CF[0, :], x_CF[1, :], label='$CF$')

A1 = A.reshape(-1,1)
B1 = B.reshape(-1,1)
C1 = C.reshape(-1,1)
D1 = D.reshape(-1,1)
E1 = E.reshape(-1,1)
F1 = F.reshape(-1,1)
tri_coords = np.block([[A1,B1,C1,D1,E1,F1]])
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','D','E','F']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(-10,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')


plt.show()

                                              #1.2.3
def line_intersect(n1,A1,n2,A2):
	N=np.block([[n1],[n2]])
	p = np.zeros(2)
	p[0] = n1@A1
	p[1] = n2@A2
	#Intersection
	P=np.linalg.inv(N)@p
	return P
G=line_intersect(norm_vec(B,E),B,norm_vec(C,F),C)
print("("+str(G[0])+","+str(G[1])+")")

x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)
x_BE = line_gen(B,E)
x_CF = line_gen(C,F)


#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_BE[0,:],x_BE[1,:],label='$BE$')
plt.plot(x_CF[0,:],x_CF[1,:],label='$CF$')

#Labeling the coordinates
A1 = A.reshape(-1,1)
B1 = B.reshape(-1,1)
C1 = C.reshape(-1,1)
D1 = D.reshape(-1,1)
E1 = E.reshape(-1,1)
F1 = F.reshape(-1,1)
G1 = G.reshape(-1,1)
tri_coords = np.block([[A1, B1, C1, D1, E1, F1, G1]])
plt.scatter(tri_coords[0, :], tri_coords[1, :])
vert_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
for i, txt in enumerate(vert_labels):
    offset = 10 if txt == 'G' else -10
    plt.annotate(txt,
                 (tri_coords[0, i], tri_coords[1, i]),
                 textcoords="offset points",
                 xytext=(0, offset),
                 ha='center')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.show()
                                                 #1.2.4
G=(A+B+C)/3
AG = np.linalg.norm(G - A)
GD = np.linalg.norm(D - G)

BG = np.linalg.norm(G - B)
GE = np.linalg.norm(E - G)
 
CG = np.linalg.norm(G - C)
GF = np.linalg.norm(F - G)

print("AG/GD= "+str(AG/GD))
print("BG/GE= "+str(BG/GE))
print("CG/GF= "+str(CG/GF))

                                                 #1.2.5  Centroid  
G=(A+B+C)/3

print("G:",G)

                                                  #1.2.6

Mat = np.array([[1,1,1],[A[0],D[0],G[0]],[A[1],D[1],G[1]]])

rank = np.linalg.matrix_rank(Mat)
if (rank<=2):
	print("Hence proved that points A,G,D in a triangle are collinear")
else:
	print("Error")

                                                    #1.2.7
print(f"A - F = {A-F}")
print(f"E - D = {E-D}")

LHS=(A-F)
RHS=(E-D)
#checking LHS and RHS 
if LHS.all()==RHS.all() :
   print("Hence a parallelogram")
else:
    print("Not equal")
    
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)
x_DE = line_gen(D,E)
x_DF = line_gen(D,F)


#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_DE[0,:],x_DE[1,:],label='$DE$')
plt.plot(x_DF[0,:],x_DF[1,:],label='$DF$')

#Labeling the coordinates
A1 = A.reshape(-1,1)
B1 = B.reshape(-1,1)
C1 = C.reshape(-1,1)
D1 = D.reshape(-1,1)
E1 = E.reshape(-1,1)
F1 = F.reshape(-1,1)
tri_coords = np.block([[A1,B1,C1,D1,E1,F1]])
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','D','E','F']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.show()
    
                                                   #1.3.1
def norm_alt(B,C):
 return omat@norm_vec(B,C)
 
print(norm_alt(B,C))

                                                  #1.3.2
def alt_foot(A,B,C):
  m = B-C
  n = np.matmul(omat,m) 
  N=np.vstack((m,n))
  p = np.zeros(2)
  p[0] = m@A 
  p[1] = n@B
  #Intersection
  P=np.linalg.inv(N.T)@p
  return P
  
P=alt_foot(A,B,C)
def result(A,B,C):
 return norm_alt(B,C)@A
 
print("The equation of altitude AP:",norm_alt(B,C),"X=",result(A,B,C))

x_AP = line_gen(A,P)
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_AP[0,:],x_AP[1,:],label='$AP$')


A1 = A.reshape(-1,1)
B1 = B.reshape(-1,1)
C1 = C.reshape(-1,1)
P1 = P.reshape(-1,1)

tri_coords = np.block([[A1,B1,C1,P1]])
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','P']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.show()

                                                    #1.3.3

print(norm_alt(C,A))
print(norm_alt(A,B))

Q=alt_foot(B,C,A)
R=alt_foot(C,A,B)

print("The equation of altitude BQ:",norm_alt(C,A),"X=",result(B,C,A))
print("The equation of altitude CR:",norm_alt(A,B),"X=",result(C,A,B))

#Generating all lines
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)
x_AP = line_gen(A,P)
x_BQ = line_gen(B,Q)
x_CR = line_gen(C,R)
x_AQ = line_gen(A,Q)
x_AR = line_gen(A,R)
x_AE = line_gen(A,E)
x_AF = line_gen(A,F)

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_AP[0,:],x_AP[1,:],label='$AP$')
plt.plot(x_BQ[0,:],x_BQ[1,:],label='$BQ$')
plt.plot(x_CR[0,:],x_CR[1,:],label='$CR$')
plt.plot(x_AE[0,:],x_AE[1,:],linestyle='dotted')
plt.plot(x_AF[0,:],x_AF[1,:],linestyle='dotted')

A1 = A.reshape(-1,1)
B1 = B.reshape(-1,1)
C1 = C.reshape(-1,1)
P1 = P.reshape(-1,1)
Q1 = Q.reshape(-1,1)
R1 = R.reshape(-1,1)

tri_coords = np.block([[A1,B1,C1,P1,Q1,R1]])
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','P','Q','R']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.show()

                                                    #1.3.4
H = line_intersect(dir_vec(A,B),C,dir_vec(C,A),B)
print("H:",H)

x_AB = line_gen(A,B)  
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)
x_AP = line_gen(A,P)
x_AQ = line_gen(A,Q)
x_BQ = line_gen(B,Q)
x_CR = line_gen(C,R)
x_AR = line_gen(A,R)
x_CH = line_gen(C,H)
x_BH = line_gen(B,H)
x_AH = line_gen(A,H)

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_AP[0,:],x_AP[1,:],label='$AP$')
plt.plot(x_BQ[0,:],x_BQ[1,:],label='$BQ_1$')
plt.plot(x_AQ[0,:],x_AQ[1,:],linestyle = 'dashed',label='$AQ_1$')
plt.plot(x_CR[0,:],x_CR[1,:],label='$CR_1$')
plt.plot(x_AR[0,:],x_AR[1,:],linestyle = 'dashed',label='$AR_1$')
plt.plot(x_CH[0,:],x_CH[1,:],label='$CH$')
plt.plot(x_BH[0,:],x_BH[1,:],label='$BH$')
plt.plot(x_AH[0,:],x_AH[1,:],linestyle = 'dashed',label='$AH$')

#Labeling the coordinates
A1 = A.reshape(-1,1)
B1 = B.reshape(-1,1)
C1 = C.reshape(-1,1)
P1 = P.reshape(-1,1)
Q1 = Q.reshape(-1,1)
R1 = R.reshape(-1,1)
H1 = H.reshape(-1,1)
tri_coords = np.block([[A1,B1,C1,P1,Q1,R1,H1]])
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','P','Q','F','H']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.show()
                                                      #1.3.5
result = int(((A - H).T) @ (B - C))

if result == 0:
  print("(A - H)^T (B - C) = 0\nHence Verified.")

else:
  print("(A - H)^T (B - C)) != 0\nHence the given statement is wrong")
  
x_AB = line_gen(A,B)  
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)
x_AP = line_gen(A,P)
x_AQ = line_gen(A,Q)
x_BQ = line_gen(B,Q)
x_CR = line_gen(C,R)
x_AR = line_gen(A,R)
x_CH = line_gen(C,H)
x_BH = line_gen(B,H)
x_AH = line_gen(A,H)

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_AP[0,:],x_AP[1,:],label='$AP$')
plt.plot(x_BQ[0,:],x_BQ[1,:],label='$BQ_1$')
plt.plot(x_AQ[0,:],x_AQ[1,:],linestyle = 'dashed',label='$AQ_1$')
plt.plot(x_CR[0,:],x_CR[1,:],label='$CR_1$')
plt.plot(x_AR[0,:],x_AR[1,:],linestyle = 'dashed',label='$AR_1$')
plt.plot(x_CH[0,:],x_CH[1,:],label='$CH$')
plt.plot(x_BH[0,:],x_BH[1,:],label='$BH$')
plt.plot(x_AH[0,:],x_AH[1,:],linestyle = 'dashed',label='$AH$')

#Labeling the coordinates
A1 = A.reshape(-1,1)
B1 = B.reshape(-1,1)
C1 = C.reshape(-1,1)
P1 = P.reshape(-1,1)
Q1 = Q.reshape(-1,1)
R1 = R.reshape(-1,1)
H1 = H.reshape(-1,1)
tri_coords = np.block([[A1,B1,C1,P1,Q1,R1,H1]])
#tri_coords = np.vstack((A,B,C,alt_foot(A,B,C),alt_foot(B,A,C),alt_foot(C,A,B),H)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','P','Q','R','H']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.show()
                                                    #1.4.1

def perpendicular_bisector(B,C):
  n=(B+C)/2
  p=n@dir_vec(B,C)
  equation=print(dir_vec(B,C),"x=",p)
  return equation
 
print("Equation of perpendicular bisector of BC:")
perpendicular_bisector(B,C)
print("Equation of perpendicular bisector of CA:")
perpendicular_bisector(C,A)
print("Equation of perpendicular bisector of AB:")
perpendicular_bisector(A,B)


                                                     #1.4.2
O = line_intersect(dir_vec(A,B),F,dir_vec(C,A),E)
print("O:",O)

x_AB = line_gen(A, B)
x_BC = line_gen(B, C)
x_CA = line_gen(C, A)
x_OD = line_gen(O,D)
x_OE = line_gen(O,E)
x_OF = line_gen(O,F)
# Plotting all lines
plt.plot(x_AB[0, :], x_AB[1, :], label='$AB$')
plt.plot(x_BC[0, :], x_BC[1, :], label='$BC$')
plt.plot(x_CA[0, :], x_CA[1, :], label='$CA$')
plt.plot(x_OD[0, :], x_OD[1, :], label='$OD$')
plt.plot(x_OE[0, :], x_OE[1, :], label='$OE$')
plt.plot(x_OF[0, :], x_OF[1, :], label='$OF$')
A1 = A.reshape(-1,1)
B1 = B.reshape(-1,1)
C1 = C.reshape(-1,1)
O1 = O.reshape(-1,1)
D1 = D.reshape(-1,1)
E1 = E.reshape(-1,1)
F1 = F.reshape(-1,1)
tri_coords = np.block([[A1,B1,C1,O1,D1,E1,F1]])
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','O','D','E','F']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.show()
                                               #1.4.3
R=int((O-((B+C)/2))@(B-C))
if R == 0:
  print("(O-((B+C)/2)).(B-C) = 0\nHence Verified.")

else:
  print("(O-((B+C)/2)).(B-C) != 0\nHence the given statement is wrong")
  
                                                 #1.4.4
a = len(O,A)
b = len(O,B)
c = len(O,C)
print(" OA, OB, OC are respectively", a,",", b,",",c, ".")
print(" OA = OB = OC. Hence verified")

                                                #1.4.5
r=a
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)
x_OA = line_gen(O,A)
x_OB = line_gen(O,B)
x_OC = line_gen(O,C)

def circ_gen(O,r):
	len = 50
	theta = np.linspace(0,2*np.pi,len)
	x_circ = np.zeros((2,len))
	x_circ[0,:] = r*np.cos(theta)
	x_circ[1,:] = r*np.sin(theta)
	x_circ = (x_circ.T + O).T
	return x_circ

x_ccirc= circ_gen(O,r)

#Plotting the circumcircle
plt.plot(x_ccirc[0,:],x_ccirc[1,:],label='$circumcircle$')
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_OA[0, :], x_OA[1, :], label='$OA$')
plt.plot(x_OB[0, :], x_OB[1, :], label='$OB$')
plt.plot(x_OC[0, :], x_OC[1, :], label='$OC$')

A1 = A.reshape(-1,1)
B1 = B.reshape(-1,1)
C1 = C.reshape(-1,1)
O1 = O.reshape(-1,1)
#Labeling the coordinates
tri_coords = np.block([[A1,B1,C1,O1]])
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','O']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() 
plt.axis('equal')
plt.show()

                                                      #1.4.6
                                                      
angle_BOC=np.arccos((len(O,B)**2 +len(O,C)**2 - len(B,C)**2) / (2 * len(O,B) * len(O,C)))

angle_BAC=np.arccos((len(A,B)**2 +len(C,A)**2 - len(B,C)**2) / (2 * len(A,B) * len(C,A)))

angle_BOC_deg = 360 - np.degrees(angle_BOC)
angle_BAC_deg = np.degrees(angle_BAC)
print("angle BOC = " + str(angle_BOC_deg))
print("angle BAC = " + str(angle_BAC_deg))
print("\nangle BOC = 2 times angle BAC\nHence the give statement is correct")

x_AB = line_gen(A,B)
x_AC = line_gen(A,C)
x_OB = line_gen(O,B)
x_OC = line_gen(O,C)

x_circ= circ_gen(O,r)

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_AC[0,:],x_AC[1,:],label='$BC$')
plt.plot(x_OB[0,:],x_OB[1,:],label='$OB$')
plt.plot(x_OC[0,:],x_OC[1,:],label='$OB$')
#Plotting the circumcircle
plt.plot(x_ccirc[0,:],x_ccirc[1,:],label='$circumcircle$')


#Labeling the coordinates
A1 = A.reshape(-1,1)
B1 = B.reshape(-1,1)
C1 = C.reshape(-1,1)
O1 = O.reshape(-1,1)
tri_coords = np.block([[A1,B1,C1,O1]])
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','O']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() 
plt.axis('equal')
plt.show()

                                                            #1.5.1
def unit_vec(A,B):
	return ((B-A)/np.linalg.norm(B-A))

def angular_bisector(A,B,C):
 E= unit_vec(A,B) + unit_vec(A,C)
 F=np.array([E[1],(E[0]*(-1))])
 C1= F@(A.T)
 return print(F,"*x = ",C1)
print("Internal Angular bisector of angle A is:")
angular_bisector(A,B,C)
print("Internal Angular bisector of angle B is:")
angular_bisector(B,C,A)
print("Internal Angular bisector of angle C is:")
angular_bisector(C,A,B)

                                                             #1.5.2 
t = norm_vec(B,C) 
n1 = t/np.linalg.norm(t)
t = norm_vec(C,A)
n2 = t/np.linalg.norm(t)
t = norm_vec(A,B)
n3 = t/np.linalg.norm(t)

I=line_intersect(n1-n3,B,n1-n2,C)
print("I:",I)

x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)
x_BI = line_gen(B,I)
x_CI = line_gen(C,I)
x_IA = line_gen(I,A)
#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_CI[0,:],x_CI[1,:],label='$CI$')
plt.plot(x_BI[0,:],x_BI[1,:],label='$BI$')
I1 = I.reshape(-1,1)
#Labeling the coordinates
tri_coords = np.block([A1,B1,C1,I1])
plt.scatter(tri_coords[:,0], tri_coords[:,1])
vert_labels = ['A','B','C','I']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.show()

                                                      #1.5.3
angle_BAI=np.arccos((len(B,A)**2 +len(A,I)**2 - len(I,B)**2) / (2 * len(B,A) * len(A,I)))
angle_CAI=np.arccos((len(C,A)**2 +len(A,I)**2 - len(I,C)**2) / (2 * len(C,A) * len(A,I)))

#Calculating the angles BAI and CAI
angle_BAI = np.degrees(angle_BAI)
angle_CAI = np.degrees(angle_CAI)


# Print the angles
print("Angle BAI:", angle_BAI)
print("Angle CAI:", angle_CAI)

if np.isclose(angle_BAI, angle_CAI):
  print("Angle BAI is approximately equal to angle CAI.")
else:
  print("error")

plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_IA[0,:],x_IA[1,:],label='$IA$')

tri_coords = np.block([A1,B1,C1,I1])
plt.scatter(tri_coords[:,0], tri_coords[:,1])
vert_labels = ['A','B','C','I']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
                 
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.show()
                                                  #1.5.4
k1 = 1
k2 = 1
def inradius(A,B,C,n1,n2,n3):
 p = np.zeros(2)
 p[0] = n1 @ B - k1 * n2 @ C
 p[1] = n2 @ C - k2 * n3 @ A

 N = np.block([[n1 - k1 * n2],[ n2 - k2 * n3]])
 I = np.linalg.inv(N)@p
 return n1 @ (B-I)

print(f"Distance from I to BC:",inradius(A,B,C,n1,n2,n3))

                                                           #1.5.5   
print("Distance between I and AB:",inradius(C,A,B,n3,n1,n2))
print("Distance between I and AC:",inradius(B,C,A,n2,n3,n1))

                                                       #1.5.7
rad=inradius(A,B,C,n1,n2,n3)
#Generating all lines
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)

x_icirc= circ_gen(I,rad)
#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')

plt.plot(x_icirc[0,:],x_icirc[1,:],label='$incircle$')
A1 = A.reshape(-1,1)
B1 = B.reshape(-1,1)
C1 = C.reshape(-1,1)
I1 = I.reshape(-1,1)
#Labeling the coordinates
tri_coords = np.block([A1,B1,C1,I1])
plt.scatter(tri_coords[:,0], tri_coords[:,1])
vert_labels = ['A','B','C','I']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
                 
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.show()
#1.5.8
p=pow(len(B,C),2)
q=2*(dir_vec(B,C)@dir_vec(B,I))
r=pow(len(I,B),2)-rad*rad

Discre=q*q-4*p*r

k=(dir_vec(B,I)@dir_vec(B,C))/(dir_vec(B,C)@dir_vec(B,C))

D_3=B+k*dir_vec(B,C)
print("the point of tangency of incircle by side BC is ",D_3)

#Generating all lines
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)

x_icirc= circ_gen(I,rad)
#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_icirc[0,:],x_icirc[1,:],label='$incircle$')
plt.plot(D_3[0],D_3[1],label='$D_3$')
A1 = A.reshape(-1,1)
B1 = B.reshape(-1,1)
C1 = C.reshape(-1,1)
I1 = I.reshape(-1,1)
D3 = D_3.reshape(-1,1)
#Labeling the coordinates
tri_coords = np.block([[A1,B1,C1,D3,I1]])
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','D3','I']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.show()
                                          #1.5.9

k1=(dir_vec(A,I)@dir_vec(A,B))/(dir_vec(A,B)@dir_vec(A,B))
k2=(dir_vec(C,I)@dir_vec(C,A))/(dir_vec(C,A)@dir_vec(C,A))
F_3=A+(k1*dir_vec(A,B))
E_3=C+(k2*dir_vec(C,A))
print("E_3: ",E_3)
print("F_3: ",F_3)

x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)
x_icirc= circ_gen(I,rad)

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_icirc[0,:],x_icirc[1,:],label='$incircle$')
A1 = A.reshape(-1,1)
B1 = B.reshape(-1,1)
C1 = C.reshape(-1,1)
I1 = I.reshape(-1,1)
E3 = E_3.reshape(-1,1)
F3 = F_3.reshape(-1,1)
tri_coords = np.block([[A1,B1,C1,I1,E3,F3]])
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','I','E3','F3']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid(True) # minor
plt.axis('equal')
plt.show()
                                                    #1.5.10
print("AE_3=", len(A,E_3) ,"\nAF_3=", len(A,F_3) ,"\nBD_3=", len(B,D_3) ,"\nBF_3=", len(B,F_3) ,"\nCD_3=", len(C,D_3) ,"\nCE_3=",len(C,E_3))

                                                        #1.5.11
a = len(B,C)
b = len(C,A)
c = len(A,B)

m = (c+b-a)/2
n = (c+a-b)/2
p = (b+a-c)/2

print("m=",m ,"n=",n ,"p=",p)



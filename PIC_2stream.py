import numpy as np
import matplotlib.pyplot as plt

ME = 9.10938356e-31
QE = 1.602176634e-19
EPS0 = 8.8541878128e-12
KB = 1.380649e-23
ev2k = QE/KB

from pathlib import Path
import os

def weights(xp,dx):
    icell = (int)(np.absolute(np.floor(xp/dx)))
    x_node_left = icell*dx
    w0 = (xp - x_node_left)/dx
    w1 = 1.0 - w0
    return w0, w1, icell

def push_particles_dt(x,vx,dt,L,dx,Efield,Qfield,Qp,Ze):
    N_part = np.size(x)
    N_nodes = np.size(Qfield)
    qmdt2 = Ze*QE/ME*dt/2.0
    # Reset charge on nodes
    for i in range(N_nodes):
        Qfield[i] = 0.0
    # Push particles and weight their charge to the nodes
    for i in range(N_part):
        # Find weights and cell number
        w0, w1, icell = weights(x[i],dx)
        # interpolate e-field at particle location
        Ex = Efield[icell]*w0 + Efield[icell+1]*w1
        # Boris-Bunemann, push velocity (e-field only)
        vx[i] += 2.0*qmdt2*Ex
        # Boris-Bunemann, push positions
        x[i] += vx[i]*dt
        # Periodicity on x-coordinate of the particle
        x[i] = np.mod(x[i],L) # if particle leaves through one side, it returns through the other
        # Find new weights and cell number
        w0, w1, icell = weights(x[i],dx)
        # Add charge to grid nodes
        Qfield[icell] += (Qp*w0)
        Qfield[icell+1] += (Qp*w1)
        
def poisson_1d_periodic(rhs):
    N_nodes = np.size(rhs)
    phi = np.zeros(N_nodes)
    phi[0] = 0.0
    for i in range(N_nodes):
        phi[0] += ((i+1)*rhs[i])
    phi[0] = phi[0]/N_nodes
    phi[1] = rhs[0] + 2.0*phi[0]
    for i in range(2,N_nodes):
        phi[i] = rhs[i-1] + 2.0*phi[i-1] - phi[i-2]
    return phi

def gradphi(phi,dx):
    N_nodes = np.size(phi)
    Efield = np.zeros(N_nodes)
    for i in range(1,N_nodes-1):
        Efield[i] = -(phi[i+1] - phi[i-1])/2.0/dx
    Efield[0] = -(phi[1] - phi[0])/2.0/dx
    Efield[N_nodes-1] = -(phi[N_nodes - 1] - phi[N_nodes - 2])/2.0/dx
    return Efield

def efield(Q_nodes, L, dx):
    N_nodes = np.size(Q_nodes)
    rho_e = np.zeros(N_nodes)
    # Find electron charge density from charge dividing by covolume (volume around a node)
    rho_e[0] = Q_nodes[0]/(0.5*dx)
    rho_e[N_nodes-1] = Q_nodes[N_nodes-1]/(0.5*dx)
    for i in range(1,N_nodes-1):
        rho_e[i] = Q_nodes[i]/dx
    # Average ion charge density
    rho_i = np.sum(Q_nodes)/L
    # Assemble RHS of poisson equation
    rhs = (rho_i - rho_e)/EPS0*dx*dx
    # Solve poisson equation
    phi = poisson_1d_periodic(rhs)
    # find e-field
    E_nodes = gradphi(phi,dx)
    return E_nodes, phi, rho_e

def main():

    # Domain
    L = 0.01
    N_nodes = 300
    dx = L/(N_nodes - 1)
    grid = np.linspace(0,L,N_nodes)


    # Time
    N_steps = 200
    dt = 5.0e-11
    
    # Particle Distribution
    Ze = -1.0                             # Charge number of electrons
    n0 = 1e16                             # Beam density
    Te = 1.0 * ev2k                       # Beam temperature [ev->K]
    Vthe = np.sqrt(2.0*KB*Te/ME)          # Electron thermal speed [m/s]
    Ue = 5.0*Vthe                         # Beam drift velocity [m/s]
    N_part = 50000                        # Number of computational particles
    p2c = n0*L/N_part                     # Physical-to-Computational ratio (number of real particles a computational particle represents)
    Qp = QE*Ze*p2c                        # Charge of a microparticle [C]
  
    my_file = Path('./Electrostatic energy - ' + str(N_part) + '.txt')
    if my_file.is_file():
        os.remove(my_file)

    # Particle list, made of 2 electron beams
    x = np.random.uniform(0,L,N_part)                 # Initial position of the particles [m]
    Npart_beam1 = (int)(np.floor(N_part/2.0))         # Number of particles in beam 1 [#]
    Npart_beam2 = N_part - Npart_beam1                 # Number of particles in beam 2 [#]
    beam1 = np.random.normal(Ue, Vthe, Npart_beam1)   # Initial speed distribution for beam 1 [m/s]
    beam2 = np.random.normal(-Ue, Vthe, Npart_beam2)  # Initial speed distribution for beam 2 [m/s]
    vx = np.concatenate((beam1, beam2))               # Initial velocity of the particles [m/s]
    
    # Fields
    Q_nodes = np.zeros(N_nodes)
    E_nodes = np.zeros(N_nodes)
    
    # Time series
    Esquare = np.zeros(N_steps)

    electrEnergy_file = open('Electrostatic energy - ' + str(N_part) + '.txt', 'w')
    
    for n in range(N_steps):
        # Push particles in time ("PARTICLE STEP")
        push_particles_dt(x,vx,dt,L,dx,E_nodes,Q_nodes,Qp,Ze)
        # Solve for electric field ("MESH STEP")
        E_nodes, phi, rho_e = efield(Q_nodes, L, dx)
        # Integral of electrostatic energy in the domain
        Esquare[n] = 0.5*EPS0*np.sum(E_nodes**2)
        electrEnergy_file.write(str(Esquare[n]) + '\n')
        if n % 10 == 0:
            print(str(n/N_steps*100) + '%')

        plt.plot(x[0:Npart_beam1],vx[0:Npart_beam1]/Vthe,'r.')
        plt.plot(x[Npart_beam1:],vx[Npart_beam1:]/Vthe,'b.')
        plt.xlim([0,L])
        plt.ylim([-12,12])
        plt.xlabel('x [m]')
        plt.ylabel('vx / vth')
        plt.title('Electron beam phase space')
        plt.draw()
        plt.pause(.02)
        if n % 25 == 0:
            plt.savefig('report_phasespace_t' +str(n)+'.png', dpi = 600)
        plt.clf()
        
    plt.figure(2)
    time = np.arange(0, N_steps*dt, dt)
    plt.plot(time,Esquare)
    plt.xlabel('Time')
    plt.ylabel(r"Electrostatic energy (0.5*$\epsilon_0$*$E^{2}$)")
    plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    # Show the minor grid as well. Style it in very light gray as a thin,
    # dotted line.
    plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    # Make the minor ticks and gridlines show.
    plt.minorticks_on()
    #plt.savefig('electrostatic.png', dpi = 600)
    plt.show()

if __name__ == '__main__':
    main()   
    
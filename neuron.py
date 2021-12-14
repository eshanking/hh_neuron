import numpy as np
import matplotlib.pyplot as plt

class Neuron():

    def __init__(self,
                v0=None,
                vK=-12,
                vNa=115,
                vL=10,
                gL=0.3,
                gK=36,
                gNa=120,
                C=1,
                dt=0.001,
                inj=None):
        
        if v0 is None:
            self.v = 0
        else:
            self.v = v0

        # initialize neuron state
        self.I = 0
        self.vK = vK
        self.vNa = vNa 
        self.gK = gK
        self.gNa = gNa
        self.gL = gL
        self.vL = vL
        self.C = C
        self.dt = dt
        
        # initialize gating variables
        self.alpha_n = 0
        self.beta_n = 0
        self.alpha_m = 0
        self.beta_m = 0
        self.alpha_h = 0
        self.beta_h = 0

        self._UpdateGateTimeConstants(self.v)

        # set steady state gating probabilities
        self.n = self.alpha_n / (self.alpha_n + self.beta_n)
        self.m = self.alpha_m / (self.alpha_m + self.beta_m)
        self.h = self.alpha_h / (self.alpha_h + self.beta_h)
        
        # initialize data variables
        self.vt = []
        self.t_vect = []
        self.t = 0

        if inj is None:
            self.inj = 0*np.ones(100)
        else:
            self.inj = inj
        
        self.n_vect = [] # n over time for phase plane analysis
        self.m_vect = []
        self.h_vect = []

        return


    def plot(self):
        t = self.t_vect*self.dt
        fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(10,4))

        ax[0].plot(t,self.vt,label='membrane potential')
        ax[0].set_ylabel('mV')
        ax[0].set_xlabel('ms')
        ax[0].set_ylim(-10,110)

        ax[0].plot(t,self.inj,c='red',label='injection current')
        ax[0].legend(frameon=False)
        
        ax[1].plot(self.vt,self.n_vect,label='n')
        ax[1].plot(self.vt,self.m_vect,c='green',label='m')
        ax[1].plot(self.vt,self.h_vect,c='red',label='h')
        ax[1].set_xlabel('mV')
        ax[1].set_xlim(-10,110)
        ax[1].set_ylim(-0.1,1.1)
        ax[1].legend(frameon=False)
        return fig,ax

    def simulate(self,n_timestep):
        
        t = np.arange(n_timestep)
        self.t_vect = t
        self.vt = np.zeros(n_timestep)
        self.n_vect = np.zeros(n_timestep)
        self.m_vect = np.zeros(n_timestep)
        self.h_vect = np.zeros(n_timestep)
        
        for i in t:
            self.increment_state()
            self.vt[i] = self.v
            self.n_vect[i] = self.n
            self.h_vect[i] = self.h
            self.m_vect[i] = self.m
            self.t+=1

    def increment_state(self):

        dt = self.dt

        dvdt = self.dv_dt()
        self.v = self.v + dvdt*dt
        dndt = self.dn_dt(self.v)
        dmdt = self.dm_dt(self.v)
        dhdt = self.dh_dt(self.v)

        self.n = self.n + dndt*dt
        self.m = self.m + dmdt*dt
        self.h = self.h + dhdt*dt
        self._UpdateGateTimeConstants(self.v)

    def dv_dt(self):

        C = self.C

        if len(self.inj) > self.t:
            I = self.inj[self.t]
            
        else:
            I = 0

        self.I = I
        dvdt = I/C + (self.INa() + self.IK() + self.IL())/C
        return dvdt

    def INa(self):
        return self.gNa*(self.m**3)*self.h*(self.vNa-self.v)

    def IK(self):
        return self.gK*(self.n**4)*(self.vK-self.v)
    
    def IL(self):
        return self.gL*(self.vL-self.v)

    
    def dn_dt(self,v=None,n=None):
        if v is None:
            v = self.v
        if n is None:
            n = self.n
        dndt = self.alpha_n*(1-n) - self.beta_n*n
        
        return dndt

    def dm_dt(self,v=None,m=None):
        if m is None:
            m = self.m
        if v is None:
            v = self.v
        dmdt = self.alpha_m*(1-m) - self.beta_m*m

        return dmdt

    def dh_dt(self,v=None,h=None):
        if h is None:
            h = self.h
        if v is None:
            v = self.v
        dhdt = self.alpha_h*(1-h) - self.beta_h*h
        return dhdt



    def _UpdateGateTimeConstants(self, Vm):
        """Update time constants of all gates based on the given Vm"""
        self.alpha_n = .01 * ((10-Vm) / (np.exp((10-Vm)/10)-1))
        self.beta_n = .125*np.exp(-Vm/80)
        self.alpha_m = .1*((25-Vm) / (np.exp((25-Vm)/10)-1))
        self.beta_m = 4*np.exp(-Vm/18)
        self.alpha_h = .07*np.exp(-Vm/20)
        self.beta_h = 1/(np.exp((30-Vm)/10)+1)
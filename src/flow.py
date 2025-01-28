import numpy as np 
import matplotlib.pyplot as plt

class Baseflow:

    """
    A class to represent the base flow in a fluid dynamics simulation.

    Attributes
    ----------
    ny : int
        Number of grid points in the y-direction.
    U : np.ndarray or None
        Velocity component in the x-direction.
    Uy : np.ndarray or None
        Derivative of U with respect to y.
    Uyy : np.ndarray or None
        Second derivative of U with respect to y.
    Ux : np.ndarray or None
        Derivative of U with respect to x.
    V : np.ndarray or None
        Velocity component in the y-direction.
    Vy : np.ndarray or None
        Derivative of V with respect to y.
    Vx : np.ndarray or None
        Derivative of V with respect to x.
    W : np.ndarray or None
        Velocity component in the z-direction.
    Wx : np.ndarray or None
        Derivative of W with respect to x.
    Wy : np.ndarray or None
        Derivative of W with respect to y.
    Wxy : np.ndarray or None
        Mixed derivative of W with respect to x and y.
    P : np.ndarray or None
        Pressure field.
    Q : np.ndarray or None
        Combined state vector.

    Methods
    -------
    get_Q():
        Returns the combined state vector Q.
    get_primitive():
        Returns the primitive variables U, V, and P.
    get_U():
        Returns the velocity component U.
    get_V():
        Returns the velocity component V.
    get_P():
        Returns the pressure field P.
    get_Ux():
        Returns the derivative of U with respect to x.
    get_Uy():
        Returns the derivative of U with respect to y.
    get_Uyy():
        Returns the second derivative of U with respect to y.
    get_Vy():
        Returns the derivative of V with respect to y.
    get_Vx():
        Returns the derivative of V with respect to x.
    Blasius(y, x=1, Uinf=1, nu=1):
        Computes the Blasius boundary layer profile.
    """

    def __init__(self,Grid):
        """
        Constructs all the necessary attributes for the Baseflow object.

        Parameters
        ----------
        Grid : object
            Grid object containing grid information.
        """

        self.ny=Grid.Ny
        self.U = None
        self.Uy = None
        self.Uyy = None
        self.Ux = None
        self.V = None
        self.Vy = None
        self.Vx = None
        self.W = None
        self.Wx = None
        self.Wy = None
        self.Wxy = None
        self.P = None
        self.Q = None

    def get_Q(self):
        """
        Returns the combined state vector Q.

        Returns
        -------
        np.ndarray
            Combined state vector Q.
        """

        if np.any(self.Q==None):
            self.Q = np.block([self.U,self.V,self.P]).flatten()
        return self.Q
    
    def get_primitive(self):
        """
        Returns the primitive variables U, V, and P.

        Returns
        -------
        tuple
            Tuple containing U, V, and P.
        """
        return self.get_U(), self.get_V(), self.get_P()
    
    def get_U(self):
        """
        Returns the velocity component U.

        Returns
        -------
        np.ndarray
            Velocity component U.
        """
        if np.any(self.U==None):
            self.U=self.Q[:self.ny]
        return self.U
    
    def get_V(self):
        """
        Returns the value of V.

        If V is None, it is calculated based on the values of Q.

        Returns:
            numpy.ndarray: The value of V.
        """
        if np.any(self.V==None):
            self.V=self.Q[self.ny:2*self.ny]
        return self.V
    
    def get_P(self):
        """
        Returns the pressure field P.

        Returns
        -------
        np.ndarray
            Pressure field P.
        """

        if np.any(self.P==None):
            self.P=self.Q[2*self.ny:3*self.ny-1]
        return self.P
    
    def get_Ux(self):
        """
        Returns the derivative of U with respect to x.

        Returns
        -------
        np.ndarray or None
            Derivative of U with respect to x.
        """

        return self.Ux
    
    def get_Uy(self):
        """
        Returns the derivative of U with respect to y.

        Returns
        -------
        np.ndarray or None
            Derivative of U with respect to y.
        """

        return self.Uy
    
    def get_Uyy(self):
        """
        Returns the second derivative of U with respect to y.

        Returns
        -------
        np.ndarray or None
            Second derivative of U with respect to y.
        """

        return self.Uyy
    
    def get_Vy(self):
        """
        Returns the derivative of V with respect to y.

        Returns
        -------
        np.ndarray or None
            D
        """
        return self.Vy
    
    def get_Vx(self):
        """
        Returns the derivative of V with respect to x.

        Returns
        -------
        np.ndarray or None
            Derivative of V with respect to x.
        """
        
        return self.Vx

    def Blasius(self, y,x=1,Uinf=1,nu=1):
        '''
        Input:
            y: array of height of channel or flat plate
            x: location along plate
            base_type: type of base_flow ['channel','plate']
        Output base flow for plane Poiseuille flow between two parallel plates or Blasius profile
            U: U mean velocity
            Uy: dU/dy of mean belocity
            Uyy: d^2 U/dy^2 of mean velocity
            Ux: dU/dx of mean velocity
            V: wall normal mean velocity
            Vy: dVdy
            Vx: dVdx
        '''
        # assume nu=1 as default
        # print("Debugging in Blasius")
        # print(f"y = {y}")
        y_uniform=np.linspace(y.min(),y.max(),y.size*100)
        eta=y_uniform*np.sqrt(Uinf/(2.*nu*x))
        deta=np.diff(eta) # assume uniform grid would mean deta is all the same
        # IC for blasius f'''-ff'' = 0
        # or changed to coupled first order ODE
        #     f'' = \int -f*f'' deta
        #     f'  = \int f'' deta
        #     f   = \int f' deta
        # initialize and ICs
        # make lambda function
        f_fs = lambda fs: np.array([
            -fs[2]*fs[0], # f'' = \int -f*f'' deta
            fs[0],        # f'  = \int f'' deta
            fs[1],        # f   = \int f' deta
            1.])       # eta = \int 1 deta
        fs = np.zeros((eta.size,4))
        fs[0,0] = 0.332057336215195*np.sqrt(2.) #0.469600 # f''
        fs[0,1] = 0.       # f'
        fs[0,2] = 0.       # f
        fs[0,3] = eta[0]      # eta
        # step through eta
        for i,ideta in enumerate(deta):
            k1 = ideta*f_fs(fs[i]);
            k2 = ideta*f_fs(fs[i]+k1/2);
            k3 = ideta*f_fs(fs[i]+k2/2);
            k4 = ideta*f_fs(fs[i]+k3);
            fs[i+1] = fs[i] + (k1+(k2*2)+(k3*2)+k4)/6;
        #print('eta,f,fp,fpp = ')
        #print(fs[:,::-1])
        # print(f"fp before interp = {fs[:,1]}")
        fpp=np.interp(y,y_uniform,fs[:,0])
        fp =np.interp(y,y_uniform,fs[:,1])
        f  =np.interp(y,y_uniform,fs[:,2])
        eta=np.interp(y,y_uniform,fs[:,3])
        fppp = np.gradient(fpp,eta)
        #print("eta = ",eta)
        #print("fp",fp)
        # print(f"fp after interp = {fp}")
        U  = Uinf*fp # f'
        Uy = fpp*np.sqrt(Uinf**3/(2.*nu*x))
        Uyy= fppp*(Uinf**2/(2.*nu*x))
        Ux = fpp*(-eta/(2.*x))
        V  = np.sqrt(nu*Uinf/(2.*x))*(eta*fp - f)
        Vy = Uinf/(2.*x) * eta*fpp
        Vx = np.sqrt(nu*Uinf/(8.*x**3)) * (-eta*fp + f - eta**2*fpp)
        self.U = U[np.newaxis,:]
        self.Uy = Uy[np.newaxis,:]
        self.Uyy = Uyy[np.newaxis,:]
        self.Ux = Ux [np.newaxis,:]
        self.V = V[np.newaxis,:]
        self.Vy = Vy[np.newaxis,:]
        self.Vx = Vx[np.newaxis,:]
        self.W = np.zeros_like(self.V)
        self.Wx = np.zeros_like(self.V)
        self.Wxy = np.zeros_like(self.V)
        self.Wy = np.zeros_like(self.V)
        self.P = np.zeros_like(self.V)

    def set_velocity_field(self, xi, eta, u, v, p):
        """
        Sets the velocity field for the baseflow.

        Args:
            u (np.ndarray): U velocity component.
            v (np.ndarray): V velocity component.
        """

        self.U = u 
        self.V = v 
        self.P = p
        # for now assume W = 0 
        self.W = np.zeros_like(self.U)

        # xi varies on rows
        # eta varies on columns
        self.Ux = np.gradient(u, xi[:,0], axis=0)
        self.Uy = np.gradient(u, eta[0,:], axis=1)
        self.Uyy = np.gradient(self.Uy, eta[0,:], axis=1)
        self.Vx = np.gradient(v, xi[:,0], axis=0)
        self.Vy = np.gradient(v, eta[0,:], axis=1)
        self.Wx = np.zeros_like(self.W)
        self.Wy = np.zeros_like(self.W)
        self.Wxy = np.zeros_like(self.W)

    def set_velocity_field_staggered(self, xi_grid_c, eta_grid_c, xi_grid_u, eta_grid_u, xi_grid_v, eta_grid_v, u_grid_stag, v_grid_stag, p_grid_stag):

        """Set velocity field and compute derivatives on staggered grid."""

        # Store grids and fields
        self.xi_grid_c = xi_grid_c
        self.eta_grid_c = eta_grid_c
        self.xi_grid_u = xi_grid_u
        self.eta_grid_u = eta_grid_u
        self.xi_grid_v = xi_grid_v
        self.eta_grid_v = eta_grid_v

        self.u_grid = u_grid_stag
        self.v_grid = v_grid_stag
        self.p_grid = p_grid_stag

        # Get 1D coordinate arrays
        xi_c = xi_grid_c[:, 0]   # Cell centers xi
        xi_u = xi_grid_u[:, 0]   # U-velocity xi
        xi_v = xi_grid_v[:, 0]   # V-velocity xi

        eta_c = eta_grid_c[0, :] # Cell centers eta
        eta_u = eta_grid_u[0, :] # U-velocity eta
        eta_v = eta_grid_v[0, :] # V-velocity eta

        # Compute U derivatives
        # ---------------------
        # Ux is naturally staggered at p-points
        self.Ux = np.zeros_like(self.xi_grid_c)
        for j in range(eta_grid_c.shape[1]):
            self.Ux[1:-1, j] = (u_grid_stag[1:, j] - u_grid_stag[:-1, j]) / \
                (xi_u[1:] - xi_u[:-1])

        # Uy needs special treatment since u is at u-points
        self.Uy = np.zeros_like(u_grid_stag)
        for i in range(u_grid_stag.shape[0]):
            self.Uy[i, :] = np.gradient(u_grid_stag[i, :], eta_u)

        # Uyy at u-points
        self.Uyy = np.zeros_like(u_grid_stag)
        for i in range(u_grid_stag.shape[0]):
            self.Uyy[i, :] = np.gradient(self.Uy[i, :], eta_u)

        # Compute V derivatives
        # ---------------------
        # Vx needs special treatment since v is at v-points
        self.Vx = np.zeros_like(v_grid_stag)
        for j in range(v_grid_stag.shape[1]):
            self.Vx[:, j] = np.gradient(v_grid_stag[:, j], xi_v)

        # Vy is naturally staggered at p-points
        self.Vy = np.zeros_like(self.xi_grid_c)
        for i in range(xi_grid_c.shape[0]):
            self.Vy[i, 1:-1] = (v_grid_stag[i, 1:] - v_grid_stag[i, :-1]) / \
                (eta_v[1:] - eta_v[:-1])

        # Keep W-related terms if needed
        if hasattr(self, 'W'):
            self.Wx = np.zeros_like(self.W)
            self.Wy = np.zeros_like(self.W)
            self.Wxy = np.zeros_like(self.W)

    def interpolate_to_centers(self, staggered_field, direction):
        """
        Interpolate staggered field to cell centers.
        
        Parameters:
            staggered_field: Field at staggered locations
            direction: 'u' for u-velocity, 'v' for v-velocity
        """
        if direction == 'u':
            centered = np.zeros((self.xi_grid_c.shape[0], staggered_field.shape[1]))
            centered[1:-1, :] = 0.5 * (staggered_field[1:, :] + staggered_field[:-1, :])
            # Extrapolate to boundaries
            centered[0, :] = 2*centered[1, :] - centered[2, :]
            centered[-1, :] = 2*centered[-2, :] - centered[-3, :]
        else:  # v-velocity
            centered = np.zeros((staggered_field.shape[0], self.eta_grid_c.shape[1]))
            centered[:, 1:-1] = 0.5 * (staggered_field[:, 1:] + staggered_field[:, :-1])
            # Extrapolate to boundaries
            centered[:, 0] = 2*centered[:, 1] - centered[:, 2]
            centered[:, -1] = 2*centered[:, -2] - centered[:, -3]
        return centered

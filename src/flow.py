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

    def FalknerSkan(self, y, x=1, Uinf=1, nu=1, beta=0):
        '''
        Input:
            y: array of height of channel or flat plate
            x: location along plate
            Uinf: free stream velocity
            nu: kinematic viscosity
            beta: pressure gradient parameter (Falkner-Skan wedge angle parameter)
                 beta = 0 corresponds to Blasius flat plate
                 beta > 0 corresponds to accelerating flow (favorable pressure gradient)
                 beta < 0 corresponds to decelerating flow (adverse pressure gradient)

        Output base flow for Falkner-Skan boundary layer
            U: U mean velocity
            Uy: dU/dy of mean velocity
            Uyy: d^2U/dy^2 of mean velocity
            Ux: dU/dx of mean velocity
            V: wall normal mean velocity
            Vy: dV/dy
            Vx: dV/dx
        '''
        from scipy.integrate import solve_ivp

        # Create a fine uniform grid for integration
        y_uniform = np.linspace(y.min(), y.max(), y.size*100)

        # Define similarity variable eta
        m = beta / (2 - beta)  # Power-law exponent for external velocity U_e ~ x^m

        # Define eta max for integration (ensure it's large enough for asymptotic behavior)
        eta_max = 20.0  # Usually sufficient for boundary layer to reach asymptotic behavior

        # Calculate similarity variable eta for original y grid
        eta = y_uniform * np.sqrt(Uinf * (1 + m) / (2 * nu * x))

        # Create a uniform eta grid for the shooting method
        eta_uniform = np.linspace(0, eta_max, 1000)

        # Define the ODE system for Falkner-Skan
        # Converting to first-order system:
        # Let z = [f', f'', f]
        # z[0] = f'
        # z[1] = f''
        # z[2] = f
        # Then:
        # z'[0] = f'' = z[1]
        # z'[1] = f''' = -z[2]*z[1] - beta*(1-z[0]^2)
        # z'[2] = f' = z[0]
        def falkner_skan_ode(t, z):
            return [z[1], -z[2]*z[1] - beta*(1 - z[0]**2), z[0]]

        # Shooting method to find the correct f''(0) value
        def shoot(fpp0_guess):
            # Initial conditions [f'(0), f''(0), f(0)]
            z0 = [0.0, fpp0_guess, 0.0]

            # Solve ODE with this guess
            sol = solve_ivp(
                falkner_skan_ode, 
                [0, eta_max], 
                z0, 
                method='RK45',
                t_eval=eta_uniform, 
                rtol=1e-6, 
                atol=1e-6
            )

            # Get f'(eta_max) - should approach 1.0 for correct solution
            fp_end = sol.y[0, -1]

            # Return the error
            return fp_end - 1.0

        # Initial guess for f''(0) based on beta
        if beta == 0:
            fpp0_guess = 0.332057336215195 * np.sqrt(2.)  # Blasius solution
        elif beta > 0:
            if beta <= 1:
                fpp0_guess = 0.332057336215195 * np.sqrt(2.) * (1 + 0.5*beta)
            else:
                fpp0_guess = 1.2 * np.sqrt(1 + beta)
        else:
            # For negative beta, more careful with initial guess
            if beta >= -0.1988:  # Theoretical limit for attached flow
                fpp0_guess = 0.332057336215195 * np.sqrt(2.) * (1 + beta)
            else:
                raise ValueError(f"Beta = {beta} is below -0.1988, which may not have an attached flow solution")

        # Use binary search for the shooting method
        # (more robust than Newton's method for this problem)
        tol = 1e-6
        max_iter = 50

        # Initial bounds for binary search
        if beta >= 0:
            low = 0.1
            high = 10.0
        else:
            # For negative beta, we need different bounds
            low = 0.01
            high = 0.332057336215195 * np.sqrt(2.)  # Blasius value as upper bound

        # Binary search iteration
        iter_count = 0
        err_low = shoot(low)
        err_high = shoot(high)

        # Check if our bounds are suitable
        if err_low * err_high >= 0:
            # If signs are the same, adjust bounds
            if abs(err_low) < abs(err_high):
                high = low
                low = low / 10
            else:
                low = high
                high = high * 10
            err_low = shoot(low)
            err_high = shoot(high)

            # If still not bracketing the root, use the initial guess
            if err_low * err_high >= 0:
                fpp0 = fpp0_guess
            else:
                # Continue with binary search
                fpp0 = (low + high) / 2
        else:
            fpp0 = (low + high) / 2

        # Main binary search loop
        while abs(high - low) > tol and iter_count < max_iter:
            fpp0 = (low + high) / 2
            err = shoot(fpp0)

            if abs(err) < tol:
                break

            if err * err_low < 0:
                high = fpp0
            else:
                low = fpp0
                err_low = err

            iter_count += 1

        # Final f''(0) value from shooting method
        fpp0 = (low + high) / 2

        # Solve the ODE with the correct initial condition
        z0 = [0.0, fpp0, 0.0]
        sol = solve_ivp(
            falkner_skan_ode, 
            [0, eta_max], 
            z0, 
            method='RK45',
            t_eval=eta_uniform, 
            rtol=1e-6, 
            atol=1e-6
        )

        # Extract solution
        fp = sol.y[0, :]  # f'
        fpp = sol.y[1, :]  # f''
        f = sol.y[2, :]  # f
        eta_sol = sol.t  # eta values

        # Interpolate to original y grid
        fp_interp = np.interp(eta, eta_sol, fp)
        fpp_interp = np.interp(eta, eta_sol, fpp)
        f_interp = np.interp(eta, eta_sol, f)

        # Calculate f'''
        # Use the ODE to get f''' directly instead of numerical differentiation
        fppp = np.zeros_like(fpp_interp)
        for i in range(len(fppp)):
            fppp[i] = -f_interp[i] * fpp_interp[i] - beta * (1 - fp_interp[i]**2)

        # Calculate velocity components and derivatives based on the solution
        # External velocity (power law)
        Ue = Uinf  # At the specified x-location

        # Scale factor for derivatives
        scale_factor = np.sqrt(Ue * (1 + m) / (2 * nu * x))

        # Interpolate solution back to original y grid
        fp = fp_interp
        fpp = fpp_interp
        f = f_interp
        fppp = fppp    # f''' already calculated from the ODE

        # External velocity (power law)
        Ue = Uinf  # At the specified x-location

        # Calculate velocity components and derivatives
        U = Ue * fp  # u = Ue * f'

        # Scale factor for derivatives
        scale_factor = np.sqrt(Ue * (1 + m) / (2 * nu * x))

        # Velocity derivatives
        Uy = fpp * scale_factor
        Uyy = fppp * (scale_factor**2)
        Ux = Ue * m / (2 * x) * (2 * fp - eta * fpp)

        # Wall-normal velocity
        V = np.sqrt(nu * Ue / (2 * x)) * ((1 - m) * f + m * eta * fp)

        # Wall-normal velocity derivatives
        Vy = Ue / (2 * x) * ((1 - m) * fp + m * eta * fpp)
        Vx = np.sqrt(nu * Ue / (8 * x**3)) * (-(1 - m) * f - m * eta * fp + (1 - m) * eta * fp + m * eta**2 * fpp)

        # Assign to self for consistent interface with Blasius method
        self.U = U[np.newaxis, :]
        self.Uy = Uy[np.newaxis, :]
        self.Uyy = Uyy[np.newaxis, :]
        self.Ux = Ux[np.newaxis, :]
        self.V = V[np.newaxis, :]
        self.Vy = Vy[np.newaxis, :]
        self.Vx = Vx[np.newaxis, :]
        self.W = np.zeros_like(self.V)
        self.Wx = np.zeros_like(self.V)
        self.Wxy = np.zeros_like(self.V)
        self.Wy = np.zeros_like(self.V)
        self.P = np.zeros_like(self.V)

        return

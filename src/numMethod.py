import numpy as np 
from scipy.special import factorial
from scipy import integrate
from scipy.interpolate import griddata, interp1d
from scipy import interpolate
from mpi4py import MPI
from inputOutput import print_rz
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
colors = sns.color_palette("husl",10)

class derivativeOperators:
    
    def __init__(self, ygrid, method):

        self.method = method
        self.ygrid = ygrid

        if self.method == "cheby" or method == "cheby2":
            self.Dy = self.map_D_cheby(self.ygrid, 1, True)
            self.Dyy = self.map_D_cheby(self.ygrid, 2, True)

        elif self.method == "fd" or self.method == "geometric" or self.method=="uniform":
            self.Dy = self.set_D_FD(self.ygrid,d=1,order=2, output_full=True, uniform=False)
            self.Dyy = self.set_D_FD(self.ygrid,d=2,order=2, output_full=True, uniform=False)

        else:
            raise ValueError("Invalid method chosen for derivatives")

    def get_D_Coeffs_FD(self, s, d=2, h=1, TaylorTable=False):

        '''
        Solve arbitrary stencil points s of length N with order of derivatives d<N
        can be obtained from equation on MIT website
        http://web.media.mit.edu/~crtaylor/calculator.html
        where the accuracy is determined as the usual form O(h^(N-d))
    
        Inputs:
            s: array like input of stencil points e.g. np.array([-3,-2,-1,0,1])
            d: order of desired derivative
        '''
        # solve using Taylor Table instead
        if TaylorTable:
            # create Taylor Table
            N=s.size # stencil length
            b=np.zeros(N)
            A=np.zeros((N,N))
            A[0,:]=1. # f=0
            for row in np.arange(1,N):
                A[row,:]=1./factorial(row) * s**row # f^(row) terms
            b[d]=-1
            x = -np.linalg.solve(A,b)
            return x
        
        # use MIT stencil
        else: 
            # let's solve an Ax=b problem
            N=s.size # stencil length
            A=[]
            for i in range(N):
                A.append(s**i)
            b=np.zeros(N)
            b[d] = factorial(d)
            x = np.linalg.solve(np.matrix(A),b)
            return x

    def set_D_FD(self, y,yP=None,order=2,T=2.*np.pi,d=2,reduce_wall_order=True,output_full=False,periodic=False,uniform=False):
        '''
        Input:
            y: array of y values of channel
            order: order of accuracy desired (assuming even e.g. 2,4,6,...)
            d: dth derivative
            T: period if using Fourier
        Output:
            D: (n-2 by n) dth derivative of order O(h^order) assuming uniform y spacing
        '''

        if isinstance(order,int):
            h = y[1]-y[0] # uniform spacing
            if not uniform:
                xi=np.linspace(0,1,y.size)
                h=xi[1] - xi[0]
            n = y.size
            ones=np.ones(n)
            I = np.eye(n)
            # get coefficients for main diagonals
            N=order+d # how many pts needed for order of accuracy
            if N>n:
                raise ValueError('You need more points in your domain, you need %i pts and you only gave %i'%(N,n))
            Nm1=N-1 # how many pts needed if using central difference is equal to N-1
            if (d % 2 != 0): # if odd derivative
                Nm1+=1 # add one more point to central, to count the i=0 0 coefficient
            # stencil and get Coeffs for diagonals
            s = np.arange(Nm1)-int((Nm1-1)/2) # stencil for central diff of order
            smax=s[-1] # right most stencil used (positive range)
            Coeffs = self.get_D_Coeffs_FD(s,d=d)
            # loop over s and add coefficient matrices to D
            D = np.zeros_like(I)
            si = np.nditer(s,('c_index',))
            while not si.finished:
                i = si.index
                if si[0]==0:
                    diag_to_add = np.diag(Coeffs[i] * ones,k=si[0])
                else:
                    diag_to_add = np.diag(Coeffs[i] * ones[:-abs(si[0])],k=si[0])

                D += diag_to_add
                if periodic:
                    if si[0]>0:
                        diag_to_add = np.diag(Coeffs[i]*ones[:abs(si[0])],k=si[0]-n)
                    elif si[0]<0:
                        diag_to_add = np.diag(Coeffs[i]*ones[:abs(si[0])],k=si[0]+n)
                    if si[0]!=0:
                        D += diag_to_add
                        
                si.iternext()
            if not periodic:
                # alter BC so we don't go out of range on bottom of channel
                for i in range(0,smax):
                    # for ith row, set proper stencil coefficients
                    if reduce_wall_order:
                        if (d%2!=0): # if odd derivative
                            s = np.arange(Nm1-1)-i # stencil for shifted diff of order-1
                        else:
                            s = np.arange(Nm1)-i # stencil for shifted diff of order-1
                    else:
                        s = np.arange(N)-i # stencil for shifted diff of order
                    Coeffs = self.get_D_Coeffs_FD(s,d=d)
                    D[i,:] = 0. # set row to zero
                    D[i,s+i] = Coeffs # set row to have proper coefficients

                    # for -ith-1 row, set proper stencil coefficients
                    if reduce_wall_order:
                        if (d%2!=0): # if odd derivative
                            s = -(np.arange(Nm1-1)-i) # stencil for shifted diff of order-1
                        else:
                            s = -(np.arange(Nm1)-i) # stencil for shifted diff of order-1
                    else:
                        s = -(np.arange(N)-i) # stencil for shifted diff of order
                    Coeffs = self.get_D_Coeffs_FD(s,d=d)
                    D[-i-1,:] = 0. # set row to zero
                    D[-i-1,s-i-1] = Coeffs # set row to have proper coefficients

            if output_full:
                D = (1./(h**d)) * D # do return the full matrix
            else:
                D = (1./(h**d)) * D[1:-1,:] # do not return the top or bottom row
            if not uniform:
                D = self.map_D_FD(D,y,order=order,d=d,reduce_wall_order=reduce_wall_order,output_full=output_full,periodic=periodic,uniform=uniform)
        elif order=='fourier':
            npts = y.size # number of points
            n = np.arange(0,npts)
            j = np.arange(0,npts)
            N,J = np.meshgrid(n,j,indexing='ij')
            D = 2.*np.pi/T*0.5*(-1.)**(N-J)*1./np.tan(np.pi*(N-J)/npts)
            D[J==N]=0
            
            if d==2:
                D=D@D
        return D

    def set_D_P(self, y,yP=None,staggered=False,return_P_location=False,full_staggered=False,order=2,d=2,reduce_wall_order=True,output_full=False,periodic=False,uniform=True):
        #DyyP=set_D_P(y,yP, order=4,d=2,output_full=True,uniform=False,staggered=True)
        '''
        Input:
            y: array of y values, (velocity points) or vertical velocity points if full staggered
            yP: array of y values, (pressure points) and horizontal velocity points if full staggered
            order: order of accuracy desired (assuming even e.g. 2,4,6,...)
            d: dth derivative
        Output:
            D: (n-2 by n) dth derivative of order O(h^order) assuming uniform y spacing
        '''
        if staggered:
            if full_staggered:
                h = y[0]-yP[0] # uniform spacing
            else:
                h = y[1]-yP[0] # uniform spacing
        else:
            h = y[1]-y[0] # uniform spacing
        if (not uniform) or full_staggered:
            if staggered:
                xi=np.linspace(0,1,y.size+yP.size)
            else:
                xi=np.linspace(0,1,y.size)
            h=xi[1] - xi[0]
        if staggered:
            n = 2*y.size-1
            if full_staggered:
                n = 2*y.size
        else:
            n=y.size
        ones=np.ones(n)
        I = np.eye(n)
        # get coefficients for main diagonals
        N=order+d # how many pts needed for order of accuracy
        if N>n:
            raise ValueError('You need more points in your domain, you need %i pts and you only gave %i'%(N,n))
        Nm1=N-1 # how many pts needed if using central difference is equal to N-1
        if (d % 2 != 0): # if odd derivative
            Nm1+=1 # add one more point to central, to count the i=0 0 coefficient
        if staggered and (d%2==0): # staggered and even derivative
            Nm1+=2  # add two more points for central
        # stencil and get Coeffs for diagonals
        if staggered:
            s = (np.arange(-Nm1+2,Nm1,2))#)-int((Nm1-1))) # stencil for central diff of order
        else:
            s = np.arange(Nm1)-int((Nm1-1)/2) # stencil for central diff of order
        #print('sc = ',s)
        smax=s[-1] # right most stencil used (positive range)
        Coeffs = self.get_D_Coeffs_FD(s,d=d)
        # loop over s and add coefficient matrices to D
        D = np.zeros_like(I)
        si = np.nditer(s,('c_index',))
        while not si.finished:
            i = si.index
            if si[0]==0:
                diag_to_add = np.diag(Coeffs[i] * ones,k=si[0])
            else:
                diag_to_add = np.diag(Coeffs[i] * ones[:-abs(si[0])],k=si[0])

            D += diag_to_add
            if periodic:
                if si[0]>0:
                    diag_to_add = np.diag(Coeffs[i]*ones[:abs(si[0])],k=si[0]-n)
                elif si[0]<0:
                    diag_to_add = np.diag(Coeffs[i]*ones[:abs(si[0])],k=si[0]+n)
                if si[0]!=0:
                    D += diag_to_add
                    
            si.iternext()
        if not periodic:
            # alter BC so we don't go out of range on bottom 
            smax_range=np.arange(0,smax)
            for i in smax_range:
                # for ith row, set proper stencil coefficients
                if reduce_wall_order:
                    if (d%2!=0): # if odd derivative
                        if staggered:
                            if i%2!=0: # odd row, for P_location
                                s = np.arange(-i,2*(Nm1-1)-i-1,2) # stencil for shifted diff of order-1
                            else: # even row, for velocity location
                                s = np.arange(-i+1,2*(Nm1-1)-i,2) # stencil for shifted diff of order-1
                        else:
                            s = np.arange(Nm1-1)-i # stencil for shifted diff of order-1
                    else:
                        if staggered:
                            if i%2!=0: # odd row, for P_location
                                s = np.arange(-i+1,2*Nm1-i-2,2)-1 # stencil for shifted diff of order-1
                            else: # even row, for velocity location
                                s = np.arange(-i+1,2*Nm1-i-2,2) # stencil for shifted diff of order-1
                        else:
                            s = np.arange(Nm1)-i # stencil for shifted diff of order-1
                else:
                    if staggered:
                        s = np.arange(-i+1,2*N-i,2) # stencil for shifted diff of order
                    else:
                        s = np.arange(N)-i # stencil for shifted diff of order
                #print('i, s,scol = ',i,', ',s,',',s+i)
                Coeffs = self.get_D_Coeffs_FD(s,d=d)
                D[i,:] = 0. # set row to zero
                D[i,s+i] = Coeffs # set row to have proper coefficients

                # for -ith-1 row, set proper stencil coefficients
                if reduce_wall_order:
                    if (d%2!=0): # if odd derivative
                        if staggered:
                            if i%2!=0: # odd row, for P_location
                                s = -np.arange(-i+1,2*(Nm1-1)-i,2)+1 # stencil for shifted diff of order-1
                            else: # if even row, return velocity location
                                s = -np.arange(-i+1,2*(Nm1-1)-i,2) # stencil for shifted diff of order-1
                        else:
                            s = -(np.arange(Nm1-1)-i) # stencil for shifted diff of order-1
                    else:
                        if staggered:
                            if i%2!=0: # odd row, for P_location
                                s = -np.arange(-i+1,2*Nm1-i-2,2)+1 # stencil for shifted diff of order-1
                            else: # even row, for velocity location
                                s = -np.arange(-i+1,2*Nm1-i-2,2) # stencil for shifted diff of order-1
                        else:
                            s = -(np.arange(Nm1)-i) # stencil for shifted diff of order-1
                else:
                    if staggered:
                        s = -np.arange(-i+1,2*N-i,2) # stencil for shifted diff of order-1
                    else:
                        s = -(np.arange(N)-i) # stencil for shifted diff of order
                #print('i,row, s,scol = ',i,',',-i-1,', ',s,',',s-i-1)
                Coeffs = self.get_D_Coeffs_FD(s,d=d)
                D[-i-1,:] = 0. # set row to zero
                D[-i-1,s-i-1] = Coeffs # set row to have proper coefficients

        # filter out for only Pressure values
        if staggered:
            if full_staggered:
                if return_P_location:
                    D = D[::2,1::2]
                else:
                    D = D[1::2,::2]
            else:
                if return_P_location:
                    D = D[1::2,::2]
                else:
                    D = D[::2,1::2]
            
            #D[[0,-1],:]=D[[1,-2],:] # set dPdy at wall equal to dPdy off wall
        if output_full:
            D = (1./(h**d)) * D # do return the full matrix
        else:
            D = (1./(h**d)) * D[1:-1,:] # do not return the top or bottom row
        if not uniform:
                D = self.map_D_FD(D,y,yP,order=order,d=d,reduce_wall_order=reduce_wall_order,output_full=output_full,periodic=periodic,uniform=uniform,
                          staggered=staggered,return_P_location=return_P_location,full_staggered=full_staggered)
        return D 

    def map_D_FD(self, D,y,yP=None,order=2,d=2,reduce_wall_order=True,output_full=False,periodic=False,uniform=True,
          staggered=False,return_P_location=False,full_staggered=False):
        if not uniform:
            xi=np.linspace(0,1,y.size)
            if d==1: # if 1st derivative operator d(.)/dy = d(.)/dxi * dxi/dy
                if staggered and return_P_location==False:
                    D1 = self.set_D_P(xi,order=order,d=1,reduce_wall_order=reduce_wall_order,output_full=output_full,periodic=periodic,uniform=True,
                                 staggered=False,return_P_location=False)
                    dydxi=D1@y
                elif full_staggered and return_P_location:
                    if(0):
                        xi=np.linspace(0,1,y.size+yP.size)
                        D1 = self.set_D_P(xi,order=order,d=1,reduce_wall_order=reduce_wall_order,output_full=output_full,periodic=periodic,uniform=True,
                                    staggered=False,return_P_location=False)
                        yall=np.zeros(y.size+yP.size)
                        yall[::2]  = yP
                        yall[1::2] = y
                        dydxi=(D1@yall)[::2]
                    elif(0):
                        xi=np.linspace(0,1,y.size)
                        xi2=np.insert(xi,0,-xi[1])
                        xiP=(xi2[1:]+xi2[:-1])/2.
                        D1 = self.set_D_P(xi,xiP,order=order,d=1,reduce_wall_order=reduce_wall_order,output_full=output_full,periodic=periodic,uniform=True,
                                    staggered=True,return_P_location=True,full_staggered=True)
                        dydxi=D1@y
                    else:
                        dydxi = D@y # matrix multiply in python3
                else:
                    dydxi = D@y # matrix multiply in python3
                dxidy = 1./dydxi # element wise invert
                return D*dxidy[:,np.newaxis] # d(.)/dy = d(.)/dxi * dxi/dy
            elif d==2: # d^2()/dy^2 = d^2()/dxi^2 (dxi/dy)^2 + d()/dxi d^2xi/dy^2
                xi=np.linspace(0,1,y.size)
                xiP=(xi[1:]+xi[:-1])/2.
                if staggered and return_P_location and full_staggered==False:
                    D1=self.set_D_P(xi,xiP,order=order,d=1,reduce_wall_order=reduce_wall_order,output_full=output_full,periodic=periodic,uniform=True,
                                staggered=True,return_P_location=return_P_location)
                    dydxi = D1@y
                elif full_staggered:
                    if(0):
                        xiall=np.linspace(0,1,y.size+yP.size)
                        D1 = self.set_D_P(xiall,order=order,d=1,reduce_wall_order=reduce_wall_order,output_full=output_full,periodic=periodic,uniform=True,
                                    staggered=False,return_P_location=False)
                        yall=np.zeros(y.size+yP.size)
                        yall[::2]  = yP
                        yall[1::2] = y
                        dydxi=(D1@yall)[::2]
                    else:
                        xi=np.linspace(0,1,y.size)
                        xiv=np.insert(xi,0,-xi[1])
                        xiP=(xiv[1:]+xiv[:-1])/2.
                        D1 = self.set_D_P(xi,xiP,order=order,d=1,reduce_wall_order=reduce_wall_order,output_full=output_full,periodic=periodic,uniform=True,
                                     staggered=True,return_P_location=True,full_staggered=True)
                        dydxi=(D1@y)
                else:
                    D1=self.set_D_P(xi,xiP,order=order,d=1,reduce_wall_order=reduce_wall_order,output_full=output_full,periodic=periodic,uniform=True,
                                staggered=False,return_P_location=return_P_location)
                    dydxi = D1@y
                dxidy = 1./dydxi # element wise invert
                if staggered and full_staggered==False:
                    D2=self.set_D_P(xi,xiP,order=order,d=2,reduce_wall_order=reduce_wall_order,output_full=output_full,periodic=periodic,uniform=True,
                                staggered=return_P_location,return_P_location=return_P_location)
                    d2xidy2 = -(D2@y)*(dxidy)**3
                    D1p=self.set_D_P(xi,xiP,order=order,d=1,reduce_wall_order=reduce_wall_order,output_full=output_full,periodic=periodic,uniform=True,
                                staggered=True,return_P_location=return_P_location)
                    return (D*(dxidy[:,np.newaxis]**2)) + (D1p*d2xidy2[:,np.newaxis])  # d^2()/dy^2 = d^2()/dxi^2 (dxi/dy)^2 + d()/dxi d^2xi/dy^2
                elif full_staggered:
                    if(0):
                        xiall=np.linspace(0,1,y.size+yP.size)
                        D2 = self.set_D_P(xiall,order=order,d=2,reduce_wall_order=reduce_wall_order,output_full=output_full,periodic=periodic,uniform=True,
                                     staggered=False,return_P_location=False,full_staggered=False)
                        yall=np.zeros(y.size+yP.size)
                        yall[::2]  = yP
                        yall[1::2] = y
                        d2xidy2 = -((D2@yall)[::2])*(dxidy)**3
                    else:
                        xi=np.linspace(0,1,y.size)
                        xiv=np.insert(xi,0,-xi[1])
                        xiP=(xiv[1:]+xiv[:-1])/2.
                        D2 = set_D_P(xi,xiP,order=order,d=2,reduce_wall_order=reduce_wall_order,output_full=output_full,periodic=periodic,uniform=True,
                                     staggered=True,return_P_location=True,full_staggered=True)
                        d2xidy2 = -(D2@y)*(dxidy)**3
                    xi=np.linspace(0,1,y.size)
                    xiv=np.insert(xi,0,-xi[1])
                    xiP=(xiv[1:]+xiv[:-1])/2.
                    D1p=self.set_D_P(xi,xiP,order=order,d=1,reduce_wall_order=reduce_wall_order,output_full=output_full,periodic=periodic,uniform=True,
                                staggered=staggered,return_P_location=return_P_location,full_staggered=full_staggered)
                    return (D*(dxidy[:,np.newaxis]**2)) + (D1p*d2xidy2[:,np.newaxis])  # d^2()/dy^2 = d^2()/dxi^2 (dxi/dy)^2 + d()/dxi d^2xi/dy^2
                else:
                    d2xidy2 = -(D@y)*(dxidy)**3
                    return (D*(dxidy[:,np.newaxis]**2)) + (D1*d2xidy2[:,np.newaxis])  # d^2()/dy^2 = d^2()/dxi^2 (dxi/dy)^2 + d()/dxi d^2xi/dy^2
            else:
                print('Cannot do this order of derivative with non-uniform mesh.  your input order of derivative = ',d)
        else:
            return D

    def set_D_cheby(self, x, order=1):

        N=len(x)-1
        if x[0]==1.:
            order=1
        elif x[0]==-1.:
            order=-1
        D = np.zeros((N+1,N+1))
        c = np.ones(N+1)
        c[0]=c[N]=2.
        for j in np.arange(N+1):
            cj=c[j]
            xj=x[j]
            for k in np.arange(N+1):
                ck=c[k]
                xk=x[k]
                if j!=k:
                    D[j,k] = cj*(-1)**(j+k) / (ck*(xj-xk))
                elif ((j==k) and ((j!=0) and (j!=N))):
                    D[j,k] = -xj/(2.*(1.-xj**2))
                elif ((j==k) and (j==0 or j==N)):
                    D[j,k] = xj*(2.*N**2 + 1)/6.
        return D
        if order==2:
            D=D@D
            return D

    def map_D_cheby(self, x, order=1, need_map=False):
        if need_map:
            N=len(x)-1
            xi = np.cos(np.pi*np.arange(N+1)[::-1]/N)
            if order==1: # if 1st derivative operator d(.)/dy = d(.)/dxi * dxi/dy
                D= self.set_D_cheby(xi, order=1)
                #dxdxi = D@x # matrix multiply in python3
                Dy = np.diag(1./(D@x))@D # d(.)/dy = d(.)/dxi * dxi/dy
                return Dy
            elif order==2: # d^2()/dy^2 = d^2()/dxi^2 (dxi/dy)^2 + d()/dxi d^2xi/dy^2
                D1= self.set_D_cheby(xi, order=1)
                D = D1@D1 # second derivative
                dxdxi = D1@x
                dxidx = 1./dxdxi # element wise invert
                d2xidx2 = -(D@x)*(dxidx)**3
                Dyy = (D*(dxidx[:,np.newaxis]**2)) + (D1*d2xidx2[:,np.newaxis])
                return Dyy
            else:
                print('Cannot do this order of derivative with non-uniform mesh.  your input order of derivative = ',order)
        else:
            if order == 1:
                Dy = self.set_D_cheby(self, x, order)
                return Dy
            elif order == 2:
                Dyy = self.set_D_cheby(self, x, order)
                return Dyy

class Grid:

    def __init__(self, Nx, Ny, hx, x0, ymax, method="cheby", need_map=True):

        self.Nx = Nx
        self.Ny = Ny
        self.hx = hx 
        self.x0 = x0 
        self.ymax = ymax
        self.method = method
        self.need_map = need_map

        self.Dy = None
        self.Dyy = None

        self.xgrid = None
        self.ygrid = None

        self.R = None # radius of curvature
        # can use constant kappa to simulate Gortler type flows
        # assigned in the initialization 
        self.kappa = None 
        self.dtheta_dxi = np.zeros((Nx, Ny))
        self.d2theta_dxi2 = np.zeros((Nx, Ny))

        # h in general is a grid metric term that does NOT need to be 1
        # for the simple cases one would want to simulate using the "flatplate" 
        # initialization, h = 1 should suffice
        self.theta = np.zeros((Nx,1))
        self.xi_grid = self.xgrid

        self.set_yGrid(self.method, 0.0, self.ymax, self.Ny)
        self.set_xGrid(self.Nx, self.hx)

        # Metric terms for flat plate
        self.x_xi = np.ones((Nx, Ny))
        self.x_eta = np.zeros((Nx, Ny))
        self.y_xi = np.zeros((Nx, Ny))
        self.y_eta = np.ones((Nx, Ny))

        self.xi_x = np.ones((Nx, Ny))
        self.xi_y= np.zeros((Nx, Ny))
        self.eta_x = np.zeros((Nx, Ny))
        self.eta_y = np.ones((Nx, Ny))

        self.J = self.x_xi * self.y_eta - self.x_eta * self.y_xi

        self.J11 = self.xi_x
        self.J12 = self.xi_y
        self.J21 = self.eta_x
        self.J22 = self.eta_y

        self.dJ11_dxi = np.zeros((Nx,Ny))
        self.dJ11_deta = np.zeros((Nx,Ny))

        self.dJ12_dxi = np.zeros((Nx,Ny))
        self.dJ12_deta = np.zeros((Nx,Ny))
        
        self.dJ21_dxi = np.zeros((Nx,Ny))
        self.dJ21_deta = np.zeros((Nx,Ny))

        self.dJ22_dxi = np.zeros((Nx,Ny))
        self.dJ22_deta = np.zeros((Nx,Ny))

        self.h = np.ones((Nx,Ny))

        derivativeOps = derivativeOperators(self.ygrid, self.method)
        self.Dy = derivativeOps.Dy
        self.Dyy = derivativeOps.Dyy

    def set_yGrid(self, method, a, b, ny, delta=5.0):

        # TODO: figure out when to use this vs the other one 

        if method == "cheby":
            xi = np.cos(np.pi*np.arange(ny-1,-1,-1)/(ny-1))
            y = b*(xi+1)/2.0 + a*(1.0-xi)/2.0
            self.ygrid = y

        if method == "cheby2":
            xi = np.cos(np.pi*np.arange(ny-1,-1,-1)/(ny-1))
            y_i = 20.0
            buf1 = y_i * b / (b - 2 * y_i)
            buf2 = 1 + 2 * buf1 / b
            y = buf1 * (1 + xi) / (buf2 - xi)
            self.ygrid = y

        elif method == "fd":
            y = np.linspace(a, b, ny)
            y = b * (1. + (np.tanh(delta * (y / b - 1.)) / np.tanh(delta)))
            self.ygrid = y

    def set_xGrid(self, Nx, hx):
        self.xgrid = np.arange(0, (Nx+1) * hx, hx) + self.x0 
        self.xi_grid = self.xgrid
        # self.xgrid = np.linspace(a, b, nx)
    
class surfaceImport:
    def __init__(self, X, Y, U, V, P, N_xi, eta_max, N_eta, method="cheby", need_map=True, Uinf=1.0, nu=1.0):

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

        self.x_field = X
        self.y_field = Y
        self.u_field = U
        self.v_field = V
        self.p_field = P
        self.Nx = N_xi
        self.Ny = N_eta
        self.eta_max = eta_max
        self.method = method
        self.need_map = need_map
        
        self.xi_grid = None
        self.hx = None
        self.eta_grid = None
        self.xgrid = None
        self.ygrid = None
        self.u_grid = None
        self.v_grid = None
        self.p_grid = None
        self.kappa = None

        self.Uinf = Uinf
        self.nu = nu

        self.generate_curvilinear_grid()
        self.transform_velocities()
        self.normalize_fields()

        derivativeOps = derivativeOperators(self.ygrid, self.method)
        self.Dy = derivativeOps.Dy
        self.Dyy = derivativeOps.Dyy

        self.check_gcl_static_2d()

    def generate_curvilinear_grid(self):
        """Generate a curvilinear grid based on the input data."""

        from scipy.integrate import cumulative_trapezoid
        X = self.x_field
        Y = self.y_field
        x_surface, y_surface = X[:, 0], Y[:, 0]

        # Calculate theta (angle of tangent to airfoil surface)
        dx = np.gradient(x_surface)
        dy = np.gradient(y_surface)
        theta = np.arctan2(dy, dx)
        
        # Calculate x1 (distance along airfoil surface)
        ds = np.sqrt(dx**2 + dy**2)
        x1 = cumulative_trapezoid(ds, initial=0)  # This ensures x1 has the same length as airfoil_x and airfoil_y

        xi_distribution = 'uniform'
        # Generate ξ distribution
        if xi_distribution == 'uniform':
            xi = np.linspace(0, x1[-1], self.Nx)
        elif xi_distribution == 'cosine':
            xi = x1[-1] * 0.5 * (1 - np.cos(np.linspace(0, np.pi, self.Nx)))
        else:
            raise ValueError("Invalid ξ_distribution. Choose 'uniform' or 'cosine'.")

        # Generate eta coordinates normal to the surface
        eta_max = self.eta_max
        if self.method == "cheby":
            a = 0.0
            b = eta_max
            eta_buf = np.cos(np.pi*np.arange(self.Ny-1,-1,-1)/(self.Ny-1))
            eta = b*(eta_buf+1)/2.0 + a*(1.0-eta_buf)/2.0
        elif self.method == "cheby2":
            a = 0.0
            b = eta_max
            y_i = 0.2 * eta_max
            eta_buf = np.cos(np.pi*np.arange(self.Ny-1,-1,-1)/(self.Ny-1))
            buf1 = y_i * b / (b - 2 * y_i)
            buf2 = 1 + 2 * buf1 / b
            eta = buf1 * (1 + eta_buf) / (buf2 - eta_buf)
        elif self.method == "geometric":
            r = 1.1 # Growth rate
            eta = eta_max * (1 - r**np.arange(self.Ny)) / (1 - r**self.Ny)
        elif self.method == "uniform":
            eta = np.linspace(0, eta_max, self.Ny)
        else:
            raise ValueError("Invalid eta distribution. Choose cheby, cheby2, geometric, or uniform.")

        X_curv = np.zeros((self.Nx, self.Ny))
        Y_curv = np.zeros((self.Nx, self.Ny))

        try:
            # Interpolate airfoil coordinates and theta to xi points
            x_surface_interp = interp1d(x1, x_surface, kind='linear', bounds_error=False, fill_value='extrapolate')
            y_surface_interp = interp1d(x1, y_surface, kind='linear', bounds_error=False, fill_value='extrapolate')
            theta_interp = interp1d(x1, theta, kind='linear', bounds_error=False, fill_value='extrapolate')
        except ValueError as e:
            print_rz(f"Interpolation error: {e}")
            print_rz(f"x1 shape: {x1.shape}, x_surface shape: {x_surface.shape},  y_surface shape: {y_surface.shape}, theta shape: {theta.shape}")
            raise

        self.theta = theta_interp(xi)
        dtheta_dxi = np.gradient(theta_interp(xi), xi, edge_order=2)
        K1 = -1.0 * dtheta_dxi

        plt.figure(figsize=(6,3),dpi=200)
        plt.plot(xi, self.theta, '-o', markeredgecolor='k', color=colors[0])
        plt.xlabel(r'$\xi$')
        plt.ylabel(r'$\theta$', rotation=0, labelpad=15)
        plt.tight_layout()
        plt.savefig('theta.png')

        plt.figure(figsize=(6,3),dpi=200)
        plt.plot(xi, dtheta_dxi, '-o', markeredgecolor='k', color=colors[0])
        plt.xlabel(r'$\xi$')
        plt.ylabel(r'$\frac{d \theta}{d \xi}$', rotation=0, labelpad=15)
        plt.tight_layout()
        plt.savefig('theta_gradient.png')

        for i, xi_val in enumerate(xi):
            x_s = x_surface_interp(xi_val)
            y_s = y_surface_interp(xi_val)
            theta_s = theta_interp(xi_val)
            
            # Calculate h (first Lamé coefficient)
            h = 1 + eta * K1[i]
            
            # Generate grid points
            X_curv[i, :] = x_s - eta * np.sin(theta_s)
            Y_curv[i, :] = y_s + eta * np.cos(theta_s)

        self.physicalX = X_curv
        self.physicalY = Y_curv

        self.xgrid = xi
        self.ygrid = eta

        self.hx = np.diff(xi)

        self.xi = xi
        self.eta = eta

        xi_grid = np.tile(xi[:, np.newaxis], (1, self.Ny))
        eta_grid = np.tile(eta[np.newaxis, :], (self.Nx,1))

        self.xi_grid = xi_grid
        self.eta_grid = eta_grid
        self.x_grid = self.xi_grid
        self.y_grid = self.eta_grid

        # input kappa is a 1d vector
        # need to expand it for consistency reasons
        d2theta_dxi2 = np.gradient(dtheta_dxi, xi)
        kappa = K1
        self.kappa = kappa[:, np.newaxis]

        self.h = 1 + self.kappa * self.eta_grid 
        self.dtheta_dxi = dtheta_dxi[:, np.newaxis]
        self.d2theta_dxi2 = d2theta_dxi2[:, np.newaxis]

    def compute_metrics(self):
        # Calculate metrics of the coordinate transformation

        # Initialize arrays for metric coefficients
        num_xi  = self.Nx
        num_eta = self.Ny

        self.x_xi  = np.zeros((num_xi, num_eta))
        self.x_eta = np.zeros((num_xi, num_eta))
        self.y_xi  = np.zeros((num_xi, num_eta))
        self.y_eta = np.zeros((num_xi, num_eta))

        self.xi_x  = np.zeros((num_xi, num_eta))
        self.xi_y  = np.zeros((num_xi, num_eta))
        self.eta_x = np.zeros((num_xi, num_eta))
        self.eta_y = np.zeros((num_xi, num_eta))

        for i in range(num_xi):
            h                = self.h[i,:]

            self.x_xi[i, :]  = h * np.cos(self.theta[i])
            self.x_eta[i, :] = -np.sin(self.theta[i])
            self.y_xi[i, :]  = h * np.sin(self.theta[i])
            self.y_eta[i, :] = np.cos(self.theta[i])

            self.xi_x[i, :]  = np.cos(self.theta[i]) / h
            self.xi_y[i, :]  = np.sin(self.theta[i]) / h
            self.eta_x[i, :] = -1.0 * np.sin(self.theta[i])
            self.eta_y[i, :] = np.cos(self.theta[i])

        self.J11 = self.xi_x
        self.J12 = self.xi_y
        self.J21 = self.eta_x
        self.J22 = self.eta_y

        # Calculate Jacobian
        self.J = self.x_xi * self.y_eta - self.x_eta * self.y_xi

        # Compute higher order metric terms

        self.dJ11_dxi = np.gradient(self.J11, self.xi, axis=0, edge_order=2)
        self.dJ11_deta = np.gradient(self.J11, self.eta, axis=1, edge_order=2)

        self.dJ12_dxi = np.gradient(self.J12, self.xi, axis=0, edge_order=2)
        self.dJ12_deta = np.gradient(self.J12, self.eta, axis=1, edge_order=2)
        
        self.dJ21_dxi = np.gradient(self.J21, self.xi, axis=0, edge_order=2)
        self.dJ21_deta = np.gradient(self.J21, self.eta, axis=1, edge_order=2)

        self.dJ22_dxi = np.gradient(self.J22, self.xi, axis=0, edge_order=2)
        self.dJ22_deta = np.gradient(self.J22, self.eta, axis=1, edge_order=2)

    def transform_velocities(self):
        """
        This function transform Cartesian velocities into xi and eta aligned velocties. It interpolates onto a reduced grid if necessary. 
        """
        print_rz(f"U Cartesian Shape: {self.u_field.shape}")
        print_rz(f"U Curvilinear Shape: {self.physicalX.shape}")
        if self.u_field.shape != self.physicalX.shape:
            # print_rz(f"These shapes do not match. Interpolation required.")
            points = np.column_stack((self.x_field.flatten(), self.y_field.flatten()))
            utemp = griddata(points, self.u_field.flatten(), (self.physicalX, self.physicalY), method='linear')
            vtemp = griddata(points, self.v_field.flatten(), (self.physicalX, self.physicalY), method='linear')
            ptemp = griddata(points, self.p_field.flatten(), (self.physicalX, self.physicalY), method='linear')
        else:
            print_rz("Interpolating")
            from scipy.interpolate import Rbf
            points = np.column_stack((self.x_field.flatten(), self.y_field.flatten()))
            # rbf_interp_u = Rbf(points[:,0], points[:,1], self.u_field.flatten(), function='thin_plate', smooth=0.1)
            # rbf_interp_v = Rbf(points[:,0], points[:,1], self.v_field.flatten(), function='thin_plate', smooth=0.1)
            # rbf_interp_p = Rbf(points[:,0], points[:,1], self.p_field.flatten(), function='thin_plate', smooth=0.1)
            # utemp = rbf_interp_u(self.physicalX, self.physicalY)
            # vtemp = rbf_interp_v(self.physicalX, self.physicalY)
            # ptemp = rbf_interp_p(self.physicalX, self.physicalY)
            #
            utemp = griddata(points, self.u_field.flatten(), (self.physicalX, self.physicalY), method='linear')
            vtemp = griddata(points, self.v_field.flatten(), (self.physicalX, self.physicalY), method='linear')
            ptemp = griddata(points, self.p_field.flatten(), (self.physicalX, self.physicalY), method='linear')

        self.compute_metrics()

        U_xi = self.xi_x * utemp + self.xi_y * vtemp
        U_eta = self.eta_x * utemp + self.eta_y * vtemp

        self.u_grid = U_xi
        self.v_grid = U_eta
        self.p_grid = ptemp

    def normalize_fields(self):

        # first compute l0 
        # use physical nu 
        nu = 3.75e-06

        Uinf = np.max(self.u_grid[0,:])
        # Uinf = 15.0 # global Uinf
        l0 = np.sqrt(nu * self.x_field[0,0] / Uinf)
        # try l0 with sqrt(2) in the numerator
        # l0 = np.sqrt(2 * nu * self.x_field[0,0] / Uinf)
        Re = Uinf * l0 / nu
        nu_nondim = 1.0 / Re
        # Nondimensionalizing the flow 
        print_rz(f"Nondimensionalizing the flow field...")
        print_rz(f"l0 = {l0}")
        print_rz(f"U_inf = {Uinf}")
        print_rz(f"X0 = {self.x_field[0,0]}")
        print_rz(f"Re = {Re}")
        print_rz(f"nu nondim = {nu_nondim}")

        f = 1860 # dimensional frequency in Hz
        omega_nondim = 2 * np.pi * f * Re * nu / Uinf**2
        F = omega_nondim / Re * 1e6
        print_rz(f"omega nondim = {omega_nondim}")
        print_rz(f"F = {F}")

        self.u_grid /= Uinf
        self.v_grid /= Uinf

        self.xi_grid /= l0 
        self.eta_grid /= l0
        self.xi /= l0 
        self.eta /= l0 
        # uncommenting line below ruins the computation of d99...
        # is it cause ygrid references eta?
        self.xgrid = self.xi
        self.ygrid = self.eta

        self.dtheta_dxi *= l0
        self.d2theta_dxi2 *= l0**2
        self.hx = np.diff(self.xi)

        self.dJ11_dxi *= l0
        self.dJ11_deta *= l0

        self.dJ12_dxi *= l0
        self.dJ12_deta *= l0
        
        self.dJ21_dxi *= l0
        self.dJ21_deta *= l0

        self.dJ22_dxi *= l0
        self.dJ22_deta *= l0


    def check_gcl_static_2d(self, tolerance=1e-10):
        """
        Check if the 2D static curvilinear grid satisfies the GCL.
        """

        J = self.J
        y_eta = self.y_eta
        y_xi = self.y_xi
        x_eta = self.x_eta
        x_xi = self.x_xi
        # Check GCL conditions

        #TODO:
        # should compute gradient in eta direction using my FD or Chebyshev operators
        gcl_x = np.gradient(J * y_eta, self.xi, axis=0, edge_order=1) - np.gradient(J * y_xi, self.eta, axis=1, edge_order=1)
        gcl_y = -np.gradient(J * x_eta, self.xi, axis=0, edge_order=1) + np.gradient(J * x_xi, self.eta, axis=1, edge_order=1)

        # gcl_x = np.gradient(J * y_eta, self.xi, axis=0) - self.Dy @ (J * y_xi)
        # gcl_y = -np.gradient(J * x_eta, self.xi, axis=0) + self.Dy @ (J * x_xi)
        
        gcl_satisfied = (np.abs(gcl_x) < tolerance).all() and (np.abs(gcl_y) < tolerance).all()

        max_error = max(np.abs(gcl_x).max(), np.abs(gcl_y).max())
        mean_error = max(np.abs(gcl_x).mean(), np.abs(gcl_y).mean())

        if gcl_satisfied:
            print_rz(f"GCL condition satisfied...")
        else:
            print_rz(f"GCL condition is NOT satisfied!")
            print_rz(f"Max GCL error = {max_error}")
            print_rz(f"Mean GCL error = {mean_error}")

            plt.figure(figsize=(6,3), dpi=200)
            plt.contourf(self.xi_grid, self.eta_grid, np.log10(np.abs(gcl_x)), levels=200, cmap='icefire')
            plt.xlabel(r'$\xi$')
            plt.ylabel(r'$\eta$')
            plt.colorbar()
            plt.title('GCL x')
            plt.tight_layout()
            plt.savefig('gcl_x.png')

            plt.figure(figsize=(6,3), dpi=200)
            plt.contourf(self.xi_grid, self.eta_grid, np.log10(np.abs(gcl_y)), levels=200, cmap='icefire')
            plt.xlabel(r'$\xi$')
            plt.ylabel(r'$\eta$')
            plt.title('GCL y')
            plt.colorbar()
            plt.tight_layout()
            plt.savefig('gcl_y.png')
        
        return gcl_satisfied, max_error
    
class gortlerGrid:
    def __init__(self, x_surface, y_surface, hx , x0, eta_max, N_eta, kappa, method="cheby", need_map=True, Uinf=1.0, nu=1.0):

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

        self.x_surface = np.asarray(x_surface)
        self.y_surface = np.asarray(y_surface)

        self.x0 = x0

        self.x_field = None
        self.y_field = None
        self.u_field = None
        self.v_field = None
        self.p_field = None
        self.Nx = None
        self.Ny = N_eta
        self.eta_max = eta_max
        self.method = method
        self.need_map = need_map
        
        self.xi_grid = None
        self.hx = hx
        self.eta_grid = None
        self.xgrid = None
        self.ygrid = None
        self.u_grid = None
        self.v_grid = None
        self.p_grid = None
        self.kappa = None
        self.kappa_temp = kappa

        self.Uinf = Uinf
        self.nu = nu

        self.generate_curvilinear_grid()
        self.compute_metrics()

        derivativeOps = derivativeOperators(self.ygrid, self.method)
        self.Dy = derivativeOps.Dy
        self.Dyy = derivativeOps.Dyy

    def generate_curvilinear_grid(self):
        """Generate a curvilinear grid based on the input data."""

        from scipy.integrate import cumulative_trapezoid
        x_surface = self.x_surface
        y_surface = self.y_surface
        num_eta = self.Ny

        print_rz(f"Initial shapes: surface_x: {x_surface.shape}, surface_y: {y_surface.shape}")
    
        # Remove any NaN or infinite values
        valid_indices = np.isfinite(x_surface) & np.isfinite(y_surface)
        x_surface = x_surface[valid_indices]
        y_surface = y_surface[valid_indices]
        
        print_rz(f"Shapes after removing NaN/inf: surface_x: {x_surface.shape}, surface_y: {y_surface.shape}")
        
        if len(x_surface) != len(y_surface):
            raise ValueError(f"surface_x and surface_y must have the same length. Current lengths: x={len(x_surface)}, y={len(y_surface)}")
        
        # Calculate theta (angle of tangent to airfoil surface)
        dx = np.gradient(x_surface)
        dy = np.gradient(y_surface)
        theta = np.arctan2(dy, dx)
        
        # Calculate x1 (distance along airfoil surface)
        ds = np.sqrt(dx**2 + dy**2)
        x1 = cumulative_trapezoid(ds, initial=0)  # This ensures x1 has the same length as x_surface and y_surface

        # Calculate number of xi points based on delta_xi
        num_xi = int(np.ceil(x1[-1] / self.hx)) 
        self.Nx = num_xi
        # Generate xi distribution
        xi_distribution = 'uniform'
        if xi_distribution == 'uniform':
            xi = np.linspace(0, (num_xi - 1) * self.hx, num_xi) + self.x0
        elif xi_distribution == 'cosine':
            xi = (num_xi - 1) * self.hx * (1 - np.cos(np.linspace(0, np.pi, num_xi))) / 2  + self.x0
        elif xi_distribution == 'hyperbolic':
            beta = 1.001  # Adjust for clustering near endpoints
            xi = (num_xi - 1) * self.hx * (np.tanh(beta * (np.linspace(0, 1, num_xi) - 0.5)) / np.tanh(beta/2) + 1) / 2 + self.x0
        else:
            raise ValueError("Invalid xi_distribution. Choose 'uniform', 'cosine', or 'hyperbolic'.")

        # Generate eta distribution
        eta_max = self.eta_max
        if self.method == 'uniform':
            eta = np.linspace(0, eta_max, num_eta)
        elif self.method == 'geometric':
            r = 1.1  # Growth rate
            eta = eta_max * (1 - r**np.arange(num_eta)) / (1 - r**num_eta)
        elif self.method == "cheby":
            a = 0.0
            b = eta_max
            eta_buf = np.cos(np.pi*np.arange(self.Ny-1,-1,-1)/(self.Ny-1))
            eta = b*(eta_buf+1)/2.0 + a*(1.0-eta_buf)/2.0
        elif self.method == "cheby2":
            a = 0.0
            b = eta_max
            y_i = 0.2 * eta_max
            eta_buf = np.cos(np.pi*np.arange(self.Ny-1,-1,-1)/(self.Ny-1))
            buf1 = y_i * b / (b - 2 * y_i)
            buf2 = 1 + 2 * buf1 / b
            eta = buf1 * (1 + eta_buf) / (buf2 - eta_buf)
        elif self.method == 'hyperbolic':
            beta = 2.0  # Adjust for clustering near the airfoil surface
            eta = eta_max * np.tanh(beta * np.linspace(0, 1, num_eta)) / np.tanh(beta)
        else:
            raise ValueError("Invalid eta_distribution. Choose 'uniform', 'geometric', 'cheby', 'cheby2' or 'hyperbolic'.")

        X_curv = np.zeros((self.Nx, self.Ny))
        Y_curv = np.zeros((self.Nx, self.Ny))

        try:
            # Interpolate airfoil coordinates and theta to xi points
            x_surface_interp = interp1d(x1, x_surface, kind='linear', bounds_error=False, fill_value='extrapolate')
            y_surface_interp = interp1d(x1, y_surface, kind='linear', bounds_error=False, fill_value='extrapolate')
            theta_interp = interp1d(x1, theta, kind='linear', bounds_error=False, fill_value='extrapolate')
        except ValueError as e:
            print_rz(f"Interpolation error: {e}")
            print_rz(f"x1 shape: {x1.shape}, x_surface shape: {x_surface.shape},  y_surface shape: {y_surface.shape}, theta shape: {theta.shape}")
            raise

        self.theta = theta_interp(xi)
        # Calculate curvature K1
        #HACK:
        # We incur some error in computing the curvature numerically
        # Perhaps better to just enforce the known value...
        dtheta_dxi = np.gradient(theta_interp(xi), xi)
        # dtheta_dxi = np.ones((self.Nx,1)) * self.kappa_temp * -1.0
        # dtheta_dxi = np.squeeze(dtheta_dxi,1)
        K1 = -1.0 * dtheta_dxi
        if self.rank == 0:
            np.save("kappa.npy",K1)

        for i, xi_val in enumerate(xi):
            x_s = x_surface_interp(xi_val)
            y_s = y_surface_interp(xi_val)
            theta_s = theta_interp(xi_val)
            
            # Calculate h (first Lamé coefficient)
            h = 1 + eta * K1[i]
            
            # Generate grid points
            X_curv[i, :] = x_s - eta * np.sin(theta_s)
            Y_curv[i, :] = y_s + eta * np.cos(theta_s)

        self.physicalX = X_curv
        self.physicalY = Y_curv

        self.xgrid = xi
        self.ygrid = eta

        self.hx = np.diff(xi)

        self.xi = xi
        self.eta = eta

        xi_grid = np.tile(xi[:, np.newaxis], (1, self.Ny))
        eta_grid = np.tile(eta[np.newaxis, :], (self.Nx,1))

        self.xi_grid = xi_grid
        self.eta_grid = eta_grid
        self.x_grid = self.xi_grid
        self.y_grid = self.eta_grid

        # input kappa is a 1d vector
        # need to expand it for consistency reasons
        d2theta_dxi2 = np.gradient(dtheta_dxi, xi)
        kappa = K1
        self.kappa = kappa[:, np.newaxis]

        self.h = 1 + self.kappa * self.eta_grid 
        self.theta = theta
        self.dtheta_dxi = dtheta_dxi[:, np.newaxis]
        self.d2theta_dxi2 = d2theta_dxi2[:, np.newaxis]

    def compute_metrics(self):
        # Calculate metrics of the coordinate transformation

        # Initialize arrays for metric coefficients
        num_xi  = self.Nx
        num_eta = self.Ny

        self.x_xi  = np.zeros((num_xi, num_eta))
        self.x_eta = np.zeros((num_xi, num_eta))
        self.y_xi  = np.zeros((num_xi, num_eta))
        self.y_eta = np.zeros((num_xi, num_eta))

        self.xi_x  = np.zeros((num_xi, num_eta))
        self.xi_y  = np.zeros((num_xi, num_eta))
        self.eta_x = np.zeros((num_xi, num_eta))
        self.eta_y = np.zeros((num_xi, num_eta))

        for i in range(num_xi):
            h                = self.h[i,:]

            self.x_xi[i, :]  = h * np.cos(self.theta[i])
            self.x_eta[i, :] = -np.sin(self.theta[i])
            self.y_xi[i, :]  = h * np.sin(self.theta[i])
            self.y_eta[i, :] = np.cos(self.theta[i])

            self.xi_x[i, :]  = np.cos(self.theta[i]) / h
            self.xi_y[i, :]  = np.sin(self.theta[i]) / h
            self.eta_x[i, :] = -1.0 * np.sin(self.theta[i])
            self.eta_y[i, :] = np.cos(self.theta[i])

        # Calculate inverse metrics
        # self.xi_x = self.y_eta / self.J
        # self.xi_y = -self.x_eta / self.J
        # self.eta_x = -self.y_xi / self.J
        # self.eta_y = self.x_xi / self.J

        self.J11 = self.xi_x
        self.J12 = self.xi_y
        self.J21 = self.eta_x
        self.J22 = self.eta_y

        # Calculate Jacobian
        self.J = self.x_xi * self.y_eta - self.x_eta * self.y_xi

        # Compute higher order metric terms
        #NOTE:
        # dJ12_eta dJ11_eta are nonzero in my log file which doesn't make sense to me 
        # Actually, figured it out: it's because of "h" prefactor which indeed varies with eta 
        # the question is if this is correct...

        self.dJ11_dxi = np.gradient(self.J11, self.xi, axis=0)
        self.dJ11_deta = np.gradient(self.J11, self.eta, axis=1)

        self.dJ12_dxi = np.gradient(self.J12, self.xi, axis=0)
        self.dJ12_deta = np.gradient(self.J12, self.eta, axis=1)
        
        self.dJ21_dxi = np.gradient(self.J21, self.xi, axis=0)
        self.dJ21_deta = np.gradient(self.J21, self.eta, axis=1)

        self.dJ22_dxi = np.gradient(self.J22, self.xi, axis=0)
        self.dJ22_deta = np.gradient(self.J22, self.eta, axis=1)
        # Compute higher order metric terms
    


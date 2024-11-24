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

        if self.method == "cheby" or method == "cheby2":
            self.map_D_cheby(self.ygrid, 1, self.need_map)
            self.map_D_cheby(self.ygrid, 2, self.need_map)

        elif self.method == "fd":
            self.Dy = self.set_D_FD(self.ygrid,d=1,order=2, output_full=True, uniform=False)
            self.Dyy = self.set_D_FD(self.ygrid,d=2,order=2, output_full=True, uniform=False)

        else:
            print("Invalid method chosen for derivatives")

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
                self.Dy = np.diag(1./(D@x))@D # d(.)/dy = d(.)/dxi * dxi/dy
            elif order==2: # d^2()/dy^2 = d^2()/dxi^2 (dxi/dy)^2 + d()/dxi d^2xi/dy^2
                D1= self.set_D_cheby(xi, order=1)
                D = D1@D1 # second derivative
                dxdxi = D1@x
                dxidx = 1./dxdxi # element wise invert
                d2xidx2 = -(D@x)*(dxidx)**3
                self.Dyy = (D*(dxidx[:,np.newaxis]**2)) + (D1*d2xidx2[:,np.newaxis])
            else:
                print('Cannot do this order of derivative with non-uniform mesh.  your input order of derivative = ',order)
        else:
            if order == 1:
                self.Dy = set_D_cheby(self, x, order)
            elif order == 2:
                self.Dyy = self.set_D_cheby(self, x, order)


class NACA_GRID:

    def __init__(self, Nx, Ny, nmax, method="cheby", need_map=True):

        self.Nx = int(Nx)
        self.Ny = int(Ny)
        self.nmax = nmax

        self.method = method
        self.need_map = need_map

        self.Dy = None
        self.Dyy = None

        self.x_airfoil = None
        self.y_airfoil = None
        self.theta = None

        self.xgrid = None
        self.ygrid = None

        self.X = None
        self.Y = None

        self.Dy = None
        self.Dyy = None

        self.J   = None
        self.J11 = None
        self.J12 = None
        self.J21 = None
        self.J22 = None
        self.H11 = None
        self.H12 = None
        self.H21 = None
        self.H22 = None

        self.gridDx = None
        self.gridDxx = None
        self.gridDy = None
        self.gridDyy = None

        # compute symmetric airfoil geometry
        # self.generate_airfoil(thickness, self.Nx)

        # compute body-fitted grid
        x_arr = self.make_x_for_airfoil()
        self.make_airfoil_grid_points(self.Ny, x_arr)
        # self.genreate_curvilinear_mesh(Ns, Nn, nmax)

        if self.method == "cheby":
            # take one of the wall normal discretizations as the ygrid needed 
            # to construct the chebyshev differentiation 
            xi = np.cos(np.pi*np.arange(self.Ny-1,-1,-1)/(self.Ny-1))
            ygrid_for_derivatives = self.nmax * (xi + 1) / 2.0
            self.ygrid = ygrid_for_derivatives

            self.map_D_cheby(self.ygrid, 1, self.need_map)
            self.map_D_cheby(self.ygrid, 2, self.need_map)
        else:
            print("Invalid method chosen for derivatives")
            print("Implemented methods are 'cheby'\n")

        # compute mesh jacobian
        self.compute_mesh_jacobian(x_arr)

    def generate_airfoil(self, t, Nx):

        # use cosine spacing for a smooth leading edge
        beta = np.linspace(0, np.pi, Nx)
        
        x = (1 - np.cos(beta)) / 2
        # sharp trailing edge equation
        y = 0.594689181*(0.298222773*np.sqrt(x) - 0.127125232*x - 0.357907906*x**2 + 0.291984971*x**3 - 0.105174606*x**4)
        # y = 5 * t * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1036*x**4)

        # theta = arctan(dy/dx)
        theta = np.zeros_like(x)
        theta[1:] = np.arctan(5 * t * (0.2969*0.5*x[1:]**(-1/2) - 0.1260 - \
            0.3516*2*x[1:] + 0.2843*3*x[1:]**2 - 0.1036*4*x[1:]**3))
        # theta(x = 0) is not well defined using this equation
        theta[0] = np.pi / 2

        self.x_airfoil = x
        self.y_airfoil = y 
        self.theta = theta

        return

    # def genreate_curvilinear_mesh(self,Ns, Nn, nmax):

    #     if Ns > self.Nx:
    #         print(f"Can't have more curvilinnear streamwise coordinates than points on the airfoil surface")
    #         exit()

    #     normal_length = nmax
    #     normals = np.zeros((Ns, 2))

    #     y = np.copy(self.y_airfoil)
    #     x = np.copy(self.x_airfoil)

    #     for ii in range(Ns-1):
    #         normals[ii,:] = [(y[ii] - y[ii+1]), x[ii+1] - x[ii]]
    #         normals[ii,:] *= normal_length / np.linalg.norm(normals[ii,:])

    #     if self.method == "fd":
    #         grid_heights = np.linspace(0, 1, Nn)
    #     elif self.method == "cheby":
    #         a = 0
    #         b = 1
    #         xi = np.cos(np.pi*np.arange(Nn-1,-1,-1)/(Nn-1))
    #         grid_heights = b*(xi+1)/2.0 + a*(1.0-xi)/2.0

    #     pts_x = np.zeros((Nn, Ns))
    #     pts_y = np.zeros((Nn, Ns))
    #     for ii in range(Nn):
    #         for jj in range(Ns-1):
    #             t = grid_heights[ii] # normalized distance along normal
    #             pts_x[ii,jj] = (1 - t)*x[jj] + t * (x[jj] + normals[jj,0])
    #             pts_y[ii,jj] = (1 - t)*y[jj] + t * (y[jj] + normals[jj,1])

    #     self.X = pts_x
    #     self.Y = pts_y

    #     return

    def make_x_for_airfoil(self):

        # naca0012 equations
        y = lambda x: 0.594689181*(0.298222773*np.sqrt(x) - 0.127125232*x - 0.357907906*x**2 + 0.291984971*x**3 - 0.105174606*x**4)
        yprime = lambda x: 0.594689181*(0.298222773*0.5*x**(-0.5) - 0.127125232 - 0.357907906*2*x + 0.291984971*3*x**2 - 0.105174606*4*x**3)

        # first find total arc length of the airfoil 
        x = np.linspace(1e-5,1,1000) # normalized between zero and 1 
        L = np.trapz(np.sqrt(1 + yprime(x)**2), x)

        # theta[1:] = np.arctan(5 * t * (0.2969*0.5*x[1:]**(-1/2) - 0.1260 - \
        #     0.3516*2*x[1:] + 0.2843*3*x[1:]**2 - 0.1036*4*x[1:]**3))
        # theta(x = 0) is not well defined using this equation

        self.x_airfoil = x
        self.y_airfoil = y(x)

        Nxi = self.Nx
        delta_xi = L / Nxi
        
        epsilon = 1e-4 # tolerance
        delta_x = 0.00001 # spacing in x used to find desired delta_xi 

        x_arr = np.ones(1) * 1e-5
        dxi_arr = np.zeros(1)

        iteration = 1 # initialize iteration variable 
        sub_iterations = True # use subiterations

        for segment in range(Nxi):
    
            count = 1
            count_sub = 1
            
            while True:
                
                if segment == 0:
                    a = x_arr[0]
                else:
                    a = np.copy(b)
                    
                x_integral = np.linspace(a, a + delta_x * count, 200)

                dL = np.trapz(np.sqrt(1 + yprime(x_integral)**2), x_integral)
                
                if np.abs(dL - delta_xi) < epsilon:
                    
                    if sub_iterations:
                        # print("In subiterations")
                        # print(f"initial dL = {dL}")
                        sub_delta_x = delta_x / 2000
                        left_ = a + delta_x * (count - 1)
                        
                        while True:
                            
                            right_ = left_ + sub_delta_x * count_sub

                            x_integral = np.linspace(a, right_, 100)
                            dL = np.trapz(np.sqrt(1 + yprime(x_integral)**2), x_integral)
                            if np.abs(dL - delta_xi) < epsilon / 1:
                                b = right_
                                x_arr = np.append(x_arr, right_)
                                dxi_arr = np.append(dxi_arr, dL)
                                break

                            elif count_sub > 6000 or right_ > 1.0:
                                print("Broke in subiterations")
                                print(f"count_sub = {count_sub}")
                                print(f"right_ = {right_}")
                                b = a + delta_x * count
                                x_arr = np.append(x_arr, b)
                                x_integral = np.linspace(a, a + delta_x * count, 100)
                                dL = np.trapz(np.sqrt(1 + yprime(x_integral)**2), x_integral)
                                dxi_arr = np.append(dxi_arr, dL)
                                break

                            count_sub += 1
                        break
                        
                    else:
                        b = a + delta_x * count
                        x_arr = np.append(x_arr,b)
                        dxi_arr = np.append(dxi_arr, dL)
                        # print(f"dL = {dL}")
                        # print(f"count = {count}")
                        break
                        
                elif count > 200000 or delta_x * count > 1.0:
                    print("Didn't find x for desired delta xi")
                    print(f"Checked up to x = {delta_x * count}")
                    print(f"dL = {dL}")
                    break
                count += 1

        # theta = arctan(dy/dx)
        theta = np.zeros_like(x_arr)
        theta = np.arctan(yprime(x_arr[1:]))
        theta[0] = np.pi / 2
        self.theta = theta
        return x_arr 

    def make_airfoil_grid_points(self, nlayers, x_arr):

        y = lambda x: 0.594689181*(0.298222773*np.sqrt(x) - 0.127125232*x - 0.357907906*x**2 + 0.291984971*x**3 - 0.105174606*x**4)

        normal_length = self.nmax
        normals = np.zeros((x_arr.size, 2))
        for ii in range(0,x_arr.size - 1):
            normals[ii,:] = [(y(x_arr[ii]) - y(x_arr[ii+1])), x_arr[ii+1] - x_arr[ii]]
            normals[ii,:] *= normal_length / np.linalg.norm(normals[ii,:])
        
        grid_heights = np.linspace(0, 1,nlayers)
        pts = np.zeros((grid_heights.size, normals.shape[0], 2))
        pts_x = np.zeros((grid_heights.size, normals.shape[0]))
        pts_y = np.zeros((grid_heights.size, normals.shape[0]))

        xi = np.cos(np.pi*np.arange(nlayers-1,-1,-1)/(nlayers-1))

        for ii in range(0,grid_heights.size):
            for jj in range(0,normals.shape[0]):

                # uniform discretiation along normal direction
                # t = grid_heights[ii]
                # pts[ii,jj,:] = np.asarray([(1 - t)*x_arr[jj] + t * (x_arr[jj]+normals[jj,0]), (1 - t)*y(x_arr[jj]) + t * (y(x_arr[jj]) + normals[jj,1])])
                # pts_x[ii,jj] = (1 - t)*x_arr[jj] + t * (x_arr[jj] + normals[jj,0])
                # pts_y[ii,jj] = (1 - t)*y(x_arr[jj]) + t * (y(x_arr[jj]) + normals[jj,1])

                # chebyshev discretization along normal direction
                t = 1*(xi[ii]+1)/2.0 #+ a*(1.0-xi)/2.0
                pts_x[ii,jj] = (1 - t)*x_arr[jj] + t * (x_arr[jj] + normals[jj,0])
                pts_y[ii,jj] = (1 - t)*y(x_arr[jj]) + t * (y(x_arr[jj]) + normals[jj,1])

        self.X = pts_x
        self.Y = pts_y

        return     

    def compute_mesh_jacobian(self, x_arr):

        theta = np.copy(self.theta)
        # x = np.copy(self.x_airfoil)
        x = np.copy(x_arr)

        # set buf[0] to zero because it is otherwise undefined
        buf = 1 / np.cos(theta)
        buf[0] = 0.
        xi = integrate.cumtrapz(buf,x,initial=0) # body fitted streamwise coordinate
        self.xgrid = xi

        # K1 = -del(theta)/del(x1)
        # need to be careful about the discretization error here
        K1 = -1 * np.gradient(theta, xi)

        # first Lame parameter
        h = 1 + self.Y * K1

        c11 = np.cos(theta) / h
        c12 = np.sin(theta) / h

        c21 = -1 * np.sin(theta) / np.ones_like(h)
        c22 = np.cos(theta) / np.ones_like(h)

        self.J11 = c11
        self.J12 = c12
        self.J21 = c21
        self.J22 = c22

        # print(f"c11 shape = {c11.shape}") # shape is Nn x Nx 
        # print(f"c12 shape = {c12.shape}")
        # print(f"c21 shape = {c21.shape}")
        # print(f"c22 shape = {c22.shape}")

        self.H11 = np.gradient(c11, self.xgrid, axis = 1)
        self.H12 = self.Dy @ c12
        self.H21 = np.gradient(c21, self.xgrid, axis = 1)
        self.H22 = self.Dy @ c22

        J = np.block([[c11, c12],
                      [c21, c22]])

        self.J = J

        return


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
                self.Dy = np.diag(1./(D@x))@D # d(.)/dy = d(.)/dxi * dxi/dy
            elif order==2: # d^2()/dy^2 = d^2()/dxi^2 (dxi/dy)^2 + d()/dxi d^2xi/dy^2
                D1= self.set_D_cheby(xi, order=1)
                D = D1@D1 # second derivative
                dxdxi = D1@x
                dxidx = 1./dxdxi # element wise invert
                d2xidx2 = -(D@x)*(dxidx)**3
                self.Dyy = (D*(dxidx[:,np.newaxis]**2)) + (D1*d2xidx2[:,np.newaxis])
            else:
                print('Cannot do this order of derivative with non-uniform mesh.  your input order of derivative = ',order)
        else:
            if order == 1:
                self.Dy = set_D_cheby(self, x, order)
            elif order == 2:
                self.Dyy = self.set_D_cheby(self, x, order)

    def form_mapped_operators(self):

        Dy  = np.copy(self.Dy)
        Dyy = np.copy(self.Dyy)

        # only backward euler implemented currently



class curvedGrid:

    def __init__(self, Nx, Ny, hx, ymax, R, method="cheby", need_map=True):


        self.Nx = int(Nx)
        self.Ny = int(Ny)
        self.hx = hx
        self.nmax = ymax

        self.method = method
        self.need_map = need_map

        self.Dy = None
        self.Dyy = None

        self.R = R # radius of curvature
        self.kappa = -1 / self.R
        self.theta = None

        self.xi_grid = None
        self.xgrid = None
        self.ygrid = None

        self.X = None
        self.Y = None

        self.Dy = None
        self.Dyy = None

        self.J   = None
        self.J11 = None
        self.J12 = None
        self.J21 = None
        self.J22 = None
        self.h = None 
        self.dJ11_dx = None 
        self.dJ11_dy = None 
        self.dJ12_dx = None 
        self.dJ12_dy = None 
        self.dJ21_dx = None 
        self.dJ21_dy = None 
        self.dJ22_dx = None
        self.dJ22_dy = None 

    def make_grid(self):

        x_arr = self.make_x_for_curved_plate()
        self.make_curved_plate_grid_points(self.Ny, x_arr)
        self.init_ygrid()

        # compute the mesh jacobian
        self.compute_mesh_jacobian()

    def init_ygrid(self):

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if self.method == "cheby":
            # take one of the wall normal discretizations as the ygrid needed 
            # to construct the chebyshev differentiation 
            xi = np.cos(np.pi*np.arange(self.Ny-1,-1,-1)/(self.Ny-1))
            ygrid_for_derivatives = self.nmax * (xi + 1) / 2.0
            self.ygrid = ygrid_for_derivatives

            self.map_D_cheby(self.ygrid, 1, self.need_map)
            self.map_D_cheby(self.ygrid, 2, self.need_map)
        else:
            print_rz("Invalid method chosen for derivatives")
            print_rz("Implemented methods are 'cheby'\n")


    def make_x_for_curved_plate(self):

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        
        # equation for half circle 
        y = lambda x: -np.sqrt(self.R**2 - x**2) + self.R 
        yprime = lambda x: x / np.sqrt(self.R**2 - x**2)

        Nxi = self.Nx
        # delta_xi = L / Nxi
        delta_xi = self.hx

        xi_total = Nxi * delta_xi 

        x0 = 181.6590

        # first find total arc length of the curved plate
        x = np.linspace(x0, x0 + xi_total*2, 1000) # approximation for x array
        L = np.trapz(np.sqrt(1 + yprime(x)**2), x)

        print_rz(f"L = {L}")
        print_rz(f"delta_xi = {delta_xi}")
        print_rz(f"xi_total = {xi_total}")

        epsilon = 1e-1 # tolerance
        # delta_x = 0.00001 # spacing in x used to find desired delta_xi 
        delta_x = delta_xi / 100

        x_arr = np.ones(1) * x0
        dxi_arr = np.zeros(1)

        iteration = 1 # initialize iteration variable 
        sub_iterations = True # use subiterations

        print_rz(f"Number of segments to generate = {Nxi}")
        print_rz("Generating mesh...\n")

        for segment in range(Nxi):

            if segment % 50 == 0:
                print_rz(f"Mesh segment = {segment}")

            count = 1
            count_sub = 1

            while True:

                if segment == 0:
                    a = x_arr[0]
                else:
                    a = np.copy(b)

                x_integral = np.linspace(a, a + delta_x * count, 1000)

                dL = np.trapz(np.sqrt(1 + yprime(x_integral)**2), x_integral)
                err = np.abs(dL - delta_xi) / delta_xi

                if dL > 1.2*delta_xi:
                    print_rz(f"Mesh generation failed")
                    print_rz(f"segment = {segment}")
                    print_rz(f"dL > delta_xi")
                    print_rz(f"dL = {dL}")
                    print_rz(f"delta_xi = {delta_xi}")
                    print_rz(f"err = {err}")
                    break

                if err < epsilon:

                    if sub_iterations:
                        # print("In subiterations")
                        # print(f"initial dL = {dL}")
                        sub_delta_x = delta_x / 100
                        left_ = a + delta_x * (count - 1)

                        while True:

                            right_ = left_ + sub_delta_x * count_sub

                            x_integral = np.linspace(a, right_, 100)
                            dL = np.trapz(np.sqrt(1 + yprime(x_integral)**2), x_integral)
                            err = np.abs(dL - delta_xi) / delta_xi
                            if err < epsilon / 10:
                                b = right_
                                x_arr = np.append(x_arr, right_)
                                dxi_arr = np.append(dxi_arr, dL)
                                break

                            elif count_sub > 20000 or right_ > L:
                                print_rz("Broke in subiterations")
                                print_rz(f"count_sub = {count_sub}")
                                print_rz(f"right_ = {right_}")
                                b = a + delta_x * count
                                x_arr = np.append(x_arr, b)
                                x_integral = np.linspace(a, a + delta_x * count, 100)
                                dL = np.trapz(np.sqrt(1 + yprime(x_integral)**2), x_integral)
                                dxi_arr = np.append(dxi_arr, dL)
                                break

                            count_sub += 1
                        break

                    else:
                        b = a + delta_x * count
                        x_arr = np.append(x_arr,b)
                        dxi_arr = np.append(dxi_arr, dL)
                        break

                elif count > 200000 or delta_x * count > L:
                    print_rz("Didn't find x for desired delta xi")
                    print_rz(f"Checked up to x = {delta_x * count}")
                    print_rz(f"dL = {dL}")
                    print_rz(f"count = {count}")
                    break
                count += 1

        print_rz("FROG")
        print_rz(x_arr)
        return x_arr


    def make_curved_plate_grid_points(self, nlayers, x_arr):

        y = lambda x: -np.sqrt(self.R**2 - x**2) + self.R
        yprime = lambda x: x / np.sqrt(self.R**2 - x**2)

        normal_length = self.nmax
        normals = np.zeros((x_arr.size - 1, 2))
        for ii in range(0,x_arr.size - 1):
            normals[ii,:] = [(y(x_arr[ii]) - y(x_arr[ii+1])), x_arr[ii+1] - x_arr[ii]]
            normals[ii,:] *= normal_length / np.linalg.norm(normals[ii,:])
        
        grid_heights = np.linspace(0, 1,nlayers)
        pts = np.zeros((grid_heights.size, normals.shape[0], 2))
        pts_x = np.zeros((grid_heights.size, normals.shape[0]))
        pts_y = np.zeros((grid_heights.size, normals.shape[0]))

        xi = np.cos(np.pi*np.arange(nlayers-1,-1,-1)/(nlayers-1))

        for ii in range(0,grid_heights.size):
            for jj in range(0,normals.shape[0]):

                # uniform discretiation along normal direction
                # t = grid_heights[ii]
                # pts[ii,jj,:] = np.asarray([(1 - t)*x_arr[jj] + t * (x_arr[jj]+normals[jj,0]), (1 - t)*y(x_arr[jj]) + t * (y(x_arr[jj]) + normals[jj,1])])
                # pts_x[ii,jj] = (1 - t)*x_arr[jj] + t * (x_arr[jj] + normals[jj,0])
                # pts_y[ii,jj] = (1 - t)*y(x_arr[jj]) + t * (y(x_arr[jj]) + normals[jj,1])

                # chebyshev discretization along normal direction
                t = 1*(xi[ii]+1)/2.0 #+ a*(1.0-xi)/2.0
                pts_x[ii,jj] = (1 - t)*x_arr[jj] + t * (x_arr[jj] + normals[jj,0])
                pts_y[ii,jj] = (1 - t)*y(x_arr[jj]) + t * (y(x_arr[jj]) + normals[jj,1])

        # theta = arctan(dy/dx)
        theta = np.zeros_like(pts_x[0,:])
        theta = np.arctan(yprime(pts_x[0,:]))
        self.theta = theta

        self.X = pts_x
        self.Y = pts_y

        return

    def compute_mesh_jacobian(self):

        theta = np.copy(self.theta)
        yprime = lambda x: x / np.sqrt(self.R**2 - x**2)

        x = np.copy(self.X[0,:])
        self.xgrid = x

        # FOR DEBUGGING
        # x0 = 36.997764
        # x = np.linspace(x0, 100,50)
        # self.xgrid = x

        # xi = np.zeros_like(x)
        # for ii in range(np.size(x)):
        #     xbuf = np.linspace(0,x[ii],1000)
        #     theta_arr = np.arctan(yprime(xbuf))
        #     integrand = 1 / np.cos(theta_arr)
        #     xi[ii] = np.trapz(integrand,xbuf)


        xi = integrate.cumtrapz(1 / np.cos(theta), x, initial = 0)

        # buf = 1 / np.cos(theta)
        # xi = integrate.cumtrapz(buf,x,initial=0) # body fitted streamwise coordinate
        self.xi_grid = xi

        # K1 = -del(theta)/del(x1)
        # need to be careful about the discretization error here
        K1 = -1 * np.gradient(theta, xi)

        # first Lame parameter
        h = 1 + self.Y * K1

        c11 = np.cos(theta) / h
        c12 = np.sin(theta) / h

        c21 = -1 * np.sin(theta) / np.ones_like(h)
        c22 = np.cos(theta) / np.ones_like(h)

        self.J11 = c11
        self.J12 = c12
        self.J21 = c21
        self.J22 = c22

        # self.H11 = np.gradient(c11, self.xi_grid, axis = 1)
        # self.H12 = self.Dy @ c12
        # self.H21 = np.gradient(c21, self.xi_grid, axis = 1)
        # self.H22 = self.Dy @ c22

        self.dJ11_dx = np.gradient(c11, xi, axis = 1)
        self.dJ12_dx = np.gradient(c12, xi, axis=1)
        self.dJ11_dy = self.Dy @ c11 
        self.dJ12_dy = self.Dy @ c12
        self.dJ21_dx = np.gradient(c21, xi, axis = 1)
        self.dJ21_dy = self.Dy @ c21
        self.dJ22_dx = np.gradient(c22, xi, axis = 1)
        self.dJ22_dy = self.Dy @ c22


        self.h = h 

        J = np.block([[c11, c12],
                      [c21, c22]])

        self.J = J

        return

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
                self.Dy = np.diag(1./(D@x))@D # d(.)/dy = d(.)/dxi * dxi/dy
            elif order==2: # d^2()/dy^2 = d^2()/dxi^2 (dxi/dy)^2 + d()/dxi d^2xi/dy^2
                D1= self.set_D_cheby(xi, order=1)
                D = D1@D1 # second derivative
                dxdxi = D1@x
                dxidx = 1./dxdxi # element wise invert
                d2xidx2 = -(D@x)*(dxidx)**3
                self.Dyy = (D*(dxidx[:,np.newaxis]**2)) + (D1*d2xidx2[:,np.newaxis])
            else:
                print('Cannot do this order of derivative with non-uniform mesh.  your input order of derivative = ',order)
        else:
            if order == 1:
                self.Dy = set_D_cheby(self, x, order)
            elif order == 2:
                self.Dyy = self.set_D_cheby(self, x, order)



class NLF_Grid:

    def __init__(self, Nx, Ny, ymax, method="cheby", need_map=True):
        self.Nx = Nx
        self.Ny = Ny
        self.hx = None 
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

        # h in general is a grid metric term that does NOT need to be 1
        # for the simple cases one would want to simulate using the "flatplate" 
        # initialization, h = 1 should suffice
        self.h = np.ones((1,Nx))

        self.theta = np.zeros((1,Nx))
        self.xi_grid = self.xgrid

        self.set_yGrid(self.method, 0.0, self.ymax, self.Ny)
        self.set_xGrid() 

        # Jacobians for flat plate 
        self.J11 = np.ones((Ny, Nx))
        self.J22 = np.ones((Ny, Nx))
        self.J12 = np.zeros((Ny, Nx))
        self.J21 = np.zeros((Ny, Nx))

        self.H11 = np.zeros((Ny, Nx))
        self.H12 = np.zeros((Ny, Nx))
        self.H21 = np.zeros((Ny, Nx))
        self.H22 = np.zeros((Ny, Nx))

        self.dJ11_dx = np.zeros((Ny,Nx))
        self.dJ12_dx = np.zeros((Ny,Nx))
        self.dJ11_dy = np.zeros((Ny,Nx))
        self.dJ12_dy = np.zeros((Ny,Nx))
        self.dJ21_dx = np.zeros((Ny,Nx))
        self.dJ21_dy = np.zeros((Ny,Nx))
        self.dJ22_dx = np.zeros((Ny,Nx))
        self.dJ22_dy = np.zeros((Ny,Nx))

        self.h = np.ones((Ny,Nx))

        if self.method == "cheby" or method == "cheby2":
            self.map_D_cheby(self.ygrid, 1, self.need_map)
            self.map_D_cheby(self.ygrid, 2, self.need_map)

        elif self.method == "fd":
            self.Dy = self.set_D_FD(self.ygrid,d=1,order=2, output_full=True, uniform=False)
            self.Dyy = self.set_D_FD(self.ygrid,d=2,order=2, output_full=True, uniform=False)

        else:
            print("Invalid method chosen for derivatives")

    def set_yGrid(self, method, a, b, ny, delta=5.0):

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

    def airfoil_interpolator(self):

        airfoil_points = np.loadtxt('/home/moin/cagonzal/glimPSE/glimPSE_src/parallel_src/nlf_coordinates.dat')
        xupper = airfoil_points[0:33, 0]
        yupper = airfoil_points[0:33, 1]

        from scipy.interpolate import Akima1DInterpolator
        from scipy import integrate

        # create interpolating function using makima interpolation
        # can read python documentation about how this works but my testing has yielded good results 

        mak = Akima1DInterpolator(xupper, yupper, method = 'makima')

        # define high resolution xgrid that will be used to compute xi 
        xfine = np.linspace(0.0,1.0,5000)
        yfine = mak(xfine)

        dyudx = np.gradient(yfine, xfine)

        # compute the arc length of the airfoil 
        Lu = np.trapz(np.sqrt(1.0 + dyudx**2), xfine)

        dxi_u = Lu / self.Nx

        # now compute theta 
        theta_u = np.arctan2(yfine, xfine)

        # with theta in hand, we can compute a xi array using simple geometry

        xi_u = integrate.cumulative_trapezoid(1.0 / np.cos(theta_u), x = xfine, initial = 0.0)

        # interpolating funnction that maps xi to x

        xu_interp_fun = Akima1DInterpolator(xi_u, xfine, method = 'makima')

        return Lu, xu_interp_fun, mak


    def set_xGrid(self):

        Lu, xu_interp_fun, y_interp_fun = self.airfoil_interpolator()

        self.xgrid = np.linspace(0.0, Lu, self.Nx) # this is the uniform xi grid

        # use interpolating function to map xi to x 

        x = xu_interp_fun(self.xgrid)
        y = y_interp_fun(x)

        theta = np.arctan2(y,x)

        # now compute the gradient of theta w.r.t xi (required to get curvature)

        dthetadxi = np.gradient(theta, self.xgrid)
        kappa = -1.0 * dthetadxi
        kappa = kappa[:, np.newaxis]

        self.xi_grid = self.xgrid
        self.kappa = kappa
        self.hx = self.xi_grid[1] - self.xi_grid[0]

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
                self.Dy = np.diag(1./(D@x))@D # d(.)/dy = d(.)/dxi * dxi/dy
            elif order==2: # d^2()/dy^2 = d^2()/dxi^2 (dxi/dy)^2 + d()/dxi d^2xi/dy^2
                D1= self.set_D_cheby(xi, order=1)
                D = D1@D1 # second derivative
                dxdxi = D1@x
                dxidx = 1./dxdxi # element wise invert
                d2xidx2 = -(D@x)*(dxidx)**3
                self.Dyy = (D*(dxidx[:,np.newaxis]**2)) + (D1*d2xidx2[:,np.newaxis])
            else:
                print('Cannot do this order of derivative with non-uniform mesh.  your input order of derivative = ',order)
        else:
            if order == 1:
                self.Dy = set_D_cheby(self, x, order)
            elif order == 2:
                self.Dyy = self.set_D_cheby(self, x, order)

class curvedSurfaceGrid:
    def __init__(self, x_surface, y_surface, x_field, y_field, u_field, v_field, p_field, Nx, Ny, ymax, method="cheby", need_map=True):

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

        self.x_surface = x_surface
        self.y_surface = y_surface
        self.x_field = x_field
        self.y_field = y_field
        self.u_field = u_field
        self.v_field = v_field
        self.p_field = p_field
        self.Nx = Nx
        self.Ny = Ny
        self.method = method
        self.need_map = need_map
        
        self.xi_grid = None
        self.hx = None
        self.eta_grid = None
        self.x_grid = None
        self.y_grid = None
        self.u_grid = None
        self.v_grid = None
        self.p_grid = None
        self.kappa = None
        self.theta = None
        
        self.generate_curvilinear_grid()
        self.interpolate_velocity_field()
        self.calculate_metrics()
        self.transform_velocities()

        if self.method == "cheby" or method == "cheby2":
            self.map_D_cheby(self.ygrid, 1, self.need_map)
            self.map_D_cheby(self.ygrid, 2, self.need_map)

        elif self.method == "fd":
            self.Dy = self.set_D_FD(self.ygrid,d=1,order=2, output_full=True, uniform=False)
            self.Dyy = self.set_D_FD(self.ygrid,d=2,order=2, output_full=True, uniform=False)

        else:
            print("Invalid method chosen for derivatives")


    def generate_curvilinear_grid(self):
        # Generate xi coordinates along the surface
        arc_length = np.cumsum(np.sqrt(np.diff(self.x_surface)**2 + np.diff(self.y_surface)**2))
        arc_length = np.insert(arc_length, 0, 0)

        # this is the uniform xi grid on which we want to 
        # solve the PSE equations
        xi = np.linspace(0, 1, self.Nx)
        self.hx = xi[1] - xi[0]

        # Interpolate surface coordinates
        x_interp = interp1d(arc_length / arc_length[-1], self.x_surface)
        y_interp = interp1d(arc_length / arc_length[-1], self.y_surface)

        x_surface_new = x_interp(xi)
        y_surface_new = y_interp(xi)

        # Generate eta coordinates normal to the surface
        if self.method == "cheby":
            a = 0.0
            b = 0.025
            eta_buf = np.cos(np.pi*np.arange(self.Ny-1,-1,-1)/(self.Ny-1))
            eta = b*(eta_buf+1)/2.0 + a*(1.0-eta_buf)/2.0

        if self.method == "cheby2":
            a = 0.0
            b = 1.0
            y_i = 0.2
            eta_buf = np.cos(np.pi*np.arange(self.Ny-1,-1,-1)/(self.Ny-1))
            buf1 = y_i * b / (b - 2 * y_i)
            buf2 = 1 + 2 * buf1 / b
            eta = buf1 * (1 + eta_buf) / (buf2 - eta_buf)

        elif self.method == "fd":
            a = 0.0
            b = 1.0
            eta = np.linspace(a, b, self.Ny)
            eta = b * (1. + (np.tanh(delta * (y / b - 1.)) / np.tanh(delta)))

        # Calculate normal vectors
        dx = np.gradient(x_surface_new)
        dy = np.gradient(y_surface_new)
        normal_x = -dy 
        normal_y = dx

        # Normalize normal vectors
        norm = np.sqrt(normal_x**2 + normal_y**2)
        normal_x /= norm
        normal_y /= norm

        # Generate grid points (this is the physical grid that corresponds to my computational grid)
        self.x_grid = np.outer(x_surface_new, np.ones(self.Ny)) + np.outer(normal_x, eta) * np.max(arc_length)
        self.y_grid = np.outer(y_surface_new, np.ones(self.Ny)) + np.outer(normal_y, eta) * np.max(arc_length)

        # DEBUGGING
        # want to plot x_grid and y_grid
        plt.figure(figsize=(6,3),dpi=200)
        plt.plot(self.x_grid, self.y_grid, ".k")
        plt.savefig('computational_mesh.png', bbox_inches=None)
        # Generate 2D xi and eta grids used for interpolating velocities
        self.xi_grid, self.eta_grid = np.meshgrid(xi, eta)
        self.xi_grid = self.xi_grid.T
        self.eta_grid = self.eta_grid.T
        # Generate 1D xi and eta grids used by the solver
        self.xgrid = xi
        self.ygrid = eta

        # now compute kappa 
        theta = np.arctan2(y_surface_new, x_surface_new)
        dthetadxi =np.gradient(theta, xi)
        kappa = -1.0 * dthetadxi
        # kappa = np.expand_dims(kappa,0)
        kappa = kappa[:, np.newaxis]
        self.theta = theta
        self.kappa = kappa
        self.dtheta_dxi = dtheta_dxi
        self.dtheta_dxi_dxi = np.gradient(dtheta_dxi, xi)
        self.h = 1 + self.kappa * self.eta_grid # need a transpose to properly multiply these

    def interpolate_velocity_field(self):

        plt.figure(figsize=(6,3),dpi=200)
        plt.plot(self.x_field, self.y_field, ".k")
        plt.ylim([0.0, 0.15])
        plt.xlim([-0.1, 1.1])
        plt.savefig('input_mesh.png', bbox_inches=None)

        #TODO:
        # Need to do some debugging to figure out what's going on with my interpolation 
        # let's find the minimum x in the RANS grid vs the interpolatedgrid
        # x_field is the physical rans grid 
        # x_grid is the new "physical" grid that corresponds to the computational grid (xi, eta)
        print(f"xmin RANS = {np.min(self.x_field)}")
        print(f"xmin new = {np.min(self.x_grid)}")

        points = np.column_stack((self.x_field.flatten(), self.y_field.flatten()))
    
        # Check for NaN or inf values in input
        if np.any(np.isnan(self.u_field)) or np.any(np.isinf(self.u_field)):
            if self.rank == 0:
                raise ValueError("Warning: NaN or inf values found in u_field")
            self.comm.Abort(1)
        if np.any(np.isnan(self.v_field)) or np.any(np.isinf(self.v_field)):
            if self.rank == 0:
                raise ValueError("Warning: NaN or inf values found in v_field")
            self.comm.Abort(1)
        if np.any(np.isnan(self.p_field)) or np.any(np.isinf(self.p_field)):
            if self.rank == 0:
                raise ValueError("Warning: NaN or inf values found in p_field")
            self.comm.Abort(1)
        
        # Interpolate with 'nearest' method and a fill_value
        self.u_grid = griddata(points, self.u_field.flatten(), (self.x_grid, self.y_grid), 
                               method='linear', fill_value=0)
        self.v_grid = griddata(points, self.v_field.flatten(), (self.x_grid, self.y_grid), 
                               method='linear', fill_value=0)
        self.p_grid = griddata(points, self.p_field.flatten(), (self.x_grid, self.y_grid), 
                               method='linear', fill_value=0)
        print_rz("Interpolating")
        print_rz(f"u_field max = {np.max(self.u_field.flatten())}")
        print_rz(f"u_grid max = {np.max(self.u_grid.flatten())}")
        
        # Check results
        if np.any(np.isnan(self.u_grid)):
            if self.rank == 0:
                print_rz(f"Percentage of NaN values in u_grid: {np.isnan(self.u_grid).mean()*100:.2f}%")
                raise ValueError("NaN values found in interpolated u_grid")
            self.comm.Abort(1)
        if np.any(np.isnan(self.v_grid)):
            if self.rank == 0:
                print_rz(f"Percentage of NaN values in v_grid: {np.isnan(self.v_grid).mean()*100:.2f}%")
                raise ValueError("NaN values found in interpolated v_grid")
            self.comm.Abort(1)
        if np.any(np.isnan(self.p_grid)):
            if self.rank == 0:
                print_rz(f"Percentage of NaN values in p_grid: {np.isnan(self.p_grid).mean()*100:.2f}%")
                raise ValueError("NaN values found in interpolated p_grid")
            self.comm.Abort(1)

    def calculate_metrics(self):
        # Calculate metrics of the coordinate transformation
        self.x_xi = np.gradient(self.x_grid, self.xi_grid[:, 0], axis=0)
        self.x_eta = np.gradient(self.x_grid, self.eta_grid[0, :], axis=1)
        self.y_xi = np.gradient(self.y_grid, self.xi_grid[:, 0], axis=0)
        self.y_eta = np.gradient(self.y_grid, self.eta_grid[0, :], axis=1)

        # Calculate Jacobian
        self.J = self.x_xi * self.y_eta - self.x_eta * self.y_xi

        # Calculate inverse metrics
        self.xi_x = self.y_eta / self.J
        self.xi_y = -self.x_eta / self.J
        self.eta_x = -self.y_xi / self.J
        self.eta_y = self.x_xi / self.J

        # Compute higher order metric terms

        # Compute higher order metric terms

        # Second derivatives
        self.x_xixi = np.gradient(self.x_xi, self.xi_grid[:, 0], axis=0)
        self.x_xieta = np.gradient(self.x_xi, self.eta_grid[0, :], axis=1)
        self.x_etaeta = np.gradient(self.x_eta, self.eta_grid[0, :], axis=1)

        self.y_xixi = np.gradient(self.y_xi, self.xi_grid[:, 0], axis=0)
        self.y_xieta = np.gradient(self.y_xi, self.eta_grid[0, :], axis=1)
        self.y_etaeta = np.gradient(self.y_eta, self.eta_grid[0, :], axis=1)

        # Compute additional metric terms
        self.xi_xx = (self.y_eta**2 * self.x_xixi - 2 * self.x_eta * self.y_eta * self.y_xixi + self.x_eta**2 * self.y_etaeta) / self.J**2
        self.xi_xy = (-self.y_eta**2 * self.y_xixi + self.x_eta * self.y_eta * (self.x_xixi + self.y_etaeta) - self.x_eta**2 * self.x_etaeta) / self.J**2
        self.xi_yy = (self.x_eta**2 * self.x_xixi - 2 * self.x_eta * self.y_eta * self.x_etaeta + self.y_eta**2 * self.y_etaeta) / self.J**2

        self.eta_xx = (self.y_xi**2 * self.x_etaeta - 2 * self.x_xi * self.y_xi * self.y_xieta + self.x_xi**2 * self.y_xixi) / self.J**2
        self.eta_xy = (-self.y_xi**2 * self.y_xieta + self.x_xi * self.y_xi * (self.x_etaeta + self.y_xixi) - self.x_xi**2 * self.x_xieta) / self.J**2
        self.eta_yy = (self.x_xi**2 * self.x_etaeta - 2 * self.x_xi * self.y_xi * self.x_xieta + self.y_xi**2 * self.y_xixi) / self.J**2

    def transform_velocities(self):
        # Transform velocities from physical to computational coordinates
        self.U_grid = self.u_grid * self.xi_x + self.v_grid * self.xi_y
        self.V_grid = self.u_grid * self.eta_x + self.v_grid * self.eta_y

    def get_metrics(self):
        return {
            'x_xi': self.x_xi, 'x_eta': self.x_eta,
            'y_xi': self.y_xi, 'y_eta': self.y_eta,
            'xi_x': self.xi_x, 'xi_y': self.xi_y,
            'eta_x': self.eta_x, 'eta_y': self.eta_y,
            'J': self.J
        }

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
                self.Dy = np.diag(1./(D@x))@D # d(.)/dy = d(.)/dxi * dxi/dy
            elif order==2: # d^2()/dy^2 = d^2()/dxi^2 (dxi/dy)^2 + d()/dxi d^2xi/dy^2
                D1= self.set_D_cheby(xi, order=1)
                D = D1@D1 # second derivative
                dxdxi = D1@x
                dxidx = 1./dxdxi # element wise invert
                d2xidx2 = -(D@x)*(dxidx)**3
                self.Dyy = (D*(dxidx[:,np.newaxis]**2)) + (D1*d2xidx2[:,np.newaxis])
            else:
                print('Cannot do this order of derivative with non-uniform mesh.  your input order of derivative = ',order)
        else:
            if order == 1:
                self.Dy = set_D_cheby(self, x, order)
            elif order == 2:
                self.Dyy = self.set_D_cheby(self, x, order)

class curvedSurfaceGrid_noInterpolation:
    def __init__(self, x_surface, y_surface, x_field, y_field, u_field, v_field, p_field, kappa, Ny, ymax, method="cheby", need_map=True, Uinf=1.0, nu=1.0):

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()

        self.x_surface = x_surface
        self.y_surface = y_surface
        self.x_field = x_field
        self.y_field = y_field
        self.u_field = u_field
        self.v_field = v_field
        self.p_field = p_field
        self.Nx = x_field.shape[0]
        self.Ny = Ny
        self.method = method
        self.need_map = need_map
        
        self.xi_grid = None
        self.hx = None
        self.eta_grid = None
        self.x_grid = None
        self.y_grid = None
        self.u_grid = None
        self.v_grid = None
        self.p_grid = None
        self.kappa = kappa

        self.Uinf = Uinf
        self.nu = nu

        self.generate_curvilinear_grid()
        self.interpolate_velocity_field()
        self.calculate_metrics()

        if self.method == "cheby" or method == "cheby2":
            self.map_D_cheby(self.ygrid, 1, self.need_map)
            self.map_D_cheby(self.ygrid, 2, self.need_map)

        elif self.method == "fd":
            self.Dy = self.set_D_FD(self.ygrid,d=1,order=2, output_full=True, uniform=False)
            self.Dyy = self.set_D_FD(self.ygrid,d=2,order=2, output_full=True, uniform=False)

        else:
            print("Invalid method chosen for derivatives")


    def generate_curvilinear_grid(self):
        # Generate xi coordinates along the surface
        arc_length = np.cumsum(np.sqrt(np.diff(self.x_surface)**2 + np.diff(self.y_surface)**2))
        arc_length = np.insert(arc_length, 0, 0)

        # In this class, xi is not a uniform field necesarilly 
        # for now, let's just use the xi field input by the user 
        # Advantage: don't need to interpolate in the streamwise direction 
        # Disadvantage: most likely the user won't supply a uniform grid
        # so we lose order of accuracy and make the spatial marching marginally more difficult
        self.hx = np.diff(self.x_field[:,0])

        # Generate eta coordinates normal to the surface
        if self.method == "cheby":
            a = 0.0
            b = 0.025
            eta_buf = np.cos(np.pi*np.arange(self.Ny-1,-1,-1)/(self.Ny-1))
            eta = b*(eta_buf+1)/2.0 + a*(1.0-eta_buf)/2.0

        if self.method == "cheby2":
            a = 0.0
            b = 1.0
            y_i = 0.2
            eta_buf = np.cos(np.pi*np.arange(self.Ny-1,-1,-1)/(self.Ny-1))
            buf1 = y_i * b / (b - 2 * y_i)
            buf2 = 1 + 2 * buf1 / b
            eta = buf1 * (1 + eta_buf) / (buf2 - eta_buf)

        elif self.method == "fd":
            # a = 0.0
            # b = 1.0
            # eta = np.linspace(a, b, self.Ny)
            # eta = b * (1. + (np.tanh(delta * (y / b - 1.)) / np.tanh(delta)))
            eta = self.y_field[0,:]


        # Generate 1D xi and eta grids used by the solver
        self.xgrid = self.x_field[:,0]
        self.ygrid = eta

        xi = self.x_field[:,0]
        xi_grid = np.tile(xi[:, np.newaxis], (1, self.Ny))

        eta_grid = np.tile(eta[np.newaxis, :], (self.x_field.shape[0],1))

        self.xi_grid = xi_grid
        self.eta_grid = eta_grid

        self.x_grid = self.xi_grid
        self.y_grid = self.eta_grid

        # input kappa is a 1d vevtor 
        # need to expand it for consistency reasons
        self.kappa = self.kappa[:, np.newaxis]
        print_rz(f"Initializing grid:")
        print_rz(f"x_grid shape: {self.x_grid.shape}")
        print_rz(f"y_grid shape: {self.y_grid.shape}")
        print_rz(f"kappa[0:20] = {self.kappa[0:20,0]}")

        self.h = 1 + self.kappa * self.eta_grid # need a transpose to properly multiply these

    def interpolate_velocity_field(self):

        # Check for NaN or inf values in input
        if np.any(np.isnan(self.u_field)) or np.any(np.isinf(self.u_field)):
            if self.rank == 0:
                raise ValueError("Warning: NaN or inf values found in u_field")
            self.comm.Abort(1)
        if np.any(np.isnan(self.v_field)) or np.any(np.isinf(self.v_field)):
            if self.rank == 0:
                raise ValueError("Warning: NaN or inf values found in v_field")
            self.comm.Abort(1)
        if np.any(np.isnan(self.p_field)) or np.any(np.isinf(self.p_field)):
            if self.rank == 0:
                raise ValueError("Warning: NaN or inf values found in p_field")
            self.comm.Abort(1)

        # Create points for interpolation 
        # points = np.column_stack((self.x_field.flatten(), self.y_field.flatten()))

        # Create the interpolation function 

        # interp_func_u = interpolate.LinearNDInterpolator(points, self.u_field.flatten(), fill_value=np.nan)
        # interp_func_v = interpolate.LinearNDInterpolator(points, self.v_field.flatten(), fill_value=np.nan)
        # interp_func_p = interpolate.LinearNDInterpolator(points, self.p_field.flatten(), fill_value=np.nan)

        # # create meshgrid for new points 
        # xi_new, eta_new = np.meshgrid(self.x_field, self.ygrid, indexing = 'ij')

        # self.u_grid = interp_func_u(self.xi_grid, self.eta_grid)
        # self.v_grid = interp_func_v(self.xi_grid, self.eta_grid)
        # self.p_grid = interp_func_p(self.xi_grid, self.eta_grid)
        self.u_grid = self.u_field
        self.v_grid = self.v_field
        self.p_grid = self.p_field

        self.U_grid = self.u_grid
        self.V_grid = self.v_grid


        # compute displacement thickness 
        self.u_field = self.u_field / self.Uinf
        self.v_field = self.v_field / self.Uinf

        self.U_grid = self.U_grid / self.Uinf
        self.V_grid = self.V_grid / self.Uinf

        delta_star = np.trapz(1 - self.u_field[0,:] / np.max(self.u_field[0,:]), self.y_field[0,:])
        R = self.Uinf / (delta_star * self.nu)

        # self.x_field = self.x_field / (R * delta_star)
        # self.y_field = self.y_field / (delta_star)
        # self.xi_grid = self.xi_grid / (R * delta_star)
        # self.eta_grid = self.eta_grid / (delta_star)
        #
        # self.xgrid = self.xgrid / (R * delta_star)
        # self.ygrid = self.ygrid / (delta_star)

        # Check results
        # if np.any(np.isnan(self.u_grid)):
        #     if self.rank == 0:
        #         print_rz(f"Percentage of NaN values in u_grid: {np.isnan(self.u_grid).mean()*100:.2f}%")
        #         raise ValueError("NaN values found in interpolated u_grid")
        #     self.comm.Abort(1)
        # if np.any(np.isnan(self.v_grid)):
        #     if self.rank == 0:
        #         print_rz(f"Percentage of NaN values in v_grid: {np.isnan(self.v_grid).mean()*100:.2f}%")
        #         raise ValueError("NaN values found in interpolated v_grid")
        #     self.comm.Abort(1)
        # if np.any(np.isnan(self.p_grid)):
        #     if self.rank == 0:
        #         print_rz(f"Percentage of NaN values in p_grid: {np.isnan(self.p_grid).mean()*100:.2f}%")
        #         raise ValueError("NaN values found in interpolated p_grid")
        #     self.comm.Abort(1)

    def calculate_metrics(self):
        # Calculate metrics of the coordinate transformation
        self.x_xi = np.gradient(self.x_grid, self.xi_grid[:, 0], axis=0)
        self.x_eta = np.gradient(self.x_grid, self.eta_grid[0, :], axis=1)
        self.y_xi = np.gradient(self.y_grid, self.xi_grid[:, 0], axis=0)
        self.y_eta = np.gradient(self.y_grid, self.eta_grid[0, :], axis=1)

        # Calculate Jacobian
        self.J = self.x_xi * self.y_eta - self.x_eta * self.y_xi

        # Calculate inverse metrics
        self.xi_x = self.y_eta / self.J
        self.xi_y = -self.x_eta / self.J
        self.eta_x = -self.y_xi / self.J
        self.eta_y = self.x_xi / self.J

        self.J11 = self.y_eta / self.J
        self.J12 = -self.x_eta / self.J
        self.J21 = -self.y_xi / self.J
        self.J22 = self.x_xi / self.J

        # Compute higher order metric terms

        self.dJ11_dxi = np.gradient(self.J11, self.xi_grid[:, 0], axis=0)
        self.dJ11_deta = np.gradient(self.J11, self.eta_grid[0, :], axis=1)

        self.dJ12_dxi = np.gradient(self.J12, self.xi_grid[:, 0], axis=0)
        self.dJ12_deta = np.gradient(self.J12, self.eta_grid[0, :], axis=1)
        
        self.dJ21_dxi = np.gradient(self.J21, self.xi_grid[:, 0], axis=0)
        self.dJ21_deta = np.gradient(self.J21, self.eta_grid[0, :], axis=1)

        self.dJ22_dxi = np.gradient(self.J22, self.xi_grid[:, 0], axis=0)
        self.dJ22_deta = np.gradient(self.J22, self.eta_grid[0, :], axis=1)
        # Compute higher order metric terms

        # Second derivatives
        self.x_xixi = np.gradient(self.x_xi, self.xi_grid[:, 0], axis=0)
        self.x_xieta = np.gradient(self.x_xi, self.eta_grid[0, :], axis=1)
        self.x_etaeta = np.gradient(self.x_eta, self.eta_grid[0, :], axis=1)

        self.y_xixi = np.gradient(self.y_xi, self.xi_grid[:, 0], axis=0)
        self.y_xieta = np.gradient(self.y_xi, self.eta_grid[0, :], axis=1)
        self.y_etaeta = np.gradient(self.y_eta, self.eta_grid[0, :], axis=1)

        # Compute additional metric terms
        self.xi_xx = (self.y_eta**2 * self.x_xixi - 2 * self.x_eta * self.y_eta * self.y_xixi + self.x_eta**2 * self.y_etaeta) / self.J**2
        self.xi_xy = (-self.y_eta**2 * self.y_xixi + self.x_eta * self.y_eta * (self.x_xixi + self.y_etaeta) - self.x_eta**2 * self.x_etaeta) / self.J**2
        self.xi_yy = (self.x_eta**2 * self.x_xixi - 2 * self.x_eta * self.y_eta * self.x_etaeta + self.y_eta**2 * self.y_etaeta) / self.J**2

        self.eta_xx = (self.y_xi**2 * self.x_etaeta - 2 * self.x_xi * self.y_xi * self.y_xieta + self.x_xi**2 * self.y_xixi) / self.J**2
        self.eta_xy = (-self.y_xi**2 * self.y_xieta + self.x_xi * self.y_xi * (self.x_etaeta + self.y_xixi) - self.x_xi**2 * self.x_xieta) / self.J**2
        self.eta_yy = (self.x_xi**2 * self.x_etaeta - 2 * self.x_xi * self.y_xi * self.x_xieta + self.y_xi**2 * self.y_xixi) / self.J**2

    def get_metrics(self):
        return {
            'x_xi': self.x_xi, 'x_eta': self.x_eta,
            'y_xi': self.y_xi, 'y_eta': self.y_eta,
            'xi_x': self.xi_x, 'xi_y': self.xi_y,
            'eta_x': self.eta_x, 'eta_y': self.eta_y,
            'J': self.J
        }

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
                self.Dy = np.diag(1./(D@x))@D # d(.)/dy = d(.)/dxi * dxi/dy
            elif order==2: # d^2()/dy^2 = d^2()/dxi^2 (dxi/dy)^2 + d()/dxi d^2xi/dy^2
                D1= self.set_D_cheby(xi, order=1)
                D = D1@D1 # second derivative
                dxdxi = D1@x
                dxidx = 1./dxdxi # element wise invert
                d2xidx2 = -(D@x)*(dxidx)**3
                self.Dyy = (D*(dxidx[:,np.newaxis]**2)) + (D1*d2xidx2[:,np.newaxis])
            else:
                print('Cannot do this order of derivative with non-uniform mesh.  your input order of derivative = ',order)
        else:
            if order == 1:
                self.Dy = set_D_cheby(self, x, order)
            elif order == 2:
                self.Dyy = self.set_D_cheby(self, x, order)

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

        if self.method == "cheby" or method == "cheby2":
            self.map_D_cheby(self.ygrid, 1, self.need_map)
            self.map_D_cheby(self.ygrid, 2, self.need_map)

        elif self.method == "fd" or self.method == "geometric" or self.method=="uniform":
            self.Dy = self.set_D_FD(self.ygrid,d=1,order=2, output_full=True, uniform=False)
            self.Dyy = self.set_D_FD(self.ygrid,d=2,order=2, output_full=True, uniform=False)

        else:
            raise ValueError("Invalid method chosen for derivatives")

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
            # print_rz("These shapes match. Interpolation is not required.")
            points = np.column_stack((self.x_field.flatten(), self.y_field.flatten()))
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
                self.Dy = np.diag(1./(D@x))@D # d(.)/dy = d(.)/dxi * dxi/dy
            elif order==2: # d^2()/dy^2 = d^2()/dxi^2 (dxi/dy)^2 + d()/dxi d^2xi/dy^2
                D1= self.set_D_cheby(xi, order=1)
                D = D1@D1 # second derivative
                dxdxi = D1@x
                dxidx = 1./dxdxi # element wise invert
                d2xidx2 = -(D@x)*(dxidx)**3
                self.Dyy = (D*(dxidx[:,np.newaxis]**2)) + (D1*d2xidx2[:,np.newaxis])
            else:
                print('Cannot do this order of derivative with non-uniform mesh.  your input order of derivative = ',order)
        else:
            if order == 1:
                self.Dy = set_D_cheby(self, x, order)
            elif order == 2:
                self.Dyy = self.set_D_cheby(self, x, order)

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

        if self.method == "cheby" or method == "cheby2":
            self.map_D_cheby(self.ygrid, 1, self.need_map)
            self.map_D_cheby(self.ygrid, 2, self.need_map)

        elif self.method == "fd" or self.method == "geometric" or self.method=="uniform":
            self.Dy = self.set_D_FD(self.ygrid,d=1,order=2, output_full=True, uniform=False)
            self.Dyy = self.set_D_FD(self.ygrid,d=2,order=2, output_full=True, uniform=False)

        else:
            raise ValueError("Invalid method chosen for derivatives")

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
        print_rz(f"Grid Generation. Kappa = {K1}")
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
                self.Dy = np.diag(1./(D@x))@D # d(.)/dy = d(.)/dxi * dxi/dy
            elif order==2: # d^2()/dy^2 = d^2()/dxi^2 (dxi/dy)^2 + d()/dxi d^2xi/dy^2
                D1= self.set_D_cheby(xi, order=1)
                D = D1@D1 # second derivative
                dxdxi = D1@x
                dxidx = 1./dxdxi # element wise invert
                d2xidx2 = -(D@x)*(dxidx)**3
                self.Dyy = (D*(dxidx[:,np.newaxis]**2)) + (D1*d2xidx2[:,np.newaxis])
            else:
                print('Cannot do this order of derivative with non-uniform mesh.  your input order of derivative = ',order)
        else:
            if order == 1:
                self.Dy = set_D_cheby(self, x, order)
            elif order == 2:
                self.Dyy = self.set_D_cheby(self, x, order)


import sys
import os
import numpy as np
import scipy as sp
from inputOutput import print_rz
from scipy.integrate import cumtrapz
from scipy.integrate import trapz
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
import copy

import numMethod as nm
import flow as flow
import initialize as ini
from numba import njit, complex128, int64
from mpi4py import MPI

@njit
def inner_product(x,y):
    ''' Return the inner product of two vectors <x,y> or y^H x operation
    For PSE I have used this type of inner product to update alpha and it seems stable enough, however a true integration inner product using Chebfun was more robust and required less iterations.

    Inputs:
        x:np.ndarray containing the x in <x,y> or y^H x inner product operation
        y:np.ndarray containing the y in <x,y>

    Returns:
        y^H x: np.complex
    '''

    return y.conj()@x

class NLPSE():
    """
    Class for Non-Linear Parabolized Stability Equations (NLPSE) solver.
    """

    def __init__(self, Grid, config, Baseflow):

        # documentation for NLPSE class
        # This class contains all the necessary information to solve the nonlinear PSE equations
        # for the mean flow. The class contains the following methods:
        # 1. __init__ : initializes the class with the necessary parameters
        # 2. viableHarmonics : computes the relevant harmonics for the computation of the nonlinear terms
        # 3. NLTHelper : computes the necessary modes for the computation of the nonlinear terms
        # 4. NLTConvolution : computes the convolution of the nonlinear terms
        # 5. computeNLT : computes the nonlinear terms for the mean flow
        # 6. solveMeanFlow : solves the mean flow equations for the mean flow
        # 7. updateMeanFlow : updates the mean flow using the computed nonlinear terms
        # 8. sampleGrid : samples the grid

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.station = None # every x station in the PSE to solve

        self.Grid      = Grid

        self.xgrid     = Grid.xgrid
        self.ygrid     = Grid.ygrid
        self.Nx        = Grid.Nx
        self.Ny        = Grid.Ny

        # Flow params
        self.config    = config
        # Base flow
        self.Baseflow  = Baseflow

        self.U         = np.zeros((self.Ny, self.Nx))
        self.Uy        = np.zeros((self.Ny, self.Nx))
        self.Ux        = np.zeros((self.Ny, self.Nx))
        self.V         = np.zeros((self.Ny, self.Nx))
        self.Vy        = np.zeros((self.Ny, self.Nx))
        self.Vx        = np.zeros((self.Ny, self.Nx))
        self.W         = np.zeros((self.Ny, self.Nx))
        self.Wy        = np.zeros((self.Ny, self.Nx))
        self.Wx        = np.zeros((self.Ny, self.Nx))
        self.P         = np.zeros((self.Ny, self.Nx))

        self.U_nlt0    = np.zeros((self.Ny, self.Nx))
        self.V_nlt0    = np.zeros((self.Ny, self.Nx))
        self.P_nlt0    = np.zeros((self.Ny, self.Nx))

        # Differentiation
        self.Dy        = Grid.Dy
        self.Dyy       = Grid.Dyy

        # self.hx = np.zeros(self.Nx)
        self.hx = Grid.hx

        # Modes
        self.numM = config['modes']['temporal']
        self.numN = config['modes']['spanwise']
        self.harmonics = None

        # Solution at the current iteration
        self.alpha = np.zeros((self.Nx, self.numM, self.numN), dtype=complex)
        self.q      = np.zeros((self.Nx, self.numM, self.numN, 4*self.Ny), dtype=complex)
        self.Fmn = np.zeros((self.Nx, self.numM, self.numN, 4 * self.Ny), dtype=complex)
        # self.opEigs = np.zeros((self.Nx, self.numM, self.numN, config['numerical']['numEigs']), dtype=complex)
        self.opEigs = np.zeros((self.Nx, self.numM, self.numN, 4*self.Ny), dtype=complex)

        # PSE Operators
        self.C = None 
        self.By = None
        self.Byy = None
        self.A = None
        self.stabilizer = 0.0

        self.A_solve = None
        self.b = None

        # helper mats
        self.helper_mats = None

        # mesh transformation stuff
        # these are the inverse metrics
        self.J11 = Grid.J11
        self.J12 = Grid.J12
        self.J21 = Grid.J21
        self.J22 = Grid.J22

        self.dJ11_dx = Grid.dJ11_dxi
        # cross terms are equal
        self.dJ12_dxi = Grid.dJ12_dxi
        self.dJ11_deta = Grid.dJ11_deta
        self.dJ12_deta = Grid.dJ12_deta

        self.dJ21_dxi = Grid.dJ21_dxi
        self.dJ22_dxi = Grid.dJ22_dxi
        self.dJ21_deta = Grid.dJ21_deta
        self.dJ22_deta = Grid.dJ22_deta

        self.h = Grid.h
        self.kappa = Grid.kappa
        
    def sampleGrid(self,Grid):

        """
        Samples the grid for the NLPSE solver.

        Parameters:
            Grid (Grid): Grid object containing mesh information.
        """

        self.xgrid     = Grid.xgrid
        self.ygrid     = Grid.ygrid
        self.Nx        = Grid.Nx
        self.Ny        = Grid.Ny

        # mesh transformation stuff
        self.J11 = Grid.J11
        self.J12 = Grid.J12
        self.J21 = Grid.J21
        self.J22 = Grid.J22

        self.dJ11_dx = Grid.dJ11_dxi
        # cross terms are equal
        self.dJ12_dxi = Grid.dJ12_dxi
        self.dJ11_deta = Grid.dJ11_deta
        self.dJ12_deta = Grid.dJ12_deta

        self.dJ21_dxi = Grid.dJ21_dxi
        self.dJ22_dxi = Grid.dJ22_dxi
        self.dJ21_deta = Grid.dJ21_deta
        self.dJ22_deta = Grid.dJ22_deta

        self.h = Grid.h
        self.kappa = Grid.kappa

    def viableHarmonics(self, init_modes, mode_to_remove=None, num_repeats=3):
        """
        Determines viable harmonics for the initial modes.

        Parameters:
            init_modes (array-like): Initial modes.
            num_repeats (int, optional): Number of repeats. Defaults to 3.

        Returns:
        list: A list of unique tuples that satisfy the conditions.

        """

        for i in range(init_modes.shape[0]):
            if i == 0:
                result = [tuple(init_modes[0])]
            else:
                result.append(tuple(init_modes[i]))
        for repeat in range(num_repeats):
            loop_arr = list(result)
            for i in range(len(loop_arr)):
                for j in range(i+1, len(loop_arr)):
                    pair_sum = tuple(map(sum, zip(loop_arr[i], loop_arr[j])))  # Calculate the sum of tuple pairs
                    pair_diff = tuple(map(lambda x, y: x - y, loop_arr[i], loop_arr[j]))  # Calculate the difference of tuple pairs
                    result.append(pair_sum)
                    result.append(pair_diff)  # Add subtracted pair to the result
        result = list(set(result))
        
        # Remove tuples whose first entry is greater than M or second entry is greater than N
        # have to subtract by one to account for MFD
        result = [t for t in result if t[0] <= self.numM-1 and t[1] <= self.numN-1]
        # Remove tuples with negative numbers
        result = [t for t in result if all(x >= 0 for x in t)]

        result = sorted(result)

        # print(f"Harmonics: {result}")

        if mode_to_remove:
            filtered_harmonics = [mode for mode in result if tuple(mode) != mode_to_remove]
            return np.array(filtered_harmonics)
        else:
            return np.array(result)

    def NLTHelper(self, mat, field):
        """
        Given (m,n) modes where m and n are positive integers, this function computes the full matrix of 
        modal information required for the nonlinear sum in the PSE equations. This function leverages known 
        symmetry properties of the modes in the linear equations to compute the full matrix.

        Parameters:
            - mat: numpy array
            The input matrix.
            - field: str
            The field for which the computation is performed. Possible values are 'alpha', 'beta', 'w', 'wx', 'wy'.

        Returns:
            - full_mat: numpy array
        """

        if self.numN == 1:

            # this is the case where there are no spanwise modes

            buf = np.copy(np.flip(mat, axis=0))
            buf[self.numM-1,:,:] = 0 # zero out the (0, 0) mode
            buf = np.conjugate(buf)

            if field == 'alpha':
                buf *= -1

            full_mat = mat + buf

            return full_mat

        elif self.numM == 1:

            # case where there are no temporal modes (crossflow instabilities only)
            buf = np.copy(np.flip(mat, axis=1))
            buf[:, self.numN-1, :] = 0 # zero out the (0, 0) mode
            buf = np.conjugate(buf)
            if field == 'alpha' or field == 'beta': 
                buf *= -1

            full_mat = mat + buf
            return full_mat

        else:

            if field == 'w' or field == 'wx' or field == 'wy': 

                buf = np.copy(np.flip(mat, axis=1))
                buf[:,self.numN-1,:] = 0
                buf[self.numM-1,self.numN-1:,:] = np.conjugate(buf[self.numM-1,self.numN-1:,:])
                buf[0:self.numM-1,self.numN-1:,:] *= -1
                step1 = mat + buf

                buf = np.copy(np.flip(step1, axis=0))
                buf[self.numM-1,:,:] = 0
                buf *= -1
                buf[self.numM:, self.numN-1,:] *= -1 
                buf[self.numM-1:,:,:] = np.conjugate(buf[self.numM-1:,:,:])
                full_mat = step1 + buf

                return full_mat

            elif field == 'beta':
                
                buf = np.copy(np.flip(mat, axis=1))
                buf[:,self.numN-1,:] = 0
                buf[:,self.numN-1:,:] *= -1
                step1 = mat + buf

                buf = np.copy(np.flip(step1, axis=0))
                buf[self.numM-1,:,:] = 0
                full_mat = step1 + buf

                return full_mat

            elif field == 'alpha':

                buf = np.copy(np.flip(mat, axis=1))
                buf[:,self.numN-1,:] = 0
                buf[self.numM-1, self.numN-1:, :] = -1 * np.conjugate(buf[self.numM-1,self.numN-1:,:])
                step1 = mat + buf

                buf = np.copy(np.flip(step1, axis=0))
                buf[self.numM-1,:,:] = 0
                buf[self.numM-1:,:,:] = -1 * np.conjugate(buf[self.numM-1:,:,:])
                full_mat = step1 + buf

                return full_mat

            else:
                
                buf = np.copy(np.flip(mat, axis=1))
                buf[:,self.numN-1,:] = 0
                buf[self.numM-1,self.numN-1:,:] = np.conjugate(buf[self.numM-1,self.numN-1:,:])
                step1 = mat + buf

                buf = np.copy(np.flip(step1, axis=0))
                buf[self.numM-1,:,:] = 0
                buf[self.numM-1:,:,:] = np.conjugate(buf[self.numM-1:,:,:])
                full_mat = step1 + buf

                return full_mat

    def NLTConvolution(self, Ny, J11, J12, J21, J22,
        sumNum_m, sumNum_n, numM, numN,
        uhat_mn, uhatx_mn, uhaty_mn,
        vhat_mn, vhatx_mn, vhaty_mn,
        what_mn, whatx_mn, whaty_mn,
        Ialpha, Ialpha_mn, alpha_mn, beta_mn,
        one_over_h, kappa):

        umomNLT = np.zeros(Ny,dtype=complex)
        vmomNLT = np.zeros(Ny,dtype=complex)
        wmomNLT = np.zeros(Ny,dtype=complex)
        Fmn = np.zeros((1,4 * Ny), dtype=complex)

        for mm1 in range(2 * numM - 1):
            for mm2 in range(2 * numM - 1):

                if mm1 + mm2 != sumNum_m:
                    continue

                else:

                    for nn1 in range(2 * numN - 1):
                        for nn2 in range(2 * numN - 1):

                            mag1 = np.max(np.abs(uhat_mn[mm2,nn2,:]))
                            mag2 = np.max(np.abs(vhat_mn[mm2,nn2,:]))
                            mag3 = np.max(np.abs(what_mn[mm2,nn2,:]))

                            if mag1 > mag2:
                                max_val = mag1
                            else:
                                max_val = mag2

                            if mag3 > max_val:
                                max_val = mag3

                            if nn1 + nn2 != sumNum_n:# or max_val < 1e-7:#or np.max([mag1,mag2,mag3]) < 1e-7:
                                continue
                            else:

                                umomNLT += (one_over_h * uhat_mn[mm1, nn1, :] * (uhatx_mn[mm2,nn2,:] + 1.0j * alpha_mn[mm2, nn2, :] * uhat_mn[mm2, nn2, :]) + \
                                      vhat_mn[mm1, nn1, :] * uhaty_mn[mm2, nn2, :] + \
                                      what_mn[mm1, nn1, :] * 1.0j * beta_mn[mm2, nn2, :] * uhat_mn[mm2, nn2, :] + \
                                        one_over_h * uhat_mn[mm1,nn1,:] * vhat_mn[mm2,nn2,:] * kappa) * (Ialpha_mn[mm1, nn1, :] * Ialpha_mn[mm2, nn2, :] * Ialpha)

                                vmomNLT += (one_over_h * uhat_mn[mm1, nn1, :] * (vhatx_mn[mm2,nn2,:] + 1.0j * alpha_mn[mm2, nn2, :] * vhat_mn[mm2, nn2, :]) + \
                                      vhat_mn[mm1, nn1, :] * vhaty_mn[mm2, nn2, :] + \
                                      what_mn[mm1, nn1, :] * 1.0j * beta_mn[mm2, nn2, :] * vhat_mn[mm2, nn2, :] + \
                                        -1 * one_over_h * kappa * uhat_mn[mm1, nn1, :] * uhat_mn[mm2, nn2, :]) * (Ialpha_mn[mm1, nn1, :] * Ialpha_mn[mm2, nn2, :] * Ialpha)

                                wmomNLT += (one_over_h * uhat_mn[mm1, nn1, :] * (whatx_mn[mm2,nn2,:] + 1.0j * alpha_mn[mm2, nn2, :] * what_mn[mm2, nn2, :]) + \
                                      vhat_mn[mm1, nn1, :] * whaty_mn[mm2, nn2, :] + \
                                      what_mn[mm1, nn1, :] * 1.0j * beta_mn[mm2, nn2, :] * what_mn[mm2, nn2, :]) * (Ialpha_mn[mm1, nn1, :] * Ialpha_mn[mm2, nn2, :] * Ialpha)

        Fmn[0,0:Ny]       = -umomNLT 
        Fmn[0,Ny:2*Ny]    = -vmomNLT 
        Fmn[0,2*Ny:3*Ny]  = -wmomNLT

        return Fmn

    def vectorized_NLTConvolution(self, Ny, J11, J12, J21, J22,
                               sumNum_m, sumNum_n, numM, numN,
                               uhat_mn, uhatx_mn, uhaty_mn,
                               vhat_mn, vhatx_mn, vhaty_mn,
                               what_mn, whatx_mn, whaty_mn,
                               Ialpha, Ialpha_mn, alpha_mn, beta_mn,
                               one_over_h, kappa):


        # Step 1: Create all possible combinations of indices
        mm1, mm2 = np.meshgrid(np.arange(2*numM-1), np.arange(2*numM-1), indexing='ij')
        nn1, nn2 = np.meshgrid(np.arange(2*numN-1), np.arange(2*numN-1), indexing='ij')
        
        # Step 2: Create masks for valid combinations
        mask_m = mm1 + mm2 == sumNum_m
        mask_n = nn1 + nn2 == sumNum_n
        mask = mask_m[:, :, np.newaxis, np.newaxis] & mask_n[np.newaxis, np.newaxis, :, :]
        
        # Step 3: Prepare indices for array indexing
        mm1 = mm1[:, :, np.newaxis, np.newaxis]
        mm2 = mm2[:, :, np.newaxis, np.newaxis]
        nn1 = nn1[np.newaxis, np.newaxis, :, :]
        nn2 = nn2[np.newaxis, np.newaxis, :, :]
        
        # Step 4: Prepare inputs by broadcasting to full 5D shape (mm1, mm2, nn1, nn2, Ny)
        uhat_mn1 = uhat_mn[mm1, nn1, :]
        vhat_mn1 = vhat_mn[mm1, nn1, :]
        what_mn1 = what_mn[mm1, nn1, :]
        
        uhat_mn2 = uhat_mn[mm2, nn2, :]
        vhat_mn2 = vhat_mn[mm2, nn2, :]
        what_mn2 = what_mn[mm2, nn2, :]
        
        uhatx_mn2 = uhatx_mn[mm2, nn2, :]
        uhaty_mn2 = uhaty_mn[mm2, nn2, :]
        vhatx_mn2 = vhatx_mn[mm2, nn2, :]
        vhaty_mn2 = vhaty_mn[mm2, nn2, :]
        whatx_mn2 = whatx_mn[mm2, nn2, :]
        whaty_mn2 = whaty_mn[mm2, nn2, :]
        
        alpha_mn2 = alpha_mn[mm2, nn2, :]
        beta_mn2 = beta_mn[mm2, nn2, :]
        
        Ialpha_mn1 = Ialpha_mn[mm1, nn1, :]
        Ialpha_mn2 = Ialpha_mn[mm2, nn2, :]
        
        # Step 5: Expand mask to match the 5D shape
        mask = mask[..., np.newaxis]
        
        # Step 6: Compute NLT terms
        umomNLT = (one_over_h * uhat_mn1 * (uhatx_mn2 + 1.0j * alpha_mn2 * uhat_mn2) +
                vhat_mn1 * uhaty_mn2 +
                what_mn1 * 1.0j * beta_mn2 * uhat_mn2 +
                one_over_h * uhat_mn1 * vhat_mn2 * kappa) * \
                (Ialpha_mn1 * Ialpha_mn2 * Ialpha)
        
        vmomNLT = (one_over_h * uhat_mn1 * (vhatx_mn2 + 1.0j * alpha_mn2 * vhat_mn2) +
                vhat_mn1 * vhaty_mn2 +
                what_mn1 * 1.0j * beta_mn2 * vhat_mn2 -
                one_over_h * kappa * uhat_mn1 * uhat_mn2) * \
                (Ialpha_mn1 * Ialpha_mn2 * Ialpha)
        
        wmomNLT = (one_over_h * uhat_mn1 * (whatx_mn2 + 1.0j * alpha_mn2 * what_mn2) +
                vhat_mn1 * whaty_mn2 +
                what_mn1 * 1.0j * beta_mn2 * what_mn2) * \
                (Ialpha_mn1 * Ialpha_mn2 * Ialpha)
        
        # Step 7: Apply mask and sum
        umomNLT = np.sum(np.where(mask, umomNLT, 0), axis=(0, 1, 2, 3))
        vmomNLT = np.sum(np.where(mask, vmomNLT, 0), axis=(0, 1, 2, 3))
        wmomNLT = np.sum(np.where(mask, wmomNLT, 0), axis=(0, 1, 2, 3))
        
        # Step 8: Prepare output
        Fmn = np.zeros((1, 4 * Ny), dtype=complex)
        Fmn[0, 0:Ny] = -umomNLT
        Fmn[0, Ny:2*Ny] = -vmomNLT
        Fmn[0, 2*Ny:3*Ny] = -wmomNLT
        
        return Fmn

    def computeNLT(self, helper_mats, station, m, n):
        """
        Computes the non-linear terms (i.e. the RHS) of the nonlinear PSE equations for the u, v, and w momentum equations.
        The computation is done in spectral space so that the built-in harmonic balancing means we don't need to worry 
        about aliasing errors

        Parameters:
            helper_mats (dict): Helper matrices.
            station (int): Station index.
            m, n (int): Mode indices.
        """
        # This method compute the nonlinear terms (i.e. the RHS) of 
        # the nonlinear PSE equations for the u, v, and w momentum, equations
        # The computation is done in spectral space so that the built-in harmonic
        # balancing means we don't need to worry about aliasing errors

        # initialize memory
        ny = self.Ny

        uhat_mn  = np.zeros((2 * self.numM - 1, 2 * self.numN - 1, self.Ny), dtype=complex)
        uhatx_mn = np.zeros((2 * self.numM - 1, 2 * self.numN - 1, self.Ny), dtype=complex)
        uhaty_mn = np.zeros((2 * self.numM - 1, 2 * self.numN - 1, self.Ny), dtype=complex)
        uhatz_mn = np.zeros((2 * self.numM - 1, 2 * self.numN - 1, self.Ny), dtype=complex)

        vhat_mn  = np.zeros((2 * self.numM - 1, 2 * self.numN - 1, self.Ny), dtype=complex)
        vhatx_mn = np.zeros((2 * self.numM - 1, 2 * self.numN - 1, self.Ny), dtype=complex)
        vhaty_mn = np.zeros((2 * self.numM - 1, 2 * self.numN - 1, self.Ny), dtype=complex)
        vhatz_mn = np.zeros((2 * self.numM - 1, 2 * self.numN - 1, self.Ny), dtype=complex)

        what_mn  = np.zeros((2 * self.numM - 1, 2 * self.numN - 1, self.Ny), dtype=complex)
        whatx_mn = np.zeros((2 * self.numM - 1, 2 * self.numN - 1, self.Ny), dtype=complex)
        whaty_mn = np.zeros((2 * self.numM - 1, 2 * self.numN - 1, self.Ny), dtype=complex)
        whatz_mn = np.zeros((2 * self.numM - 1, 2 * self.numN - 1, self.Ny), dtype=complex)

        alpha_mn  = np.zeros((2 * self.numM - 1, 2 * self.numN - 1, self.Ny), dtype=complex)
        beta_mn   = np.zeros((2 * self.numM - 1, 2 * self.numN - 1, self.Ny), dtype=complex)
        Ialpha_mn = np.zeros((2 * self.numM - 1, 2 * self.numN - 1, self.Ny), dtype=complex)

        mCount = self.numM-1
        mMid   = self.numM -1

        nCount = self.numN - 1
        nMid   = self.numN - 1

        # note the minus sign to match Herbert in the special AGARD report
        Ialpha = np.exp(-1.0j * trapz(self.alpha[0:station+1, m, n], self.xgrid[0:station+1] - self.xgrid[0]))

        for mm in range(0, self.numM):
            for nn in range(0, self.numN):

                q = np.copy(self.q[station, mCount-mm, nCount-nn, :])
                qm1 = np.copy(self.q[station - 1, mCount-mm, nCount-nn, :])

                # if mCount == mm and nCount == nn:
                #     print_rz(f"qMFD in nlt convolution = ")
                #     print_rz(f"{self.q[station, 0, 0, :]}")

                alpha_mn[mm, nn, :]    = self.alpha[station, mCount-mm, nCount-nn] 
                Ialpha_mn[mm, nn, :]   = np.exp(1.0j * trapz(self.alpha[0:station+1, mCount-mm, nCount-nn], self.xgrid[0:station+1] - self.xgrid[0]))
                beta_mn[mm, nn, :]     = self.config['disturbance']['beta']* (nCount - nn)

                uhat_mn[mm, nn, :]  = helper_mats['u_from_SPE'] @ q 
                uhatx_mn[mm, nn, :] = ((helper_mats['u_from_SPE'] @ q) - (helper_mats['u_from_SPE'] @ qm1)) / self.hx[station]
                uhaty_mn[mm, nn, :] = self.Dy @ (helper_mats['u_from_SPE'] @ q)

                vhat_mn[mm, nn, :]  = helper_mats['v_from_SPE'] @ q 
                vhatx_mn[mm, nn, :] = ((helper_mats['v_from_SPE'] @ q) - (helper_mats['v_from_SPE'] @ qm1)) / self.hx[station]
                vhaty_mn[mm, nn, :] = self.Dy @ (helper_mats['v_from_SPE'] @ q)

                what_mn[mm, nn, :]  = helper_mats['w_from_SPE'] @ q 
                whatx_mn[mm, nn, :] = ((helper_mats['w_from_SPE'] @ q) - (helper_mats['w_from_SPE'] @ qm1)) / self.hx[station]
                whaty_mn[mm, nn, :] = self.Dy @ (helper_mats['w_from_SPE'] @ q)

        alpha_mn = self.NLTHelper(alpha_mn, 'alpha')
        Ialpha_mn = self.NLTHelper(Ialpha_mn, 'Ialpha')
        beta_mn   = self.NLTHelper(beta_mn, 'beta')

        uhat_mn = self.NLTHelper(uhat_mn, 'u')
        uhatx_mn = self.NLTHelper(uhatx_mn, 'ux')
        uhaty_mn = self.NLTHelper(uhaty_mn, 'uy')

        vhat_mn = self.NLTHelper(vhat_mn, 'v')
        vhatx_mn = self.NLTHelper(vhatx_mn, 'vx')
        vhaty_mn = self.NLTHelper(vhaty_mn, 'vy')

        what_mn = self.NLTHelper(what_mn, 'w')
        whatx_mn = self.NLTHelper(whatx_mn, 'wx')
        whaty_mn = self.NLTHelper(whaty_mn, 'wy')

        # manually compute the convolution
        # get the correct contributions to the nonlinear forcing
        # array is sorted as e.g. 3F, 2F, 1F, 0F, -1F, -2F, -3F

        sumNum_m = (self.numM-1) * 2 - m
        sumNum_n = (self.numN-1) * 2 - n

        # zero out contribution from (0,0) mode in the NLT
        # MFD is modifying the baseflow in the PSE operator

        uhat_mn[self.numM-1, self.numN-1, :] = 0
        vhat_mn[self.numM-1, self.numN-1, :] = 0
        what_mn[self.numM-1, self.numN-1, :] = 0

        uhatx_mn[self.numM-1, self.numN-1, :] = 0
        vhatx_mn[self.numM-1, self.numN-1, :] = 0
        whatx_mn[self.numM-1, self.numN-1, :] = 0

        uhaty_mn[self.numM-1, self.numN-1, :] = 0
        vhaty_mn[self.numM-1, self.numN-1, :] = 0
        whaty_mn[self.numM-1, self.numN-1, :] = 0

        alpha_mn[self.numM-1, self.numN-1, :] = 0
        Ialpha_mn[self.numM-1, self.numN-1, :] = 1.0

        J11 = self.J11[station, :]
        J12 = self.J12[station, :]
        J21 = self.J21[station, :]
        J22 = self.J22[station, :]

        one_over_h = 1.0 / self.h[station, :]
        kappa = self.kappa[station, :]

        if self.config['simulation']['linear']:
            # linear PSE has no forcing
            Fmn = np.zeros((1,4 * self.Ny), dtype=complex)
            pass
        else:
            Fmn = self.NLTConvolution(self.Ny, J11, J12, J21, J22,
                sumNum_m, sumNum_n, self.numM, self.numN,
                uhat_mn, uhatx_mn, uhaty_mn,
                vhat_mn, vhatx_mn, vhaty_mn,
                what_mn, whatx_mn, whaty_mn,
                Ialpha, Ialpha_mn, alpha_mn, beta_mn,
                one_over_h, kappa)

        if m == 0 and n == 0:
            print_rz(f"Imag (0,0) forcing:")
            print_rz(f"Fmn_1 = {np.max(np.abs(np.imag(Fmn[0,0:ny])))}")
            print_rz(f"Fmn_2 = {np.max(np.abs(np.imag(Fmn[0,ny:2*ny])))}")
            print_rz(f"Fmn_3 = {np.max(np.abs(np.imag(Fmn[0,2*ny:3*ny])))}")
            print_rz("\n")

            if np.any(np.isnan(Fmn[0,0:ny])):
                if self.rank == 0:
                    raise ValueError("Warning: NaN or inf values found in Fmn_1")
                self.comm.Abort(1)
            if np.any(np.isnan(Fmn[0,ny:2*ny])):
                if self.rank == 0:
                    raise ValueError("Warning: NaN or inf values found in Fmn_2")
                self.comm.Abort(1)
            if np.any(np.isnan(Fmn[0,2*ny:3*ny])):
                if self.rank == 0:
                    raise ValueError("Warning: NaN or inf values found in Fmn_3")
                self.comm.Abort(1)

        return Fmn

    def solveMeanFlowBlock(self, Dy, Dyy, Uk, Vk, Pk, fkp1, fk, station):
        """
        Solves the mean flow equations using a block matrix approach.
        U dV/dx + VdV/dy + dp/dy - 1/Re d²V/dy² = 0
        U dU/dx + VdU/dy + dp/dx - 1/Re d²U/dy² = 0
        dU/dx + dV/dy = 0

        Parameters:
            Dy, Dyy (array-like): Derivative matrices
            Uk, Vk, Pk (array-like): Known state variables at k
            fkp1, fk (array-like): Forcing terms at k+1 and k
            station (int): Station index
        """
        ny = self.Ny
        Re = self.config['flow']['Re']
        hx = self.hx[station]
        N = 3*ny  # Total system size
        
        # Initialize block matrix and RHS
        A = np.zeros((N, N))
        b = np.zeros(N)
        
        # Common operators
        H = -np.diag(Vk) @ Dy + 1.0/Re * Dyy
        
        # U-momentum blocks (First ny rows)
        A[0:ny, 0:ny] = np.diag(Uk)/hx - 0.5*H  # Coefficient of U_{k+1}
        A[0:ny, 2*ny:3*ny] = np.eye(ny)/hx      # dp/dx term
        b[0:ny] = (np.diag(Uk)/hx + 0.5*H) @ Uk + 0.5*(fkp1[0:ny] + fk[0:ny])
        
        # V-momentum blocks (Second ny rows)
        A[ny:2*ny, ny:2*ny] = np.diag(Uk)/hx - 0.5*H  # Coefficient of V_{k+1}
        A[ny:2*ny, 2*ny:3*ny] = Dy                     # dp/dy term
        b[ny:2*ny] = (np.diag(Uk)/hx + 0.5*H) @ Vk + 0.5*(fkp1[ny:2*ny] + fk[ny:2*ny])
        
        # Continuity blocks (Third ny rows)
        A[2*ny:3*ny, 0:ny] = np.eye(ny)/hx     # dU/dx term
        A[2*ny:3*ny, ny:2*ny] = Dy             # dV/dy term
        b[2*ny:3*ny] = Uk/hx
        
        # U boundary conditions
        # U = 0 at wall
        row = 0
        A[row, :] = 0
        A[row, row] = 1
        b[row] = 0
        
        # U = Uinf at infinity
        row = ny-1
        A[row, :] = 0
        A[row, row] = 1
        if self.config['geometry']['type'] == "import_geom":
            b[row] = self.Baseflow.U[station, -1]
        else:
            b[row] = self.config['flow']["Uinf"]
        
        # V boundary conditions
        # V = 0 at wall
        row = ny
        A[row, :] = 0
        A[row, ny] = 1
        b[row] = 0
        
        # V = 0 at infinity (changed from dV/dy = 0)
        row = 2*ny-1
        A[row, :] = 0
        A[row, 2*ny-1] = 1
        if self.config['geometry']['type'] == 'import_geom':
            b[row] = self.Baseflow.V[station, -1]
        else:
            b[row] = 0
        
        # Pressure boundary conditions
        # p = 0 at wall
        row = 2*ny
        A[row, :] = 0
        A[row, 2*ny] = 1
        b[row] = 0
        
        # dp/dy = 0 at infinity
        row = 3*ny-1
        A[row, :] = 0
        for j in range(ny):
            A[row, 2*ny+j] = Dy[-1, j]
        b[row] = 0
        
        # Add small regularization to pressure diagonal to improve conditioning
        eps = 1e-10
        for i in range(2*ny, 3*ny):
            A[i,i] += eps
        
        # Debug: Check matrix conditioning
        cond = np.linalg.cond(A)
        if cond > 1e15:  # Warning threshold
            print(f"Warning: Matrix condition number is high: {cond}")
            
        try:
            sol = sp.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            print("Linear solver failed. Attempting with pseudo-inverse...")
            sol = np.linalg.lstsq(A, b, rcond=None)[0]
        
        # Extract solutions
        Ubar = sol[0:ny]
        Vbar = sol[ny:2*ny]
        Pbar = sol[2*ny:3*ny]
        Wbar = np.zeros_like(Ubar)
        
        return Ubar, Vbar, Pbar, Wbar

    
    def solveMeanFlow(self, Dy, Dyy, Uk, Vk, Pk, fkp1, fk, station):
        """
        Solves the mean flow equations.

        Parameters:
            Dy, Dyy (array-like): Derivative matrices.
            Uk, Vk, Pk (array-like): Mean flow variables.
            fkp1, fk (array-like): Forcing terms.
            station (int): Station index.
        """

        ny = self.Ny
        Re = self.config['flow']['Re']
        hx = self.hx[station]

        one_over_h = 1 / self.h[station, :]
        kappa = self.kappa[station, 0]

        L1 = np.copy(Dy)
        R1 = 0.5 * (fkp1[ny:2*ny] + fk[ny:2*ny]) 

        np.save("DY_OP.npy", L1)

        #BUG: solution for p at station 1 must be incorrect 
        # comparing the unperturbed base flow and the solution from 
        # solving the BL equations below confirms that 
        # i can forsee the issue coming from 
        # 1) nondimensionalization / scaling
        # 2) boundary conditions 
        # 3) the equations being solved are not valid on this airfoil

        if self.config['geometry']['type'] == "import_geom":

            # set p(0) = 0 
            L1[0, :] = 0
            L1[0, 0] = 1
            # R1[0]    = 0

            R1[0]    = self.Baseflow.P[station, 0]

            # set dp/dy(0)
            dpdy = Dy @ self.Baseflow.P[station, :]

            # R1[0] = dpdy[0]

            # R1[1] = dpdy[1]
            # R1[-1]   = dpdy[-1]

            print_rz(f"Setting BC for dp/eta")
            print_rz(f"dp/deta = {dpdy[-1]}")

            if station == 1:
                plt.figure(figsize=(6,3),dpi=200)
                plt.plot(self.Baseflow.P[station,:], self.ygrid)
                plt.tight_layout()
                plt.savefig("baseflow_p.png")

                plt.figure(figsize=(6,3),dpi=200)
                plt.plot(dpdy, self.ygrid)
                plt.tight_layout()
                plt.savefig("baseflow_dpdeta.png")

        else:

            # set p(0) = 0
            L1[0, :] = 0
            L1[0, 0] = 1
            R1[0]    = 0

            # L1 is Dy 
            # therefore, setting  R1[-1] = 0 is equivalent to setting dp/dy = 0 at y->inf
            R1[-1]    = 0 

        Pbar = sp.linalg.solve(L1, R1)
        dpdx = (Pbar - Pk) / hx

        # Pbar = self.Baseflow.P[station,:]
        # dpdx = (self.Baseflow.P[station,:] - self.Baseflow.P[station-1,:]) / hx

        H = -np.diag(Vk) @ Dy + 1.0/Re * Dyy
        # H = -np.diag(Vk) @ Dy + 1.0/Re * Dyy - np.diag(one_over_h * Vk) * kappa
        L2 = (1. / hx) * np.diag(Uk) - 0.5 * H
        R2 = ((1. / hx) * np.diag(Uk) + 0.5 * H) @ Uk - dpdx + 0.5 * (fkp1[0:ny] + fk[0:ny])

        if self.config['geometry']['type'] == "import_geom":

            L2[0, :] = 0
            L2[0, 0] = 1
            R2[0]    = 0

            # set U(y->inf) = unperturbed rans solution
            L2[-1, :]  = 0
            L2[-1, -1] = 1
            R2[-1]     = self.Baseflow.U[station, -1]
            print_rz(f"Setting BC for Uinf")
            print_rz(f"Uinf = {self.Baseflow.U[station, -1]}")

        else:

            # set U(0) = 0 
            L2[0, :] = 0
            L2[0, 0] = 1
            R2[0]    = 0

            # set U(y->inf) = 1
            L2[-1, :]  = 0
            L2[-1, -1] = 1
            R2[-1]     = self.config['flow']["Uinf"]

        Ubar = sp.linalg.solve(L2, R2)

        L3 = np.copy(Dy)
        # L3 = Dy #+ np.diag(one_over_h) * kappa
        R3 = (-1.0 / hx) * (Ubar - Uk) #* np.diag(one_over_h)

        if self.config['geometry']['type'] == "import_geom":

            # set V(0) = 0
            L3[0, :] = 0
            L3[0, 0] = 1
            R3[0]    = 0

            # set dV/dy = 0 at y->inf
            R3[-1]   = self.Baseflow.Vy[station, -1]
            print_rz(f"Setting BC for dV/deta")
            print_rz(f"dV/deta = {self.Baseflow.Vy[station, -1]}")

            if station == 1:

                plt.figure(figsize=(6,3),dpi=200)
                plt.plot(self.Baseflow.U[station,:], self.ygrid)
                plt.tight_layout()
                plt.savefig("baseflow_u.png")

                plt.figure(figsize=(6,3),dpi=200)
                plt.plot(Ubar, self.ygrid)
                plt.tight_layout()
                plt.savefig("ubar.png")

                # plt.figure(figsize=(6,3),dpi=200)
                # plt.plot(Pbar, self.ygrid)
                # plt.tight_layout()
                # plt.savefig("pbar.png")

                plt.figure(figsize=(6,3),dpi=200)
                plt.plot(self.Baseflow.V[station,:], self.ygrid)
                plt.tight_layout()
                plt.savefig("baseflow_v.png")

                plt.figure(figsize=(6,3),dpi=200)
                plt.plot(self.Baseflow.Vy[station,:], self.ygrid)
                plt.tight_layout()
                plt.savefig("baseflow_vy.png")

        else:

            # set V(0) = 0
            L3[0, :] = 0
            L3[0, 0] = 1
            R3[0]    = 0

            # set dV/dy = 0 at y->inf
            R3[-1]   = 0

        Vbar = sp.linalg.solve(L3, R3)
        Wbar = np.zeros_like(Ubar)

        return Ubar, Vbar, Pbar, Wbar

    def useExternalMeanFlow(self, station):

        # Primary use case for this function is for LPSE with externally generated 
        # mean flow. 
        self.U[:, station] = self.Baseflow.U[station, :]
        self.V[:, station] = self.Baseflow.V[station, :]
        self.W[:, station] = self.Baseflow.W[station, :]

        self.Ux[:, station] = self.Baseflow.Ux[station, :]
        self.Vx[:, station] = self.Baseflow.Vx[station, :]
        self.Wx[:, station] = self.Baseflow.Wx[station, :]

        self.Uy[:, station] = self.Baseflow.Uy[station, :]
        self.Vy[:, station] = self.Baseflow.Vy[station, :]
        self.Wy[:, station] = self.Baseflow.Wy[station, :]

        self.P[:, station] = self.Baseflow.P[station, :]

        return

    def updateMeanFlow(self, helper_mats, station):
        """
        Updates the mean flow.

        Parameters:
            helper_mats (dict): Helper matrices.
            station (int): Station index.
        """

        Re = self.config['flow']['Re']
        Dy = self.Dy
        Dyy = self.Dyy
        ny = self.Ny
        hx = self.hx[station]

        print_rz(f"Station = {station}")

        Uk = np.copy(self.U[:, station - 1])
        Vk = np.copy(self.V[:, station - 1])
        Pk = np.copy(self.P[:, station - 1])

        if self.config['simulation']['linear']:
            # For linear PSE, there can be no RHS forcing
            # multiply by zero just to be sure
            fkp1 = np.real(self.Fmn[station, 0, 0, :]) * 0 
            fk = np.real(self.Fmn[station - 1, 0, 0, :]) * 0 

            Ubar, Vbar, Pbar, Wbar= self.solveMeanFlow(Dy, Dyy, Uk, Vk, Pk, fkp1, fk, station)
            self.U[:, station] = np.copy(Ubar)
            self.V[:, station] = np.copy(Vbar)
            self.P[:, station] = np.copy(Pbar)

            dUbar_dx = (self.U[:, station] - self.U[:, station-1]) / hx
            dVbar_dx = (self.V[:, station] - self.V[:, station-1]) / hx

            self.Ux[:, station] = dUbar_dx
            self.Vx[:, station] = dVbar_dx

            self.Uy[:, station] = Dy @ Ubar
            self.Vy[:, station] = Dy @ Vbar

            return
        
        else:

            # Solving nonlinear PSE here 

            fkp1 = np.real(self.Fmn[station, 0, 0, :]) 
            fk = np.real(self.Fmn[station - 1, 0, 0, :]) 

            imag_error = np.max(np.abs(np.imag(self.Fmn[station,0,0,:])))
            if imag_error >= 1e-12:
                print_rz(f"Error in computing F_00\n")
                print_rz(f"imag error = {imag_error}\n")

            self.U_nlt0[:, station], self.V_nlt0[:, station], self.P_nlt0[:, station], W_nlt0 = self.solveMeanFlow(np.copy(Dy), np.copy(Dyy), \
                self.U_nlt0[:, station-1], self.V_nlt0[:, station-1], self.P_nlt0[:, station-1], fkp1 * 0, fk * 0, station)
            Ubar, Vbar, Pbar, Wbar= self.solveMeanFlow(np.copy(Dy), np.copy(Dyy), Uk, Vk, Pk, fkp1, fk, station)

            MFD_U = Ubar - self.U_nlt0[:, station]
            MFD_V = Vbar - self.V_nlt0[:, station]
            # MFD_W = Wbar - W_nlt0

            self.q[station, 0, 0, 0:ny] = MFD_U
            self.q[station, 0, 0, ny:2*ny] = MFD_V
            # self.q[station, 0, 0, 2*ny:3*ny] = MFD_W

            self.U[:, station] = np.copy(Ubar)
            self.V[:, station] = np.copy(Vbar)
            self.P[:, station] = np.copy(Pbar)

            dUbar_dx = (self.U[:, station] - self.U[:, station-1]) / hx
            dVbar_dx = (self.V[:, station] - self.V[:, station-1]) / hx

            self.Ux[:, station] = dUbar_dx
            self.Vx[:, station] = dVbar_dx

            self.Uy[:, station] = Dy @ Ubar
            self.Vy[:, station] = Dy @ Vbar

        return

    def formOperators(self, helper_mats, station, m, n, s, setup_eigs=False):
        """
        Forms the operators for the NLPSE solver.

        Parameters:
            helper_mats (dict): Helper matrices.
            station (int): Station index.
            m, n, s (int): Mode indices.
        """

        Ny = self.Ny
        hx = self.hx[station]

        # Differencing operators
        Dy = self.Dy
        Dyy = self.Dyy

        U  = np.diag(self.U[:, station])
        Ux = np.diag(self.Ux[:, station])
        Uy = np.diag(self.Uy[:, station])
        V  = np.diag(self.V[:, station])
        Vx = np.diag(self.Vx[:, station])
        Vy = np.diag(self.Vy[:, station])
        W  = np.diag(self.W[:, station])
        Wx = np.diag(self.Wx[:, station])
        Wy = np.diag(self.Wy[:, station])

        kappa = self.kappa[station, 0]

        alpha = self.alpha[station, m, n]
        omega = self.config['disturbance']['omega'] * m
        beta  = self.config['disturbance']['beta'] * n
        Re    = self.config['flow']['Re']

        # If user is running with dpdx stabilization set dpdx term in the 
        # operator to nearly zero
        if self.config['numerical']['dpdx_stabilizer']:
            dpdx_term = 1e-12
        else:
            dpdx_term = 1.0

        # extract helper matrices
        I=helper_mats['I']
        zero=helper_mats['zero']

        one_over_h = np.diag(1.0 / self.h[station, :])
        #HACK:
        # one_over_h = np.copy(I)

        # set imaginary i and delta
        i=1.j

        # Define terms
        J11 = np.diag(self.Grid.J11[station, :])
        J12 = np.diag(self.Grid.J12[station, :])
        J21 = np.diag(self.Grid.J21[station, :])
        J22 = np.diag(self.Grid.J22[station, :])
        
        dJ11_dxi = np.diag(self.Grid.dJ11_dxi[station, :])
        dJ12_dxi = np.diag(self.Grid.dJ12_dxi[station, :])
        dJ11_deta = np.diag(self.Grid.dJ11_deta[station, :])
        dJ12_deta = np.diag(self.Grid.dJ12_deta[station, :])

        dJ21_dxi = np.diag(self.Grid.dJ21_dxi[station, :])
        dJ22_dxi = np.diag(self.Grid.dJ22_dxi[station, :])
        dJ21_deta = np.diag(self.Grid.dJ21_deta[station, :])
        dJ22_deta = np.diag(self.Grid.dJ22_deta[station, :])

        dtheta_dxi = np.diag(self.Grid.dtheta_dxi[station, :])
        d2theta_dxi2 = np.diag(self.Grid.d2theta_dxi2[station, :])

        Q1 = 1.0/Re * (J11 * dJ11_dxi + J12 * dJ12_dxi + J21 * dJ11_deta + J22 * dJ12_deta )

        Q2 = 1.0/Re * ( J11 * dJ21_dxi + J12 * dJ22_dxi + J21 * dJ21_deta + J22 * dJ22_deta )

        V1 = 1.0/Re * dtheta_dxi * ( J11 * J11 + J12 * J12 )

        V2 = 1.0/Re * dtheta_dxi * (J11 * J21  + J12 * J22)

        S1 = 1.0/Re * dtheta_dxi * dtheta_dxi * ( J11 * J11 + J12 * J12 )

        S2 = 1.0/Re * d2theta_dxi2 *  ( J11 * J11 + J12 * J12 ) \
                + 1.0/Re * dtheta_dxi * ( J11 * dJ11_dxi + J12 * dJ12_dxi + J21 * dJ11_deta + J22 * dJ12_deta  )

        H11 = ( J11 * J11 + J12 * J12 )
        H12 = ( J11 * J21 + J12 * J22 )
        H21 = H12
        H22 = ( J21 * J21 + J22 * J22 )

        d12 = -1.0 * one_over_h * dtheta_dxi
        d21 = one_over_h * (Ux - V * dtheta_dxi) + S1
        d22 = Uy - (one_over_h * U) * dtheta_dxi + S2
        d31 = 2 * one_over_h * U * dtheta_dxi + one_over_h * Vx - S2
        d32 = Vy + S1

        delta1 = -i * omega * I + (one_over_h * U - Q1) * i * alpha + 1.0/Re * (H11 * alpha**2 + beta**2 * I)

        delta2 = one_over_h * U - Q1 - 1.0/Re * 2*i*alpha * H11 # general_equations2

        delta3 = V - Q2 - 1.0/Re * 2 * i * alpha * H12
        #BUG:
        # Note sure if this is a bug or what. delta3 in the simplified set of equations has a kappa term that I am missing in this set of equations. Adding it in to see what the effect is. Other than that, I think this equation set is correct. Next step is probably to rerun with 1/h = 1.
        # delta3 = V - 1.0/Re * kappa * one_over_h - Q2 - 1.0/Re * 2 * i * alpha * H12 # used in general_equations5
        # discrepancy bewween delta 3 in this functivs simple function is Q2
        # delta3 = V - 1.0/Re * kappa * one_over_h -  1.0/Re * 2 * i * alpha * H12 # general_equations2
        # delta3 = V - 1.0/Re * kappa * one_over_h # general_equations_1

        A0 = np.block([
            # u         v           w           P
            [delta1 + d21,          2.0*i*alpha*V1 + d22, zero,     one_over_h*i*alpha], # u-mom
            [-2.0*i*alpha*V1 + d31, delta1 + d32,         zero,                   zero], # v-mom
            [zero,                  zero,                 delta1,             i*beta*I], # w-mom
            [i*alpha*one_over_h,    d12,                  i*beta*I,               zero]  # continuity
            ])

        A1 = np.block([
            # u         v           w           P
            [delta2,     2.0*V1, zero,   one_over_h * dpdx_term], # u-mom # multiple one_over_h by 1e-12 to turn off dp/dx term
            [-2.0*V1,    delta2, zero,                     zero], # v-mom
            [zero,       zero,   delta2,                   zero], # w-mom
            [one_over_h, zero,   zero,                     zero]  # continuity
            ])

        A2 = np.block([
            # u         v           w           P
            [delta3 @ Dy,  2.0*V2 @ Dy,  zero,       zero], # u-mom
            [-2.0*V2 @ Dy, delta3 @ Dy, zero,          Dy], # v-mom
            [zero,         zero,        delta3 @ Dy, zero], # w-mom
            [zero,         Dy,          zero,        zero]  # continuity
            ])

        A3 = np.block([
            # u         v           w           P
            [-1./Re * H22 @ Dyy,      zero,         zero,         zero], # u-mom
            [zero,              -1./Re * H22 @ Dyy, zero,         zero], # v-mom
            [zero,              zero,         -1./Re * H22 @ Dyy, zero], # w-mom
            [zero,              zero,         zero,         zero]  # continuity
            ])

        A4 = np.block([
            # u         v           w           P
            [-1./Re * H12 @ Dy,      zero,         zero,         zero], # u-mom
            [zero,              -1./Re * H12 @ Dy, zero,         zero], # v-mom
            [zero,              zero,         -1./Re * H12 @ Dy, zero], # w-mom
            [zero,              zero,         zero,         zero]  # continuity
            ])

        A_s = A0 + A2 + A3 
        A_diff = A1 + A4
        
        # Let's add the option to use BDF2 
        # If we are at the first station, we must use implicit euler
        if self.config['numerical']['method'] == 'BDF2':

            if station >= 4:
                A_solve = (A1 / hx + 2/3 * A_s + s * A_s / hx)
                b = (A1 / hx + s / hx * A_s) @ (4/3 * self.q[station -1, m, n, :] - 1/3 * self.q[station -2, m, n, :])
            else:
                A_solve = A_s + A_diff / hx + s * A_s / hx
                b = ((1.0 / hx) * A_diff + s / hx * A_s) @ self.q[station - 1, m, n, :] 
        else:
            # Use implicit euler
            A_solve = A_s + A_diff / hx + s * A_s / hx
            b = ((1.0 / hx) * A_diff + s / hx * A_s) @ self.q[station - 1, m, n, :] 

        self.A_solve = A_solve
        self.b       = b 

        if setup_eigs:
            # PSE operator presented in Andersson 
            L = -1 * np.linalg.inv(A1) @ A_s
            buf = (np.eye(L.shape[0]) - self.stabilizer * L)
            self.L = np.linalg.inv(buf) @ L 

        return A_solve, b

    def formOperators_simple(self, helper_mats, station, m, n, s, setup_eigs=False):
        """
        Forms the operators for the NLPSE solver.

        Parameters:
            helper_mats (dict): Helper matrices.
            station (int): Station index.
            m, n, s (int): Mode indices.
        """

        Ny = self.Ny
        hx = self.hx[station]

        # Differencing operators
        Dy = self.Dy
        Dyy = self.Dyy

        U  = np.diag(self.U[:, station])
        Ux = np.diag(self.Ux[:, station])
        Uy = np.diag(self.Uy[:, station])
        V  = np.diag(self.V[:, station])
        Vx = np.diag(self.Vx[:, station])
        Vy = np.diag(self.Vy[:, station])
        W  = np.diag(self.W[:, station])
        Wx = np.diag(self.Wx[:, station])
        Wy = np.diag(self.Wy[:, station])

        kappa = self.kappa[station, 0]

        alpha = self.alpha[station, m, n]
        omega = self.config['disturbance']['omega'] * m
        beta  = self.config['disturbance']['beta'] * n
        Re    = self.config['flow']['Re']

        # If user is running with dpdx stabilization set dpdx term in the 
        # operator to nearly zero
        if self.config['numerical']['dpdx_stabilizer']:
            dpdx_term = 1e-12
        else:
            dpdx_term = 1.0

        # extract helper matrices
        I=helper_mats['I']
        zero=helper_mats['zero']

        # one_over_h = np.diag(1.0 / self.h[:, station])
        # TODO 
        one_over_h = np.copy(I)

        # set imaginary i and delta
        i=1.j
        
        delta1 = -i * omega * I + one_over_h * i * alpha * U + 1.0/Re * (alpha**2 * one_over_h**2 + beta**2 * I)
        delta2 = one_over_h * U - 1.0/Re * 2*i*alpha * one_over_h**2
        delta3 = V - 1.0/Re * kappa * one_over_h

        # Governing equation is 
        # A0 phi + A1 phi_xi + A2 phi_eta + A3 phi_eta_eta + A4 phi_eta,xi = F

        A0 = np.block([
            # u         v           w           P
            [delta1 + Ux + kappa*V*one_over_h,           Uy + kappa * one_over_h * U, zero,     one_over_h*i*alpha], # u-mom
            [one_over_h * Vx - 2*one_over_h * kappa * U, delta1 + Vy,                 zero,                   zero], # v-mom
            [zero,                                       zero,                        delta1,             i*beta*I], # w-mom
            [i*alpha*one_over_h,                         kappa * one_over_h,          i*beta*I,               zero]  # continuity
            ])

        A1 = np.block([
            # u         v           w           P
            [delta2,     zero,   zero,   one_over_h * dpdx_term], # u-mom # multiple one_over_h by 1e-12 to turn off dp/dx term
            [zero,       delta2, zero,                         zero], # v-mom
            [zero,       zero,   delta2,                       zero], # w-mom
            [one_over_h, zero,   zero,                         zero]  # continuity
            ])

        A2 = np.block([
            # u         v           w           P
            [delta3 @ Dy, zero,        zero,        zero], # u-mom
            [zero,        delta3 @ Dy, zero,          Dy], # v-mom
            [zero,        zero,        delta3 @ Dy, zero], # w-mom
            [zero,        Dy,          zero,        zero]  # continuity
            ])

        A3 = np.block([
            # u         v           w           P
            [-1./Re * Dyy,      zero,         zero,         zero], # u-mom
            [zero,              -1./Re * Dyy, zero,         zero], # v-mom
            [zero,              zero,         -1./Re * Dyy, zero], # w-mom
            [zero,              zero,         zero,         zero]  # continuity
            ])

        D = A1 
        A_s = A0 + A2 + A3 

        A_solve = A_s + D / hx + s * A_s / hx
        b = ((1.0 / hx) * D + s / hx * A_s) @ self.q[station - 1, m, n, :] 

        self.A_solve = A_solve
        self.b       = b 

        # L = -1 * np.linalg.inv(D) @ A_s
        # self.A_solve = np.eye(L.shape[0]) - hx * L
        # self.b = self.q[station -1, m, n, :]

        if setup_eigs:
            # for eigenvalues:
            # (A0 + A2 + A3) q + A1 dq/dx = 0
            # dqdx = - inv(A1) * (A0 + A3 + A3) q 
            
            # Andersson stabilization 
            # q_x = Lq + sLq_x 
            # (I - sL) q_x = L q 
            # q_x = (I - sL)^(-1) * L q 

            # self.L = -1 * np.linalg.inv(D) * A_s

            # Should triple check this algebra, but this should correspond to the stabilized 
            # PSE operator presented in Andersson 
            L = -1 * np.linalg.inv(D) @ A_s
            buf = (np.eye(L.shape[0]) - self.stabilizer * L)
            self.L = np.linalg.inv(buf) @ L 

        return A_solve, b

    def setBCs(self, m, n, setup_eigs=False):
        """
        Sets the boundary conditions for the given parameters. Dirichelet boundary conditions 
        at the wall and at the freestream for u,w, and p. Neumann boundary condition for v at the freestream.

        Parameters:
        - m (int): The value of m.
        - n (int): The value of n.

        Returns:
        None
        """

        ny = self.Ny

        self.A_solve[(0*ny-0,1*ny-0,2*ny-0),:] = 0.
        self.A_solve[(1*ny-1,2*ny-1,3*ny-1),:] = 0.
        self.b[[0*ny-0,1*ny-0,2*ny-0]] = 0.0
        self.b[[1*ny-1,2*ny-1,3*ny-1]] = 0.0
        # self.A_solve[(0*ny-0,1*ny-0,2*ny-0),(0*ny-0,1*ny-0,2*ny-0)] = 500*1.0j
        # self.A_solve[(1*ny-1,2*ny-1,3*ny-1),(1*ny-1,2*ny-1,3*ny-1)] = 500*1.0j
        self.A_solve[(0*ny-0,1*ny-0,2*ny-0),(0*ny-0,1*ny-0,2*ny-0)] = 1.0
        self.A_solve[(1*ny-1,2*ny-1,3*ny-1),(1*ny-1,2*ny-1,3*ny-1)] = 1.0

        if setup_eigs:
            self.L[(0*ny-0,1*ny-0,2*ny-0),:] = 0.
            self.L[(1*ny-1,2*ny-1,3*ny-1),:] = 0.
            self.L[(0*ny-0,1*ny-0,2*ny-0),(0*ny-0,1*ny-0,2*ny-0)] = 500*1.0j
            self.L[(1*ny-1,2*ny-1,3*ny-1),(1*ny-1,2*ny-1,3*ny-1)] = 500*1.0j

    def setBCsMFD(self, setup_eigs=False):
        """
        Sets mixed boundary conditions:
        - Dirichlet at wall for u,v,w,p
        - Dirichlet at freestream for u,w,p
        - Neumann at freestream for v
        """
        ny = self.Ny
        
        # Wall boundary conditions (unchanged)
        self.A_solve[(0*ny-0,1*ny-0,2*ny-0),:] = 0.
        self.A_solve[(0*ny-0,1*ny-0,2*ny-0),(0*ny-0,1*ny-0,2*ny-0)] = 1.0
        self.b[[0*ny-0,1*ny-0,2*ny-0]] = 0.0
        
        # Freestream boundary conditions
        # Clear existing conditions first
        self.A_solve[(1*ny-1,3*ny-1),:] = 0.
        
        # Set Dirichlet for u,w at freestream
        self.A_solve[1*ny-1,1*ny-1] = 1.0  # u
        self.A_solve[3*ny-1,3*ny-1] = 1.0  # w
        
        # Set corresponding boundary values for u,w
        self.b[[1*ny-1,3*ny-1]] = 0.0

        # lines are commented out because continuity equation is 
        # already enforcing dv/dy=0 in the free stream 
        
        # Set Neumann for v at freestream: dv/dy = 0
        # self.A_solve[2*ny-1,ny:2*ny] = self.Dy[ny-1,:]  # Use last row of Dy matrix

        # set for v
        # self.b[[2*ny-1]] = 0.0

    @staticmethod
    @njit
    def convergeAlpha(hx, alpha, q_old, q_new, relaxation=1):
        """
        Converges the value of alpha based on the current and previous eigenfunctions of the system.

        Parameters:
        - hx (float): The step size.
        - alpha (complex): The current value of alpha.
        - q_old (ndarray): The previous eigenfunction.
        - q_new (ndarray): The current eigenfunction.
        - relaxation (float, optional): The relaxation factor. Defaults to 1.

        Returns:
        - alpha_new (complex): The new value of alpha.
        """
        
        if inner_product(q_new, q_new) == 0.0:
            alpha_new = alpha
        else:
            alpha_new = alpha - relaxation * 1.j / hx * (inner_product(q_new - q_old, q_new) / inner_product(q_new, q_new))

        return alpha_new

    def _initialize_iteration_arrays(self, numModes, numEigs):
        """Initialize arrays needed for non-linear iteration.
        
        Args:
            numModes (int): Number of modes to process
            numEigs (int): Number of eigenvalues to compute
            
        Returns:
            tuple: Initialized arrays for Fr, eigs, alpha, delta_alpha, and q (both local and global)
        """
        modes_per_rank = numModes // self.size
        Fr_local = np.zeros((4*self.Ny), dtype=complex)
        Fr_global = np.zeros((numModes, 4*self.Ny), dtype=complex)
        
        eigs_local = np.zeros((numEigs), dtype=complex)
        eigs_global = np.zeros((numModes, numEigs), dtype=complex)
        
        alpha_local = 0.0j
        alpha_global = np.zeros((numModes), dtype=complex)
        delta_alpha_mn = np.zeros((numModes), dtype=float)
        q_local = np.zeros((4*self.Ny), dtype=complex)
        q_global = np.zeros((numModes, 4*self.Ny), dtype=complex)
        
        return (modes_per_rank, Fr_local, Fr_global, eigs_local, eigs_global, 
                alpha_local, alpha_global, delta_alpha_mn, q_local, q_global)

    def _process_first_iteration(self, station, mode, Fr_local, Fr_global, numModes):
        """Process the first iteration for a given mode.
        
        Args:
            station (int): Current station index
            mode (tuple): Current mode being processed
            Fr_local (np.ndarray): Local forcing term
            Fr_global (np.ndarray): Global forcing term
            numModes (int): Total number of modes
        """
        Fr_local = self.computeNLT(self.helper_mats, station, mode[0], mode[1]) 
        Fr_global = np.asarray(self.comm.allgather(Fr_local))
        
        # need to move Fr into Fmn 
        if self.rank == 0:
            for jj in range(numModes):
                self.Fmn[station, self.harmonics[jj][0], self.harmonics[jj][1], :] = np.copy(Fr_global[jj, :])
        
        # broadcast the forcing to all ranks
        self.Fmn[station, :, :, :] = self.comm.bcast(self.Fmn[station, :, :, :], root=0)

    def _handle_mean_flow_distortion(self, station):
        """Handle the mean flow distortion (MFD) calculation.
        
        Args:
            station (int): Current station index
            
        Returns:
            np.ndarray: Updated q_local array
        """
        if "baseflow" in self.config["flow"].keys():
            if self.config["flow"]["baseflow"] == "external":
                self.useExternalMeanFlow(station)
            else:
                print_rz("Invalid option for baseflow in input file!")
        else:

            #TODO: Do i need to update the presure field in this case?
            # print_rz(f"x for blasius = {self.xgrid[station]}")
            self.Baseflow.Blasius(self.ygrid, x=self.xgrid[station], Uinf = self.config['flow']['Uinf'], nu = self.config['flow']['nu'])
            self.U[:,station] = self.Baseflow.U[0,:]
            self.Uy[:,station] = self.Baseflow.Uy[0,:]
            self.Ux[:,station] = self.Baseflow.Ux[0,:]
            self.V[:,station] = self.Baseflow.V[0,:]
            self.Vy[:,station] = self.Baseflow.Vy[0,:]
            self.Vx[:,station] = self.Baseflow.Vx[0,:]

        if not self.config['simulation']['linear']:

            self.formOperators(self.helper_mats, station, 0, 0, self.stabilizer)
            self.b += self.Fmn[station, 0, 0, :]
            self.setBCsMFD()

            q_local = np.real(sp.linalg.solve(self.A_solve, self.b))

            umfd = self.helper_mats['u_from_SPE'] @ q_local
            vmfd = self.helper_mats['v_from_SPE'] @ q_local

            umfd_dy = self.Dy @ umfd
            vmfd_dy = self.Dy @ vmfd

            qm1 = np.real(self.q[station -1, 0, 0, :])
            umfd_m1 = self.helper_mats['u_from_SPE'] @ qm1
            vmfd_m1 = self.helper_mats['v_from_SPE'] @ qm1

            umfd_dx = (umfd - umfd_m1) / self.hx[station]
            vmfd_dx = (vmfd - vmfd_m1) / self.hx[station]

            self.U[:,station] += umfd
            self.V[:,station] += vmfd

            self.Uy[:,station] += umfd_dy
            self.Vy[:,station] += vmfd_dy

            self.Ux[:,station] += umfd_dx
            self.Vx[:,station] += vmfd_dx

        else:
            q_local = np.zeros_like(self.q[0,0,0,:])

        if self.config['simulation']['linear']:
            print_rz("Finished computing mean flow")
        else:
            print_rz("Finished computing the mean flow distortion")
            
        return q_local

    def _broadcast_flow_variables(self, station):
        """Broadcast flow variables to all ranks.
        
        Args:
            station (int): Current station index
        """
        self.U  = self.comm.bcast(self.U,  root=0)
        self.V  = self.comm.bcast(self.V,  root=0)
        self.P  = self.comm.bcast(self.P,  root=0)
        self.Ux = self.comm.bcast(self.Ux, root=0)
        self.Vx = self.comm.bcast(self.Vx, root=0)
        self.Uy = self.comm.bcast(self.Uy, root=0)
        self.Vy = self.comm.bcast(self.Vy, root=0)
        self.U_nlt0  = self.comm.bcast(self.U_nlt0,  root=0)
        self.V_nlt0  = self.comm.bcast(self.V_nlt0,  root=0)
        self.P_nlt0  = self.comm.bcast(self.P_nlt0,  root=0)
        self.q[station, 0, 0, 0:self.Ny]    = self.comm.bcast(self.q[station, 0, 0, 0:self.Ny], root=0)
        self.q[station, 0, 0, self.Ny:2*self.Ny] = self.comm.bcast(self.q[station, 0, 0, self.Ny:2*self.Ny], root=0)

    def _check_nlt_convergence(self, station, mode, Fr_local, numModes, modes_per_rank, iteration):
        """Check convergence of non-linear terms.
        
        Args:
            station (int): Current station index
            mode (tuple): Current mode being processed
            Fr_local (np.ndarray): Local forcing term
            numModes (int): Total number of modes
            modes_per_rank (int): Number of modes per rank
            iteration (int): Current iteration count
            
        Returns:
            tuple: Boolean indicating convergence and updated Fr_global array
        """
        delta_Fmn = np.zeros((self.harmonics.shape[0]))
        Fr_local = self.computeNLT(self.helper_mats, station, mode[0], mode[1])

        # print_rz(f"Fr local = {Fr_local}")
        # print_rz(f"F old = {self.Fmn[station, 0, 0, :]}")
        
        if np.linalg.norm(Fr_local) == 0.0:
            delta_Fmn_local = 0.0
        else:
            delta_Fmn_local = np.linalg.norm(Fr_local - self.Fmn[station, mode[0], mode[1], :]) / np.linalg.norm(self.Fmn[station, mode[0], mode[1], :])
        
        self.comm.Barrier()
        delta_Fmn = np.asarray(self.comm.allgather(delta_Fmn_local))
        
        if any(value > 1e3 for value in delta_Fmn) and iteration > 5:
            print_rz("PSE is unstable. Killing job.\n")
            sys.exit()
        
        print_rz("ΔFmn = \n")
        print_rz(delta_Fmn)
        print_rz("\n")
        
        converged = np.max(delta_Fmn) < self.config['numerical']['NL_tol']
        
        if converged:
            print_rz("NLTs converged to tolerance")
            print_rz(f"NLTs converged in {iteration} iterations")
        
        return converged, Fr_local

    def _compute_eigenvalues(self, station, mode, numModes, modes_per_rank):
        """Compute eigenvalues if configured.
        
        Args:
            station (int): Current station index
            mode (tuple): Current mode being processed
            numModes (int): Total number of modes
            modes_per_rank (int): Number of modes per rank
            
        Returns:
            np.ndarray: Computed eigenvalues
        """
        eigs_local = np.zeros((4*self.Ny), dtype=complex)
        if not (mode[0] == 0 and mode[1] == 0):
            self.formOperators(self.helper_mats, station, mode[0], mode[1], 
                            self.stabilizer, setup_eigs=self.config['numerical']['compute_eigs'])
            self.setBCs(mode[0], mode[1], setup_eigs=self.config['numerical']['compute_eigs'])
            if self.config['numerical']['compute_eigs']:
                eigs_local = np.linalg.eigvals(self.L)
        return eigs_local

    def NL_iteration(self, converged, station):
        """Performs a non-linear iteration.
        
        Parameters:
            converged (bool): Convergence flag.
            station (int): Station index.
        """
        NL_max_iteration = self.config['numerical']['NL_max_iteration']
        iteration = 0 
        
        numModes = self.harmonics.shape[0]
        numEigs = 4*self.Ny
        
        # Initialize arrays
        (modes_per_rank, Fr_local, Fr_global, eigs_local, eigs_global, 
        alpha_local, alpha_global, delta_alpha_mn, q_local, q_global) = self._initialize_iteration_arrays(numModes, numEigs)
        
        print_rz(f"modes_per_rank = {modes_per_rank}")
        
        # make sure all processes are ready to start
        self.comm.Barrier() 
        
        while not converged:
            iteration += 1 
            
            for idx in range(self.rank * modes_per_rank, (self.rank + 1) * modes_per_rank):
                mode = self.harmonics[idx]
                
                # Handle first iteration
                if iteration == 1:
                    self._process_first_iteration(station, mode, Fr_local, Fr_global, numModes)
                
                # Handle mean flow distortion
                if mode[0] == 0 and mode[1] == 0:
                    # solve the boundary layer equations
                    q_local = self._handle_mean_flow_distortion(station)

                self.comm.Barrier()
                self._broadcast_flow_variables(station)
                
                # Handle non-MFD modes
                delta_alpha_mn_local = 0.0
                if not (mode[0] == 0 and mode[1] == 0):
                    delta_alpha_mn_local = self.alpha_iteration(mode, station)
                    alpha_local = self.alpha[station, mode[0], mode[1]] 
                    q_local = self.q[station, mode[0], mode[1], :]
            
            # Synchronize and gather results
            self.comm.Barrier()
            alpha_global = np.asarray(self.comm.allgather(alpha_local))
            q_global = np.asarray(self.comm.allgather(q_local))
            delta_alpha_mn = np.asarray(self.comm.allgather(delta_alpha_mn_local))
            
            if any(value > self.config['numerical']['alpha_tol'] for value in delta_alpha_mn):
                print_rz("Warning: at least one value of alpha failed to converge. Consider increasing the number of alpha iterations.\n")
            
            # Update global arrays on rank 0
            if self.rank == 0:
                for jj in range(numModes):
                    self.alpha[station, self.harmonics[jj][0], self.harmonics[jj][1]] = np.copy(alpha_global[jj])
                    self.q[station, self.harmonics[jj][0], self.harmonics[jj][1], :] = np.copy(q_global[jj, :])
            
            # Broadcast updated arrays to all ranks
            self.alpha[station, :, :] = self.comm.bcast(self.alpha[station, :, :], root=0)
            self.q[station, :, :, :] = self.comm.bcast(self.q[station, :, :, :], root=0)
            self.comm.Barrier()
            
            if self.config['simulation']['linear']:
                converged = True
                continue
                
            # Check NLT convergence
            print_rz("===========================")
            print_rz("Checking NLT convergence...")
            print_rz("===========================\n")
            
            converged, Fr_local = self._check_nlt_convergence(station, mode, Fr_local, 
                                                            numModes, modes_per_rank, iteration)
            
            if not converged and iteration > NL_max_iteration:
                print_rz("Warning: NLTs did not converge")
                converged = True
            
            # Compute eigenvalues if configured
            if converged and self.config['numerical']['compute_eigs']:
                eigs_local = self._compute_eigenvalues(station, mode, numModes, modes_per_rank)
                eigs_global = np.asarray(self.comm.allgather(eigs_local))
            
            # Gather and update final results
            self.comm.Barrier()
            Fr_global = np.asarray(self.comm.allgather(Fr_local))
            
            if self.rank == 0:
                for jj in range(numModes):
                    self.Fmn[station, self.harmonics[jj][0], self.harmonics[jj][1], :] = np.copy(Fr_global[jj, :])
                    if self.config['numerical']['compute_eigs']:
                        self.opEigs[station, self.harmonics[jj][0], self.harmonics[jj][1], :] = np.copy(eigs_global[jj, :])
            
            # Broadcast the forcing to all ranks
            self.Fmn[station, :, :, :] = self.comm.bcast(self.Fmn[station, :, :, :], root=0)

    def alpha_iteration(self, mode, station):
        """
        Performs an alpha iteration.

        Parameters:
            mode (int): Mode index.
            station (int): Station index.
        """

        alpha_converged = False
        alpha_iterator = 0
        '''
        Note that increasing alpha_max_iteration passed 15~20 can cause
        numerical instability and seriously slow down the code. Either way 
        it shouldn't be necessary because alpha will be iterated on every 
        nonlinear iteration. 
        '''
        alpha_max_iteration = self.config['numerical']['alpha_max_iteration']
        alpha_tol = self.config['numerical']['alpha_tol']

        delta_alpha = 0.0 # initialize for return later
        while not alpha_converged:

            alpha_iterator += 1 

            self.formOperators(self.helper_mats, station, mode[0], mode[1], self.stabilizer)

            # if not linear, add the NLT forcing to the RHS
            if not self.config['simulation']['linear']:
                #BUG: 
                # if BDF2, need to add in correct linear combination of Fmn, Fmn -1, and Fmn -2
                self.b += self.Fmn[station, mode[0], mode[1], :]
            
            self.setBCs(mode[0], mode[1])

            alpha_old = self.alpha[station, mode[0], mode[1]]
            q_old      = np.copy(self.q[station-1, mode[0], mode[1], :])

            disturbance_buf = self.getDisturbance_mn(mode[0], mode[1], self.helper_mats, station - 1, 'u')
            u_old_max = np.max(np.abs(disturbance_buf[-1, :]))

            q_new = sp.linalg.solve(self.A_solve, self.b)

            if np.any(np.isnan(q_new)) or np.any(np.isinf(q_new)):
                raise ValueError("q must not vontain infs or NaNs")
                self.comm.Abort(1)

            # if u_old_max <= 1e-8 or station == 1:
            if u_old_max <= 1e-8:
                self.alpha[station, mode[0], mode[1]] = alpha_old
                self.q[station, mode[0], mode[1], :] = q_new
                # if station == 1:
                #     print("Want all the change to be in q at the inlet for stability\n")
                # else:
                #     print("Mode magnitude below tolerance")
                break

            alpha_new = self.convergeAlpha(self.hx[station], self.alpha[station, mode[0], mode[1]], q_old, q_new, 1.0)
            self.alpha[station, mode[0], mode[1]] = alpha_new
            # self.alpha[station, mode[0], mode[1]] = 1.0j * np.imag(alpha_new)

            delta_alpha = np.abs(alpha_new - alpha_old)
            if delta_alpha <= alpha_tol or alpha_iterator > alpha_max_iteration:
                # print(f"alpha closure iterations = {alpha_iterator}")
                self.alpha[station, mode[0], mode[1]] = alpha_old # to match q at the end point
                self.q[station, mode[0], mode[1], :] = q_new
                # print(f"Wave angle = {np.arctan2(self.Param['beta'] * mode[1], np.real(self.alpha[station, mode[0], mode[1]]))}")
                alpha_converged = True

        return delta_alpha

    def Solve(self, station):
        """
        Solves the NLPSE equations.

        Parameters:
            station (int): Station index.
        """

        # Use LST to initialize the PSE calculation
        if station == 0:

            self.U_nlt0[:, station] = self.Baseflow.U[station, :]

            try:
                idx99 = np.where(self.Baseflow.U[station, :] >= 0.99)[0][0]
                delta99 = self.ygrid[idx99]
                print_rz(f"delta99 = {delta99}")
            except:
                print_rz(f"Couldn't find delta99")

            self.V_nlt0[:, station] = self.Baseflow.V[station,:]
            self.P_nlt0[:, station] = self.Baseflow.P[station,:]

            self.U[:, station] = self.Baseflow.U[station,:]
            self.Ux[:, station] = self.Baseflow.Ux[station,:]
            self.Uy[:, station] = self.Baseflow.Uy[station,:]

            self.V[:, station] = self.Baseflow.V[station,:]
            self.Vx[:, station] = self.Baseflow.Vx[station,:]
            self.Vy[:, station] = self.Baseflow.Vy[station,:]
            self.W[:, station] = self.Baseflow.W[station,:]
            self.Wx[:, station] = self.Baseflow.Wx[station,:]
            self.Wy[:, station] = self.Baseflow.Wy[station,:]
            self.P[:, station] = self.Baseflow.P[station,:]

            print_rz("Initializing the NLPSE calculation with Linear solve\n")

            # for nested dictionaries need to use deep copy 
            # https://stackoverflow.com/questions/2465921/how-to-copy-a-dictionary-and-only-edit-the-copy
            Param_INIT = copy.deepcopy(self.config)

            mode1 = self.config['disturbance']['init_modes'][0]
            mode2 = self.config['disturbance']['init_modes'][1]

            A0 = self.config['disturbance']['amplitudes']['A0']
            A1 = self.config['disturbance']['amplitudes']['A1']

            self.harmonics = self.viableHarmonics(self.config['disturbance']['init_modes'], None, num_repeats=4)
            numModes = self.harmonics.shape[0]
            modes_per_rank = numModes // self.size
            extra_modes = numModes % self.size
            my_start = modes_per_rank * self.rank + min(self.rank, extra_modes)
            my_end = my_start + modes_per_rank + (1 if self.rank < extra_modes else 0)

            alpha_local = []
            q_local = []

            print(f"Rank {self.rank} handling modes {my_start} to {my_end-1}")

            for idx in range(my_start, my_end):
                mode = self.harmonics[idx]
                if mode[0] == 0 and mode[1] == 0:
                    current_alpha = 0.0 + 0.0j
                    current_q = np.zeros((4*self.Ny), dtype=complex)

                elif mode[0] == 0 and mode[1] > 0:

                    # Gortler type instability
                    Param_INIT['disturbance']['omega'] = self.config['disturbance']['omega'] * mode[0]
                    Param_INIT['disturbance']['beta']  = self.config['disturbance']['beta'] * mode[1]

                    if self.Grid.kappa[0,0] == 0:

                        print_rz(f"Initializing mode ({mode[0]}, {mode[1]}) with Orr-Sommerfeld",1)
                        Param_INIT['simulation']['type'] = "LST"
                        INIT_EQS = LST(self.Grid, Param_INIT, self.Baseflow, self.helper_mats)

                    else:

                        print_rz(f"Initializing mode ({mode[0]}, {mode[1]}) with Gortler Flow",1)
                        # print_rz(f"omega = {Param_INIT['disturbance']['omega']}",1) 
                        # print_rz(f"beta = {Param_INIT['disturbance']['beta']}",1) 

                        Param_INIT['simulation']['type'] = 'GORTLER'
                        INIT_EQS = Gortler(self.Grid, Param_INIT, self.Baseflow, self.helper_mats)

                elif mode[0] > 0:

                    Param_INIT['disturbance']['omega'] = self.config['disturbance']['omega'] * mode[0]
                    Param_INIT['disturbance']['beta']  = self.config['disturbance']['beta']  * mode[1]
                    print(f"omega = {Param_INIT['disturbance']['omega']}") 
                    print(f"beta = {Param_INIT['disturbance']['beta']}") 

                    #HACK: 
                    # self.Grid.kappa[0,0] = 0
                    if self.Grid.kappa[0,0] == 0:
                        Param_INIT['simulation']['type'] = "LST"
                        INIT_EQS = LST(self.Grid, Param_INIT, self.Baseflow, self.helper_mats)
                        print(f"Initializing mode ({mode[0]}, {mode[1]}) with LST")
                    else:
                        Param_INIT['simulation']['type'] = 'GORTLER'
                        # INIT_EQS = Gortler(self.Grid, Param_INIT, self.Baseflow, self.helper_mats)
                        # print(f"Initializing mode ({mode[0]}, {mode[1]}) with GORTLER")
                        #HACK:
                        INIT_EQS = LST(self.Grid, Param_INIT, self.Baseflow, self.helper_mats)
                        print(f"Initializing mode ({mode[0]}, {mode[1]}) with LST")

                if mode[0] == 0 and mode[1] == 0:
                    pass
                else:
                    INIT_EQS.Solve(station)

                    current_alpha = INIT_EQS.alpha[station]
                    current_q = INIT_EQS.q[:, station]

                # normalize the disturbance amplitude
                utemp = self.helper_mats['u_from_SPE'] @ current_q
                # norm1 = np.max(np.sqrt(2.0 * utemp * utemp.conj()))
                norm1 = np.max(np.abs(utemp))

                if mode[0] == mode1[0] and mode[1] == mode1[1]:
                    current_q = self.config['flow']["Uinf"] * A0 / np.sqrt(2) * current_q / norm1

                elif mode[0] == mode2[0] and mode[1] == mode2[1]:
                    current_q = self.config['flow']["Uinf"] * A1 / np.sqrt(2) * current_q / norm1

                else:
                    current_q *= 0


                # make alpha exactly imaginary if real part is smaller than some threshold 
                # this is important for modes of the form (0, n)
                # threshold arbitrarily set
                if np.real(current_alpha) <= 1e-8:
                    current_alpha = 1.0j * np.imag(current_alpha)

                alpha_local.append((idx, current_alpha))
                q_local.append((idx, current_q))

            # Synchronize before gathering results
            self.comm.Barrier()

            # Gather results from all ranks
            all_alpha = self.comm.allgather(alpha_local)
            all_q = self.comm.allgather(q_local)

            if self.rank == 0:
                print_rz(f"{numModes} modes in this calculation:\n")
                # First combine all results into arrays
                for rank_results in all_alpha:
                    for idx, alpha in rank_results:
                        print_rz(f"Mode {self.harmonics[idx]}: alpha = {alpha}")
                        self.alpha[station, self.harmonics[idx][0], self.harmonics[idx][1]] = alpha

                for rank_results in all_q:
                    for idx, q in rank_results:
                        self.q[station, self.harmonics[idx][0], self.harmonics[idx][1], :] = q

            # Broadcast the complete arrays from rank 0 to all ranks
            self.alpha[station, :, :] = self.comm.bcast(self.alpha[station, :, :], root=0)
            self.q[station, :, :, :] = self.comm.bcast(self.q[station, :, :, :], root=0)

            print_rz("All ranks have finished initializing the NLPSE calculation\n")

            # Now use NLPSE
        else:

            # self.q[2*self.Ny:3*self.Ny, 0, :, :] = 0.0 # zero out the w equation

            # Nonlinear marching procedure is as follows:
            # Compute nonlinear term and MFD using initial guess. Add this to RHS for PSE
            # For every combination of (m,n), solve the PSE problem until alpha convergence is satisfied. 
            # Recompute the NLT terms. If NLT is converged, finished with the current station. Otherwise,
            # repeat the solution procedure

            if isinstance(self.hx, float):
                self.hx = np.ones((self.Nx)) * self.hx

            print_rz(f"Starting station {station}")

            if self.config['numerical']['stabilizer']:
                if 'hx_factor' in self.config['grid'].keys():
                    alpha_min = self.config['grid']['hx_factor'] / self.hx[station]
                    s = (1 / alpha_min  - self.hx[station]) * 0.5 * 2

                elif 'hx' in self.config['grid'].keys():
                    # s = 0.5 * self.config['grid']['hx']
                    s = 2.0 * self.config['grid']['hx']

                print_rz(f"Stabilizing parameter s = {s}")
                self.stabilizer = s
            else:
                self.stabilizer = 0.0

            if self.config['numerical']['adaptive']:
                # want to choose alpha based on modes that have nonzero temporal wavenumber
                buf = np.abs(np.real(self.alpha[0:station,1:,:]))
                alpha_min = np.min(buf[np.nonzero(buf)])
                self.hx = 1 / alpha_min * self.config["grid"]["hx_factor"] 
                print_rz(f"hx = {self.hx}")

            # initialize alpha and q
            for m in range(self.numM):
                for n in range(self.numN):

                    # MFD initialized as 0
                    if m == 0 and n == 0:
                        self.q[station, m, n, :]  = np.copy(self.q[station - 1, m, n, :])
                        self.alpha[station, m, n] = 0
                    else:
                        self.q[station, m, n, :]  = np.copy(self.q[station - 1, m, n, :])
                        self.alpha[station, m, n] = np.copy(self.alpha[station - 1, m ,n])

            NLT_converged = False

            self.NL_iteration(NLT_converged, station)

    def getDisturbance_mn(self, m, n, helper_mats, station, field):

        if field == 'u':
            extract = helper_mats['u_from_SPE']
        elif field == 'v':
            extract = helper_mats['v_from_SPE']
        elif field == 'w':
            extract = helper_mats['w_from_SPE']
        elif field =='p':
            extract = helper_mats['P_from_SPE']

        disturbance = np.zeros((station+1, self.Ny), dtype = complex)

        if station == 0:
            Ialpha = 1.0
        else:
            Ialpha = cumtrapz(self.alpha[0:station+1, m, n], self.xgrid[0:station+1] - self.xgrid[0], initial=0)
        if m == 0 and n == 0:
            if station == 0:
                # disturbance[0, :] = (extract @ self.q[:, 0, m, n]).T
                disturbance[0, :] = (extract @ self.q[0, m, n, :].T).T
            else:
                # disturbance[:, :] = (extract @ self.q[:, 0:station+1, m, n]).T
                disturbance[:, :] = (extract @ self.q[0:station+1, m, n, :].T).T
        else:
            if station == 0:
                # qPSE = (extract @ self.q[:, 0, m, n]).T * np.exp(1.j * Ialpha) 
                qPSE = (extract @ self.q[0, m, n, :].T).T * np.exp(1.j * Ialpha)
                disturbance[0, :] = qPSE
            else:
                # qPSE = (extract @ self.q[:, 0:station+1, m, n]).T * np.exp(1.j * Ialpha[:, np.newaxis]) 
                qPSE = (extract @ self.q[0:station+1, m, n, :].T).T * np.exp(1.j * Ialpha[:, np.newaxis]) 
                disturbance[:, :] = qPSE

        return disturbance

    def getDisturbance(self, helper_mats, station, field):

        if field == 'u':
            extract = helper_mats['u_from_SPE']
        elif field == 'v':
            extract = helper_mats['v_from_SPE']
        elif field == 'w':
            extract = helper_mats['w_from_SPE']
        elif field =='p':
            extract = helper_mats['P_from_SPE']

        disturbance = np.zeros((self.numM, self.numN, station+1, self.Ny), dtype = complex)
        for m in range(self.numM):
            for n in range(self.numN):
                if station == 0:
                    Ialpha = 0.0
                else:
                    Ialpha = cumtrapz(self.alpha[0:station+1, m, n], self.xgrid[0:station+1] - self.xgrid[0], initial=0)
                if m == 0 and n == 0:
                    if station == 0:
                        disturbance[m, n, 0, :] = (extract @ self.q[0, m, n,:].T).T
                    else:
                        disturbance[m, n, :, :] = (extract @ self.q[0:station+1, m, n, :].T).T
                else:
                    if station == 0:
                        qPSE = (extract @ self.q[0, m, n, :].T).T * np.exp(1.j * Ialpha) 
                        disturbance[m, n, 0, :] = qPSE
                    else:
                        qPSE = (extract @ self.q[0:station+1, m, n, :].T).T * np.exp(1.j * Ialpha[:, np.newaxis]) 
                        disturbance[m, n, :, :] = qPSE

        return disturbance

class LST():
    """
    Class for Linear Stability Theory (LST) solver.
    """

    def __init__(self, Grid, config, Baseflow, helper_mats):
        """
        Initializes the LST solver.

        Parameters:
            Grid (Grid): Grid object containing mesh information.
            config (dict): Configuration dictionary.
            Baseflow (Baseflow): Baseflow object containing base flow information.
            helper_mats (dict): Dictionary containing helper matrices.
        """

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.xgrid     = Grid.xgrid
        self.ygrid     = Grid.ygrid
        self.Nx        = Grid.Nx
        self.Ny        = Grid.Ny

        self.hx        = None

        # Flow params
        self.config    = config
        # Base flow
        self.Baseflow  = Baseflow
        # helper mats 
        self.helper_mats = helper_mats

        # Differentiation
        self.Dy        = Grid.Dy
        self.Dyy       = Grid.Dyy

        # Solution at the current iteration
        self.alpha = np.zeros((self.Nx), dtype=complex)
        self.q     = np.zeros((4*self.Ny, self.Nx), dtype=complex)

        # Operators
        self.L = None
        self.M = None



    def formOperators(self, station):
        """
        Forms the operators for the LST solver.
        """
        # non-parallel LST
        def polyeig2(L0,L1,L2,helper_mats,**kwargs):
            '''Given a polynomial eigenvalue problem up to alpha^2, return a inflated matrices for a general eigenvalue problem
                Given: L0 q + α L1 q + α^2 L2 q = 0
                helper_mats is a dictionary that must contain identity 'I' and matching shape 'zero' matrix for infaltion
                **kwargs are keyword arguments that match the inputs for scipy.linalg.eig(L,b=M,**kwargs)
                Returns: L,M such that L qinf = αM inflated matrix
                [[0  I ]   [[q ]        [[I  0 ]   [[q ]  
                 [L0 L1]]   [αq]]  = α   [0 -L2]]   [αq]] 
            '''
            O = helper_mats['zero']
            I = helper_mats['I']

            L = np.block([
                [O,  I],
                [L0, L1],
            ])
            M = np.block([
                [I,  O],
                [O, -L2]
            ])
            return L, M

        Re=self.config['flow']['Re']
        omega=self.config['disturbance']['omega']
        beta=self.config['disturbance']['beta']

        Dy = self.Dy
        Dyy = self.Dyy

        ny = self.Ny

        U=np.diag(self.Baseflow.U[0,:])
        Ux=np.diag(self.Baseflow.Ux[0,:])
        Uy=np.diag(self.Baseflow.Uy[0,:])
        Uxy=np.diag(Dy@self.Baseflow.Ux[0,:])
        V=np.diag(self.Baseflow.V[0,:])
        Vy=np.diag(self.Baseflow.Vy[0,:]) 
        W=np.diag(self.Baseflow.W[0,:]) * 0
        Wx=np.diag(self.Baseflow.Wx[0,:]) * 0
        Wy=np.diag(self.Baseflow.Wy[0,:]) * 0
        Wxy=np.diag(self.Baseflow.Wxy[0,:]) * 0

        I=self.helper_mats['I']
        zero=self.helper_mats['zero']

        i=1.j

        Delta=i*Re*omega*I + (Dyy-beta**2*I)
        L = np.block([
            # u   v   w   p
            # au  av  aw  ap
            [np.zeros((4*ny,4*ny)),                 np.eye(4*ny)],                     # inflation matrices
            [Delta, -Re*Uy,  zero,   zero,         -i*Re*U, zero,  zero,   -i*Re*I], # u-mom
            [zero,  Delta,    zero,   -Re*Dy,       zero,  -i*Re*U, zero,   zero],    # v-mom
            [zero,  zero,     Delta,  -i*Re*beta*I, zero,  zero,     -i*Re*U, zero],  # w-mom
            [zero,  Dy,       i*beta*I,  zero,      i*I,   zero,     zero,    zero]    # continuity
            ])
        M = np.block([
            # u   v   w   p
            # au  av  aw  ap
            [np.eye(4*ny),                          np.zeros((4*ny,4*ny)),],           # inflation matrices
            [zero,  zero,   zero,   zero,           I,     zero,   zero,   zero],      # u-mom
            [zero,  zero,   zero,   zero,           zero,  I,      zero,   zero],      # v-mom
            [zero,  zero,   zero,   zero,           zero,  zero,   I,      zero],      # w-mom
            [zero,  zero,   zero,   zero,           zero,  zero,   zero,   zero],      # continuity
            ])

        self.L = L
        self.M = M

    def setBCs(self):
        """
        Sets the boundary conditions.
        """
        ny = self.Ny
        self.L[(0*ny-0,1*ny-0,2*ny-0),:] = 0.
        self.L[(1*ny-1,2*ny-1,3*ny-1),:] = 0.
        self.L[(4*ny-0,5*ny-0,6*ny-0),:] = 0.
        self.L[(5*ny-1,6*ny-1,7*ny-1),:] = 0.
        self.M[(0*ny-0,1*ny-0,2*ny-0),:] = 0.
        self.M[(1*ny-1,2*ny-1,3*ny-1),:] = 0.
        self.M[(4*ny-0,5*ny-0,6*ny-0),:] = 0.
        self.M[(5*ny-1,6*ny-1,7*ny-1),:] = 0.
        self.L[(0*ny-0,1*ny-0,2*ny-0),(0*ny-0,1*ny-0,2*ny-0)] = 1.
        self.L[(1*ny-1,2*ny-1,3*ny-1),(1*ny-1,2*ny-1,3*ny-1)] = 1.
        self.L[(4*ny-0,5*ny-0,6*ny-0),(4*ny-0,5*ny-0,6*ny-0)] = 1.
        self.L[(5*ny-1,6*ny-1,7*ny-1),(5*ny-1,6*ny-1,7*ny-1)] = 1.

    def Solve(self, station):
        """
        Solves the LST equations.

        Parameters:
            station (int): Station index.
        """
        # Solve the Spatial Orr-Sommerfeld Problem
        Baseflow = self.Baseflow
        helper_mats = self.helper_mats

        self.formOperators(station)
        self.setBCs()

        eigvals, eigfuncl, eigfuncr = sp.linalg.eig(self.L, b=self.M, left=True)

        idx_to_keep = np.isfinite(eigvals) # remove NaN / infinite eigenvalues due to numerics

        count = 0
        modesLen = eigvals.shape[0]
        modes = np.zeros((modesLen,1),dtype=complex)
        for ii in range(eigvals.size):
            if idx_to_keep[ii]:
                if count == 0:
                    alphas = eigvals[ii]
                    modes[:,count]  = eigfuncr[:,ii]
                    count += 1
                else:
                    alphas = np.append(alphas, eigvals[ii])
                    buf = eigfuncr[:,ii].reshape((modesLen,1))
                    modes = np.append(modes, buf, axis=1)

        blasiusScale = 1.0
        alphas *= blasiusScale
        modes *= blasiusScale
        
        def is_isolated(eigenvalue, all_eigenvalues, radius):
            distances = np.abs(all_eigenvalues - eigenvalue)
            # Remove the distance to itself (which would be 0)
            distances = distances[distances > 0]
            return np.all(distances > radius)

        def find_ts_mode(alphas, blasiusScale):

            # Find the TS mode
            radius = (1.0 / self.config['flow']['Uinf'] * 1.2 * self.config['disturbance']['omega'] * blasiusScale) * 0.01
            alphas_ = alphas[(alphas.imag >= -0.05) & (alphas.real > 1.0 / self.config['flow']['Uinf'] * 1.2 * self.config['disturbance']['omega'] * blasiusScale)]

            if len(alphas_) == 0:
                return None # no eigenvalues meet the initial criteria

            TS_idx_buf  = (alphas_.imag).argmin()
            TS_mode_frequency = alphas_[TS_idx_buf]

            # check if the TS candidate is isolated
            if is_isolated(TS_mode_frequency, alphas, radius):
                TS_idx = np.where(alphas == TS_mode_frequency)[0][0]
                TS_mode = modes[:,TS_idx]
                return TS_idx, TS_mode_frequency, TS_mode

            # if initial mode is not isoalted, look for other isolated candidates 
            isolated_candidates = [alpha for alpha in alphas_ if is_isolated(alpha, alphas, radius)]

            if isolated_candidates:
                # Choose the isolated candidate with the minimum imaginary part
                TS_mode_frequency = min(isolated_candidates, key=lambda x: x.imag)
                TS_idx = np.where(alphas == TS_mode_frequency)[0][0]
                TS_mode = modes[:,TS_idx]
                return TS_idx, TS_mode_frequency, TS_mode

            # if no isolated candidates are found...
            return None

        result = find_ts_mode(alphas, blasiusScale)
        if result:
            TS_idx, TS_mode_frequency, TS_mode = result
        else:
            print("No suitable TS mode found")

        self.alpha[station] = TS_mode_frequency
        self.q[:,station]    = helper_mats['uvwP_from_LST'] @ TS_mode

        return 

class Gortler():
    """
    Class for Gortler flow solver, used to initialize NLPSE calculations on curved surfaces.
    """

    def __init__(self, Grid, config, Baseflow, helper_mats):
        """
        Initializes the Gortler solver.

        Parameters:
            Grid (Grid): Grid object containing mesh information.
            config (dict): Configuration dictionary.
            Baseflow (Baseflow): Baseflow object containing base flow information.
            helper_mats (dict): Helper matrices.
        """

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.xgrid     = Grid.xgrid
        self.ygrid     = Grid.ygrid
        self.Nx        = Grid.Nx
        self.Ny        = Grid.Ny

        self.hx        = None

        self.kappa = Grid.kappa[0,0]

        # Flow params
        self.config     = config
        # Base flow
        self.Baseflow  = Baseflow
        # helper_mats
        self.helper_mats = helper_mats

        # Differentiation
        self.Dy        = Grid.Dy
        self.Dyy       = Grid.Dyy

        # Solution at the current iteration
        self.alpha = np.zeros((self.Nx), dtype=complex)
        self.q     = np.zeros((4*self.Ny, self.Nx), dtype=complex)

        # Operators
        self.L = None
        self.M = None

    def formOperators(self):
        """
        Forms the operators for the Gortler solver.
        """

        Re = self.config['flow']['Re']
        omega = self.config['disturbance']['omega']
        beta = self.config['disturbance']['beta']

        kappa = self.kappa

        Dy = self.Dy
        Dyy = self.Dyy

        U = np.diag(self.Baseflow.U[0, :])
        Uy = np.diag(self.Baseflow.Uy[0, :])

        I = self.helper_mats['I']
        zero = self.helper_mats['zero']

        i = 1.0j
        Delta = -1.0/Re * (Dyy - beta**2 * I)

        L = np.block([
            [-i*omega*I + Delta, Uy,                 zero,               zero    ],
            [-2*kappa*U,         -i*omega*I + Delta, zero,               Dy      ],
            [zero,               zero,               -i*omega*I + Delta, i*beta*I],
            [zero,               Dy,                 i*beta*I,           zero    ]
            ])

        M = np.block([
            [-U*i,  zero, zero, -i*I],
            [zero, -U*i,  zero, zero],
            [zero, zero, -U*i,  zero],
            [-i*I,  zero, zero, zero]
            ])

        self.L = L
        self.M = M

        # save L and M
        # compute eigs in Matlab to see if eigenfunctions 
        # match the pythons result 

        return


    def setBCs(self):
        """
        Sets the boundary conditions.
        """
        ny = self.Ny

        self.L[(0*ny-0, 1*ny-0, 2*ny-0),:] = 0
        self.L[(1*ny-1, 2*ny-1, 3*ny-1),:] = 0

        self.M[(0*ny-0, 1*ny-0, 2*ny-0),:] = 0
        self.M[(1*ny-1, 2*ny-1, 3*ny-1),:] = 0

        self.L[(0*ny-0,1*ny-0,2*ny-0),(0*ny-0,1*ny-0,2*ny-0)] = 1.
        self.L[(1*ny-1,2*ny-1,3*ny-1),(1*ny-1,2*ny-1,3*ny-1)] = 1.

        return

    def Solve(self, station):
        """
        Solves the Gortler equations (spatial formulation).

        Parameters:
            station (int): Station index.
        """

        self.formOperators()
        self.setBCs()

        eigvals, eigfuncr = sp.linalg.eig(self.L, b=self.M, left=False)

        idx_to_keep = np.isfinite(eigvals) # remove NaN / infinite eigenvalues due to numerics

        count = 0
        modesLen = eigvals.shape[0]
        modes = np.zeros((modesLen,1),dtype=complex)
        for ii in range(eigvals.size):
            if idx_to_keep[ii]:
                if count == 0:
                    alphas = eigvals[ii]
                    modes[:,count]  = eigfuncr[:,ii]
                    count += 1
                else:
                    alphas = np.append(alphas, eigvals[ii])
                    buf = eigfuncr[:,ii].reshape((modesLen,1))
                    modes = np.append(modes, buf, axis=1)

        alphas_ = alphas[(alphas.imag >= -0.02)]
        TS_idx_buf = (alphas_.imag).argmin()
        TS_idx = np.where(alphas == alphas_[TS_idx_buf])[0]
        TS_mode_frequency = alphas[TS_idx]
        TS_mode = modes[:,TS_idx]

        self.alpha[station] = TS_mode_frequency
        self.q[:,station]   = TS_mode[:,0]

        return



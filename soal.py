"""Snake on a Leash (SOAL)

Control analysis toolkit
"""

import numpy as np
import os


class Soal(object):
    """The Snake-on-a-Leash base class
"""
    pass

class Poly(Soal):
    """Polynomial class
    p = Poly(C)
    
C is an array-like series of polynomial coefficients.  They must be
ordered so that the polynomial will be
    # p(x) = C[0] + C[1]*x + C[2]*x**2 + C[3]*x**3 + ...
    
There is an optional keyword, "natural", that will reverse the order
to the "natural" order if it is set to "True"

    p = Poly(C, natural=True)
    # p(x) = C[0]x**n + C[1]*x**(n-1) + C[2]*x**(n-2) + ...
    
Once defined, a polynomial can be called like a function
    p = Poly([1,1])
    p(2)    # Returns 3
"""
    def __init__(self, C, natural=False, copyroots=True, rzeros=True):
        
        if isinstance(C,Poly):
            self.coef = np.array(C.coef)
            if copyroots and C._roots is not None:
                self._roots = np.array(C._roots)
            else:
                self._roots = None
        else:
            self.coef = np.array(C,dtype=float)
            if self.coef.ndim != 1:
                self.coef = self.coef.reshape((self.coef.size,))
            if natural:
                self.coef = np.flip(C, 0)
            # Initialize the roots record
            self._roots = None
        if rzeros:
            self._rzeros()
        
        
    def __call__(self, x):
        y = self.coef[-1]
        for c in self.coef[-2::-1]:
            y = c + x*y
        return y
        
    def __repr__(self, x='x'):
        maxline = 74
        out = '\n'
        L2 = '%.3e'%self.coef[0]
        L1 = ' ' * len(L2)
        for ii in range(1,len(self.coef)):
            term = ' + %.3e %c'%(self.coef[ii], x)
            ex = '%d'%ii
            if len(term) + len(ex) + len(L1) > maxline:
                out += L1 + '\n' + L2 + '\n'
                L1 = ''
                L2 = ''
            L1 += ' ' * len(term) + ex
            L2 += term + ' ' * len(ex)
        out += L1 + '\n' + L2 + '\n'
        return out
        
        
    def __len__(self):
        return self.coef.size
        
    def __getitem__(self, ii):
        return self.coef[ii]
        
    def __setitem__(self, ii, value):
        if ii > self.coef.size:
            temp = np.zeros((ii+1,))
            temp[:self.coef.size] = self.coef
            self.coef = temp
        self.coef[ii] = value
        self._roots = None
        self._rzeros()
    
    #
    # Math operators
    #
    def __neg__(self):
        return Poly(-self.coef)
        
    def __add__(self, b):
        """ p1 + p2
"""
        c = Poly(b, copyroots=False)
        if c.coef.size < self.coef.size:
            temp = np.zeros(self.coef.shape)
            temp[:c.coef.size] = c.coef
            c.coef = temp
        c.coef[:self.coef.size] += self.coef
        c._rzeros()
        return c

    def __radd__(self,b):
        return self.__add__(b)

    def __sub__(self, b):
        """ p1 - p2
"""
        c = Poly(b, copyroots=False)
        c.coef = -c.coef
        if c.coef.size < self.coef.size:
            temp = np.zeros(self.coef.shape)
            temp[:c.coef.size] = c.coef
            c.coef = temp
        c.coef[:self.coef.size] += self.coef
        c._rzeros()
        return c
            
    def __rsub__(self,a):
        c = Poly(a, copyroots=False)
        if c.coef.size < self.coef.size:
            temp = np.zeros(self.coef.shape)
            temp[:c.coef.size] = c.coef
            c.coef = temp
        c.coef[:self.coef.size] -= self.coef
        c._rzeros()
        return c
        
    def __mul__(self,b):
        c = Poly(b)
        # If the roots of the multiplicands are available or easily
        # calculated, do so
        if (self._roots is not None or self.coef.size<=3) and \
                (c._roots is not None or c.coef.size<=3):
            if self._roots is None:
                self._roots = self._rexpl()
            if c._roots is None:
                c._roots = c._rexpl()
            c._roots = np.concatenate((self._roots, c._roots))
        # Perform the multiplication
        c.coef = np.convolve(self.coef, c.coef, mode='full')
        return c
        
    def __rmul__(self,a):
        return self.__mul__(a)
        
    def __divmod__(self,b):
        # Initialize the denominator
        if isinstance(b,Poly):
            d = b
        else:
            d = Poly(b)
        # Initialize the remainder
        r = Poly(self)
        # calculate orders
        Np = self.order()   # numerator
        Nd = d.order()      # divisor
        Nq = Np-Nd          # quotient
        Nr = Nd-1           # remainder
        # If the quotient is empty, no need to divide!
        if Nq < 0:
            return Poly([0.]), r
        # Initialize the quotient full of zeros
        q = Poly([0.]*(Nq+1), rzeros=False)
        
        # Loop through quotient indices starting with the highest order
        for qi in range(Nq,-1,-1):
            # The remainder index will be the divisor order plus the quotient index
            ri = qi + Nd
            q.coef[qi] = r.coef[ri] / d.coef[-1]
            # subtract the divisor*quotient from the remainder
            # Note that the leading remainder term will be zero and is
            # ignored.
            r.coef[ri-Nd:ri] -= d.coef[:-1] * q.coef[qi]
        # Truncate off the zero remainder terms
        r.coef = r.coef[:Nd]
        r._rzeros()
        q._rzeros()
        return q,r
        
    def __div__(self, d):
        return self.__divmod__(d)[0]
        
    def __mod__(self, d):
        return self.__divmod__(d)[1]

    def _rzeros(self):
        """
Eliminate "leading zero" coefficients and calculate the roots if the
polynomial is simple enough for explicit root calculation.
"""
        ii = self.coef.size-1
        while ii>0 and self.coef[ii] == 0.:
            ii-=1
        # If leading coefficients are zero
        # Eliminate them and invalidate the roots
        if ii<self.coef.size-1:
            self.coef = self.coef[:ii+1]
            self._roots = None


    def _riter(self, zinit=0., small=1e-6, smallj=1e-6, 
                tiny=1e-15, Nmax=100, verbose=False):
        """Iterative root finding algorithm
    z = p._riter(zinit = 0., small=1e-6, smallj=1e-6 tiny=1e-16, 
                    Nmax=100, verbose=False)

Returns a single root of p using a Mueler iteration variant.

zinit is the initial guess for the zero.
Quadratic iteration is used until the polynomial evaluates to less
than SMALL.  Then, Newton iteration is used until the polynomial
evaluates to less than TINY.
Nmax is the maximum number of polynomial evaluations permitted.
If the imaginary component of the root is less than SMALL, it is
forced to zero.
"""
        count=0
        z = complex(zinit)
        # Start with quadratic iteration for global convergence
        p,dp,ddp = self.d(z,order=2)
        if verbose:
            os.sys.stdout.write(
                "Quadratic iteration...\n" +
                "z %d = %e + %ej\n"%(count, z.real, z.imag) +
                "p(z) = %e + %ej\n"%(p.real, p.imag))
        while np.abs(p) > small and count < Nmax:
            z += 2*p / (-dp - np.sqrt(dp*dp - 2.*p*ddp))
            p,dp,ddp = self.d(z,order=2)
            if verbose:
                os.sys.stdout.write(
                    "z %d = %e + %ej\n"%(count, z.real, z.imag) +
                    "p(z) = %e + %ej\n"%(p.real, p.imag))
            count += 1
        # Polish with Newton iteration
        dz = -p/dp      # First iteration step
        e = np.abs(p)   # Error
        n = 1           # Redundant roots?
        if verbose:
            os.sys.stdout.write("Newton polishing...\n")
        while e > tiny and count < Nmax:
            znew = z + dz       # Tentatively test a new z value
            p,dp = self.d(znew,1)
            if verbose:
                os.sys.stdout.write(
                    "z %d = %e + %ej\n"%(count, znew.real, znew.imag) +
                    "p(z) = %e + %ej\n"%(p.real, p.imag))
            enew = np.abs(p)    # New error
            if enew < e:        # If the error is reduced, keep the new value
                z = znew
                e = enew
                dznew = -p/dp
                # Check for slow convergence due to multiple roots
                # n and nnew are approximations for the number of 
                # redundant roots that the Newton algorithm is 
                # approaching.  If nnew and n are consistent and they
                # are both greater than 1, then accelerate the 
                # convergence appropriately.
                nnew = np.round(np.abs(dz / (dz - dznew)))
                if nnew == n and n>1:
                    if verbose:
                        os.sys.stdout.write(
                            "  Adjusting step by a factor of %d for redundant zeros\n"%n)
                    dz = n * dznew
                    n = 1   # Force n to 1 to prevent this twice in a row.
                else:
                    dz = dznew
                    n = nnew
            else:
                if verbose:
                    os.sys.stdout.write(
                        "  Overshot the solution; back-tracking\n")
                n = 1
                dz /= 2.
            count += 1
                
        if count >= Nmax:
            raise Exception('_RITER: Failed to converge in %d iterations'%Nmax)

        # Test for small imaginary components
        if abs(z.imag) < smallj:
            z = z.real
        return z
        
    def _rexpl(self):
        """Explicitly calculate polynomial roots
This method can be called for polynomials of order 2 or less
    Z = p._rexpl()

Z is an array of roots 0 to 2 elements long.
"""
        if self.coef.size == 1:
            return np.array([])
        elif self.coef.size == 2:
            return np.array([-self.coef[0]/self.coef[1]])
        elif self.coef.size == 3:
            c = self.coef
            temp = c[1]*c[1] - 4*c[2]*c[0]
            # If the result is complex
            if temp<0.:
                temp = 1j * np.sqrt(np.abs(temp)) / 2. / c[2]
                return np.array([temp, -temp]) - c[1]/2./c[2]
            # Now case out which method should be used to minimize numerical error
            elif c[1]<0:
                temp = -c[1] - np.sqrt(temp)
                return np.array([temp/2./c[2], 2.*c[0]/temp])
            else:
                temp = -c[1] + np.sqrt(temp)
                return np.array([temp/2./c[2], 2.*c[0]/temp])
            
        return None

    def order(self):
        """order()
    n = p.order()

Returns the polynomial order, n
"""
        return self.coef.size-1
        
    def d(self,x,order=1):
        """Evaluate the polynomial and its derivatives
    dp = p.d(x)

Returns an array, dp, containing p(x) and the first derivative dpdx

    dp = p.d(x, order=n)

Returns an n-element array, dp, containing p(x) and n of its 
derivatives in order, so that dp[n] is the nth derivative.
"""
        y = [0.]*(order+1)
        for cc in self.coef[-1::-1]:
            for m in range(order,0,-1):
                y[m] = m * y[m-1] + x*y[m]
            y[0] = cc + x*y[0]
        return np.array(y)
        
        
    def roots(self):
        """Calculate the roots (zeros) of the polynomial
    z = p.roots()
Returns an array, z, of the polynomial roots so that p(z)==0
Uses Mueler's method of quadratic root extrapolation for global 
root finding and Newton's method for "polishing"

Once a polynomial's roots have been calculated, they are stored
in the p._roots member.  If _roots is not None, then it will be
returned to prevent redundant iteration.  It is, therefore, 
efficient to make multiple calls to the roots method.
"""
        if self._roots is None:
            _roots = []
            dd = Poly(self)
            while dd.coef.size > 3:
                z = dd._riter()
                _roots.append(z)
                if z.imag:
                    _roots.append(np.conj(z))
                    dd /= [z.imag*z.imag+z.real*z.real, -2*z.real, 1.]
                else:
                    dd /= [-z, 1.]
            if dd.coef.size>1:
                self._roots = np.concatenate((_roots, dd._rexpl()))
            else:
                self._roots = np.array(_roots)
        return self._roots

class Rational(Soal):
    def __init__(self,num,den):
        self.num = Poly(num)
        self.den = Poly(den)
        
    def __call__(self,x):
        return self.num(x) / self.den(x)
        

"""Snake on a Leash (SOAL)

Control analysis toolkit
"""

import numpy as np
import os
import matplotlib.pyplot as plt


class Soal(object):
    """The Snake-on-a-Leash base class
"""
    def tplot(self,t,x, figure=None, xlines=None, tlines=None, *varg, **kwarg):
        """Time plot
    ax = tplot(t,x)
"""
        if figure is None:
            f = plt.figure()
        else:
            f = plt.figure(figure)
        ax = f.add_subplot(111)
        ax.set_ylabel('response')
        ax.set_xlabel('time')

        tmin = np.min(t)
        tmax = np.max(t)
        xmin = np.min(x)
        xmax = np.max(x)
        # Round xmin and xmax
        dx = xmax-xmin
        power = 10**(np.floor(np.log10(dx))-1)
        xmin = np.floor(xmin/power-1)*power
        xmax = np.ceil(xmax/power+1)*power
        
        ax.set_xlim([tmin, tmax])
        ax.set_ylim([xmin, xmax])
        
        if xlines is not None:
            plt.hlines(xlines, xmin=tmin, xmax=tmax, linestyles='dashed', colors='k')
        if tlines is not None:
            plt.vlines(tlines, ymin=xmin, ymax=xmax, linestyles='dashed', colors='k')

        ax.plot(t,x, *varg, **kwarg)
        ax.grid('on')
        plt.show(block=False)
        
        return ax
        
        

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
        
    def __eq__(self, b):
        """ p1 == p2"""
        if isinstance(b,Poly):
            if b.coef.size != self.coef.size:
                return False
            return (self.coef == b.coef).all()
        return self.coef.size==1 and self.coef[0] == b
        
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
        
    def __pow__(self, ex):
        out = Poly(self)
        for ii in range(ex-1):
            out *= self
        return out

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

    def _rsmallj(self, z, pz=None, small=1e-6, verbose=False):
        """Test a complex root to see if it can be truncated to a real root
    z = _rsmallj(z, pz, small=1e-6)
    
z is the root under test
pz is the value of the polynomial evaluated at z or its absolute value.  
    If it is not specified, the polynomial will be evaluated to provide
    a value.  

If the imaginary component is less than SMALL, the imaginary component 
is discarded.  If the new purely real zero evaluates to an error smaller
than the previous complex zero, then the real value is returned.
"""
        if pz is None:
            pz = self(z)
        if abs(z.imag) < small \
                and np.abs(self(z.real)) <= np.abs(pz):
            z = z.real
            if verbose:
                os.sys.stdout.write(
                    "  Truncating to a real root\n")
        return z


    def _riter(self, zinit=0., small=1e-6, smallj=1e-6, 
                tiny=1e-15, Nmax=100, verbose=False):
        """Iterative root finding algorithm
    z = p._riter(zinit = 0., small=1e-6, smallj=1e-6 tiny=1e-15, 
                    Nmax=100, verbose=False)

Returns a single root of p.  Iteration is conducted in three stages:
1) a quadratic iteration is used for global convergence.  The second and
    first derivatives are evaluated using p.d(zguess, 2).  The next 
    guess is taken to be one of the quadratic roots of the polynomial
        f + df*(x-zguess) + 0.5*ddf*(x-zguess)**2
    when f, df, and ddf are the polynomial value and its derivatives.
    This method is used until the absolute value of f is less than
    SMALL.  Similar to Mueler iteration, it provides global convergence.
    
2) a newton algorithm provides root "polishing" until the polynomial
    evaluates to a magnitude less thant TINY.  This stage has two 
    logical checks to prevent accidental divergence and to automatically
    detect slow convergence due to redundant roots.

2a) No guess may evaluate to a polynomial value greater in magnitude 
    than the one prior.  If this event is detected, it means the guess
    has "overshot" the root.  The distance to the guess is halved and 
    the check is repeated until a lower error is found.
    
2b) When Newton iteration approaches redundant roots, the guesses adopt 
    a geometric convergence that slows with the number of local roots.
    For a polynomial,
        p(x) = (x-z)**n * q(x)
    The newton step size will be
        dx = -p / p' =approx= (x-z)/n
    For each step of newton iteration, the number of local roots, n is 
    approximated by comparing dx against the prior dx.  If the 
    approximation for n is stable for two iteration steps in a row, then
    the next step is compensated to place it as close as possible to the
    actual root.
    
    In tests with precisely redundant roots, the algorithm usually 
    converges immediately.  In tests with redundant roots that are only
    quite close, this places the next guess near the center of the root
    cluster, at which points the roots no longer appear identical, and 
    normal newton iteration can resume.

3) Once convergence is detected, the root is "cleaned."  In this stage,
    it is checked for proximity to the real axis.  If the imaginary 
    component is less than SMALLJ, the imaginary component is discarded.
    If the new purely real zero evaluates to an error smaller than the 
    previous complex zero, then the real value is returned.

ZINIT is the initial guess for the zero.
SMALL is the threshold for transition from quadratic to newton iteration
TINY is the threshold used for convergence
NMAX is the maximum number of polynomial evaluations permitted before an
    exception is raised.
SMALLJ is threshold for proximity to the real axis that will trigger 
    testing for truncating to a purely real root.
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
            temp = (-dp - np.sqrt(dp*dp - 2.*p*ddp)) / 2. / p
            # If the point is quite near an inflection, perturb the 
            # root by a random interval and try again
            if np.abs(temp) < small:
                z+=np.random.rand()
            else:
                z += 1./temp
            p,dp,ddp = self.d(z,order=2)
            if verbose:
                os.sys.stdout.write(
                    "z %d = %e + %ej\n"%(count, z.real, z.imag) +
                    "p(z) = %e + %ej\n"%(p.real, p.imag))
            count += 1
            
        # Check for proximity to the real axis
        z = self._rsmallj(z,p,verbose=verbose)
            
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
        z = self._rsmallj(z,e, verbose=verbose)

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
                return np.array([temp,-temp]) - c[1]/2./c[2]
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
        
        
    def roots(self, verbose=False):
        """Calculate the roots (zeros) of the polynomial
    z = p.roots()
Returns an array, z, of the polynomial roots so that p(z)==0
Uses Mueler's method of quadratic root extrapolation for global 
root finding and Newton's method for "polishing"

Roots are always 

Once a polynomial's roots have been calculated, they are stored
in the p._roots member.  If _roots is not None, then it will be
returned to prevent redundant iteration.  It is, therefore, 
efficient to make multiple calls to the roots method.
"""
        if self.coef.size <= 1:
            if verbose:
                os.sys.stdout.write(
                    "Polynomial is only a constant: no roots.\n")
            self._roots = []
            return self._roots
        if self._roots is None:
            if verbose:
                os.sys.stdout.write(
                    "Found no root history: calculating roots from scratch.\n")
            _roots = []
            # Create a deflatable polynomial
            dd = Poly(self)
            dd._rzeros()

            if verbose:
                os.sys.stdout.write(
                    "Creating a deflatable polynomial:%s"%repr(dd))

            # Remove any roots at the origin
            for ii in range(self.coef.size):
                if self.coef[ii]!=0.:
                    break
            if ii>0:
                _roots += [0.]*ii
                dd.coef = dd.coef[ii:]
                    
                if verbose:
                    os.sys.stdout.write(
                        "Found %d roots at the origin... deflating.\n"%ii)

            # If there are no roots left, halt
            if dd.coef.size<=1:
                if verbose:
                    os.sys.stdout.write(
                        "All roots were at the origin...[done]\n")
                self._roots = _roots
                return self._roots

            # Now that the constant term is definitely non-zero
            # Re-scale x so that the highest-order coefficient and the 
            # constant term are both unity.
            dd.coef /= dd.coef[0]
            scale = dd.coef[-1] ** (-1. / (dd.coef.size-1))
            temp = scale
            for ii in range(1,dd.coef.size):
                dd.coef[ii] *= temp
                temp *= scale
                
            if verbose:
                os.sys.stdout.write(
                    "Rescaling the deflatable polynomial:%s"%repr(dd))
                
            # Start iteration
            z = 0.
            while dd.coef.size > 3:
                # Use the root from the last solution as the initial 
                # guess so that redundant roots will be quickly
                # identified and will be in sequence to one another
                z = dd._riter(zinit=z)
                _roots.append(z)
                
                if verbose:
                    os.sys.stdout.write(
                        "Found root using _riter():\n  %s\n"%repr(z))
                
                if z.imag:
                    _roots.append(np.conj(z))
                    dd /= [z.imag*z.imag+z.real*z.real, -2*z.real, 1.]
                    if verbose:
                        os.sys.stdout.write(
                            "Root was complex: adding the conjugate.\n")
                else:
                    dd /= [-z, 1.]
                if verbose:
                    os.sys.stdout.write(
                        "Deflating the polynomial:%s"%repr(dd))
            if dd.coef.size>1:
                z = dd._rexpl()
                if verbose:
                    os.sys.stdout.write(
                        "Polynomial was deflated to permit explicit root solution:\n")
                for zz in z:
                    os.sys.stdout.write("  %s\n"%repr(zz))
                self._roots = np.concatenate((_roots, z))
            else:
                self._roots = np.array(_roots)
            
            self._roots *= scale
            
            if verbose:
                os.sys.stdout.write(
                    "Rescaling the roots:\n%s\n"%repr(self._roots))
        elif verbose:
            os.sys.stdout.write(
                "Found a root record: returning\n" + 
                "To clear, set the _roots member to None\n")
        return self._roots
        
        
    def get_coef(self,k):
        """Get coefficient
    ck = p.get_coef(k)

Retrieves a the polynomial coefficient, k, so that
    p(x) = c0 + c1 x + c2 x**2 + ... ck x**k ... + cn x**n
    
This is different from p.coef[k] in that if k is greater than n,
0. is returned rather than throwing an error.
"""
        if k>=self.coef.size:
            return 0.
        elif k<0:
            raise Exception("GET_COEF: Received negative coefficient index, %s."%repr(k))
        return self.coef[k]


class Rational(Soal):
    """Rational class
    r = Rational(num,den)
    
Defines a rational function, r, comprised of a polynomial numerator, 
num, and denominator, den.  They may be polynomials or array-like 
coefficient arrays, and they will be stored as Poly instances at r.num 
and r.den.

The rational class uses the same function-like call implementation
    r(x)
to evaluate the rational function.  It also supports evaluation of the 
first rational derivative
    r.d(x)
Higher derivatives like those supported by Poly are not yet implemented.
"""
    def __init__(self,num,den=None, natural=False):
        if issubclass(type(num), Rational):
            if den is not None:
                raise Exception("Rational: Cannot simultaneously copy a rational and define a denominator.")
            self.den = Poly(num.den)
            self.num = Poly(num.num)
        elif den is None:
            self.num = Poly(num,natural=natural)
            self.den = Poly(1)
        else:
            self.den = Poly(den,natural=natural)
            self.num = Poly(num,natural=natural)
        
    def __call__(self,x):
        return self.num(x) / self.den(x)
        
    def __repr__(self, x='x'):
        N = self.num.__repr__(x)
        D = self.den.__repr__(x)
        width = 0
        line = 0
        for cc in N:
            line += 1
            if cc == '\n':
                width = max(width,line)
                line = 0
        line = 0
        for cc in D:
            line += 1
            if cc == '\n':
                width = max(width,line)
                line = 0
        return N + '-' * width + D
        
    def __neg__(self):
        return self.__class__(-self.num, self.den)
        
    def __add__(self, b):
        c = self.__class__(b)
        if c.den == self.den:
            c.num += self.num
        else:
            c.num *= self.den
            c.num += self.num * c.den
            c.den *= self.den
        return c
        
    def __radd__(self, b):
        return self.__add__(b)
        
    def __sub__(self, b):
        c = -self.__class__(b)
        if c.den == self.den:
            c.num += self.num
        else:
            c.num *= self.den
            c.num += self.num * c.den
            c.den *= self.den
        return c
        
    def __rsub__(self,b):
        c = self.__class__(b)
        if c.den == self.den:
            c.num -= self.num
        else:
            c.num *= self.den
            c.num -= self.num * c.den
            c.den *= self.den
        return c
        
    def __mul__(self, b):
        c = self.__class__(b)
        c.num *= self.num
        c.den *= self.den
        return c
        
    def __rmul__(self, b):
        return self.__mul__(b)
        
    def __div__(self, b):
        c = self.__class__(b)
        temp = c.num
        c.num = c.den
        c.den = temp
        c.num *= self.num
        c.den *= self.den
        return c

    def __rdiv__(self, b):
        c = self.__class__(b)
        c.num *= self.den
        c.den *= self.num
        return c

    def order(self):
        """System order
    N = G.order()

N is the number of system poles.
"""
        return self.den.order()
        
    def poles(self):
        """Return the roots of the denomenator
    p = r.poles()
    
Uses the denomenator's roots() method, so redundant calls to poles() are
efficient.

See also: zeros()
"""
        return self.den.roots()
        
    def zeros(self):
        """Return the roots of the numerator
    z = r.zeros()
    
Uses the numerator's roots() method, so redundant calls to zeros() are 
efficient.

See also: poles()
"""
        return self.num.roots()
        
        
    def sub(self, x):
        """Substitute a rational into the rational
    r1 = soal.Rational( ... )
    r2 = soal.Rational( ... )
    r3 = r2.sub(r1)
    
r3 will be a new rational with coefficients calculated by substituting 
r1 into r2.
"""
        if not issubclass(type(x),Rational):
            raise Exception('SUB: The substituted value must be a rational')
        
        N = max(self.num.coef.size, self.den.coef.size)
        ncoef = np.zeros((N,))
        dcoef = np.zeros((N,))
        ncoef[:self.num.coef.size] = self.num.coef
        dcoef[:self.den.coef.size] = self.den.coef
        term = x.den ** (N-1)
        out = Rational(0,0)
        for ii in range(N):
            out.num += ncoef[ii] * term
            out.den += dcoef[ii] * term
            term /= x.den
            term *= x.num
        return out
                
        
    def d(self,x):
        n,dn = self.num.d(x)
        d,dd = self.den.d(x)
        return (dn - n*dd/d)/d


    def expand(self):
        """Perform a partial-fraction expansion
    [g1, g2, g3 ... ] = G.expand()

Where gx is a rational with one real pole of G or two complex conjugate
poles of G.  The sum of the g's is equal to G.
"""
        g = []
        for z in self.poles():
            if z.imag<0:    # Ignore complex conjugates
                pass
            else:
                A = self.num(z) / self.den.coef[-1]
                redundant = False
                for zz in self.poles():
                    if z!=zz:
                        A /= (z-zz)
                    elif redundant:
                        raise Exception("EXPAND: Failed with redundant poles.")
                    else:
                        redundant = True
                if z.imag:
                    gnew = Continuous(
                        [-2*(A*z.conj()).real, 2.*A.real],
                        [(z*z.conj()).real, -2*z.real, 1.])
                    gnew._roots = np.array([z, z.conj()])
                else:
                    gnew = Continuous(
                        [A], [-z, 1.])
                    gnew._roots = np.array([z])
                g.append(gnew)
        return g


class Continuous(Rational):
    
    def __repr__(self):
        return Rational.__repr__(self,'s')


    def _tgen(self, Nmax=1000):
        """Generate a time array that will capture the most relevant aspects of
the system's response
    t = G._tgen(Nmax=1000)
    
Nmax is the maximum number of points allowed
"""
        small = 1e-6
        tfast = float('inf')
        tslow = 0.
        for z in self.poles():
            T = np.sqrt(z.real*z.real + (z.imag/2/np.pi)**2)
            if T > small:
                T = 1./T
                tfast = min(tfast, T)
                tslow = max(tslow, T)
        if tslow == 0. or tfast == float('inf'):
            tfast = 1.
            tslow = 1.
        dt = tfast / 10.
        tmax = tslow * 10.
        dt = max(dt, tmax / Nmax)
        return np.arange(0.,tmax, dt)
        

    def time(self, t=None):
        """Revert to the time domain
    t,x = G.time()
        OR
    t,x = G.time(t=[t0, t1, ... ])
    
This method is only available on proper transfer functions of order 
greater than 0 and less than 3.
"""
        if t is None:
            dt,tmax = self.timescales()
            dt = max(dt, 1e-2*tmax)
            t = np.arange(0.,10.*tmax,.1*dt)
                
        if self.order()==2:
            sigma = self.poles()[0].real
            omega = np.abs(self.poles()[0].imag)
            # Cosine coefficient
            A = self.num.get_coef(1)
            # Sine coefficient
            B = (self.num.get_coef(0) + A*sigma)/omega
            x = np.exp(sigma*t) * (A * np.cos(omega*t) + B * np.sin(omega*t))
        elif self.order()==1:
            sigma = self.poles()[0]
            A = self.num.get_coef(0)
            if sigma != 0.:
                x = A * np.exp(sigma*t)
            else:
                x = A * np.ones(t.shape)
        else:
            raise Exception("TIME: Order %d cannot be converted to the time domain"%self.order())
        return t,x
        
        
    def state(self):
        """Return state matrices representing the system
    A,B,C,D = c.state()

Where
    xdot = Ax + Bu
    y = Cx + Du
"""
        N = self.den.order()
        if N==0 or self.num.order()>=N:
            raise Exception("STATE: The transfer function is improper; cannot produce a state space realization.")
        A = np.matrix(np.zeros((N,N)))
        B = np.matrix(np.zeros((N,1)))
        C = np.matrix(np.zeros((1,N)))
        D = np.matrix(np.zeros((1,1)))
        A[N-1,:] = -self.den.coef[:-1]/self.den.coef[-1]
        for ii in range(N-1):
            A[ii,ii+1] = 1.
        B[N-1,0] = 1
        C[0,:self.num.coef.size] = self.num.coef / self.den.coef[-1]
        return A,B,C,D
        
        

                
        
    def step(self, t=None, figure=None, verbose=True, coef=False):
        """Calculate the step response of the system
    t,x = G.step()
    
t and x are arrays of time and the system's state variable, x.
"""
        gg = (self*Continuous([1],[0,1])).expand()
        if t is None:
            t = self._tgen()
        x = np.zeros(t.shape)
        for g in gg:
            _,xx = g.time(t=t)
            x += xx
        if verbose:
            if self.den.get_coef(0)==0.:
                xss = None
            else:
                xss = self.num.get_coef(0) / self.den.get_coef(0)
            self.tplot(t,x,figure=figure,xlines=xss)
        return t,x

    def impulse(self, t=None, figure=None, verbose=True, coef=False):
        """Calculate the impulse response of the system
    t,x = G.impulse()
    
t and x are arrays of time and the system's state variable, x.
"""
        gg = self.expand()
        if t is None:
            t = self._tgen()
        x = np.zeros(t.shape)
        for g in gg:
            _,xx = g.time(t=t)
            x += xx
        if verbose:
            if self.den.get_coef(0)==0.:
                if self.den.get_coef(1) == 0.:
                    xss = None
                else:
                    xss = self.num.get_coef(0)/self.den.get_coef(1)
            else:
                xss = 0.
            self.tplot(t,x,figure=figure,xlines=xss)
        return t,x



class Discrete(Rational):

    def __init__(self, num, den=None, T=1, *arg, **kwarg):
        Rational.__init__(self,num,den, *arg, **kwarg)
        self.T = T
        self._xhist = np.zeros((self.den.coef.size-1,))
        self._uhist = np.zeros((self.num.coef.size-1,))

    def __repr__(self):
        return Rational.__repr__(self,'z')

    def state(self):
        """Return state matrices representing the system
    A,B,C,D = c.state()

Where
    xk+1 = A xk + B uk+1
    yk+1 = C xk+1 + D uk+1
"""
        N = self.den.order()
        if N==0 or self.num.order()>=N:
            raise Exception("STATE: The transfer function is improper; cannot produce a state space realization.")
        A = np.matrix(np.zeros((N,N)))
        B = np.matrix(np.zeros((N,1)))
        C = np.matrix(np.zeros((1,N)))
        D = np.matrix(np.zeros((1,1)))
        A[N-1,:] = -self.den.coef[:-1]/self.den.coef[-1]
        for ii in range(N-1):
            A[ii,ii+1] = 1.
        B[N-1,0] = 1
        C[0,:self.num.coef.size] = self.num.coef / self.den.coef[-1]
        return A,B,C,D
        
    def sim(self, uk=0., xinit=None, uinit=None):
        """Calculate the next sample of the system respone to an input uk
    xk = G.sim(uk=0.)
    
To initialize a new simulation, use the optional xinit and uinit 
arguments.

    xk = G.sim(uk, xinit=[xk-1, xk-2, ...], uinit=[uk-1, uk-2, ...])
    
The length of xinit must be the same as the denominator order.  The 
length of uinit must be the same as the numerator order.
"""
        if xinit is not None:
            xinit = np.array(xinit)
            if xinit.size != self.den.coef.size-1:
                raise Exception("SIM: xinit must have length den.order()")
            self._xhist = xinit
        if uinit is not None:
            uinit = np.array(uinit)
            if uinit.size != self.num.coef.size-1:
                raise Exception("SIM: uinit must have length num.order()")
            self._uhist = uinit
        xk = uk * self.num.coef[-1] + np.dot(self.num.coef[:-1], self._uhist)
        xk += -np.dot(self.den.coef[:-1], self._xhist) / self.den.coef[-1]
        # Rotate in the new values
        if self.den.order():
            self._xhist = np.roll(self._xhist,-1)
            self._xhist[-1] = xk
        if self.num.order():
            self._uhist = np.roll(self._uhist,-1)
            self._uhist[-1] = uk
        return xk
        
    def impulse(self, Nmax=None, verbose=True, figure=None):
        """System step response
"""
        if Nmax is None:
            # Auto-detect the sample count
            Nmax = 20
            for z in self.poles():
                if z!=1.:
                    z = np.log(z)
                    sigma = np.abs(z.real)
                    omega = np.abs(z.imag)
                    if sigma > 1e-6:
                        Nmax = max(Nmax,
                            int(10/sigma))
                    if omega > 1e-6:
                        Nmax = max(Nmax,
                            int(4*np.pi/omega))
        
        x = np.zeros((Nmax,))
        x[0] = self.sim(1./self.T, 
                xinit = np.zeros((self.den.order(),)),
                uinit = np.zeros((self.num.order(),)))
        for k in range(1,Nmax):
            x[k] = self.sim(0.)
            
        if verbose:
            if figure is None:
                figure=1
            ax = plt.figure(figure).get_axes()
            if ax:
                ax = ax[0]
            else:
                ax = plt.figure(figure).add_subplot(111)
            t = np.arange(0., self.T * x.size, self.T)
            ax.plot(t,x)
            ax.set_xlabel('Time')
            ax.set_ylabel('X')
            ax.grid('on')
            plt.show(block=False)
            
        return x

        
    def step(self, Nmax=None, verbose=True, figure=None):
        """System step response
"""
        if Nmax is None:
            # Auto-detect the sample count
            Nmax = 20
            for z in self.poles():
                if z!=1.:
                    z = np.log(z)
                    sigma = np.abs(z.real)
                    omega = np.abs(z.imag)
                    if sigma > 1e-6:
                        Nmax = max(Nmax,
                            int(10/sigma))
                    if omega > 1e-6:
                        Nmax = max(Nmax,
                            int(4*np.pi/omega))
        
        x = np.zeros((Nmax,))
        x[0] = self.sim(1.,
                xinit = np.zeros((self.den.order(),)),
                uinit = np.zeros((self.num.order(),)))
        for k in range(1,Nmax):
            x[k] = self.sim(1.)
            
        if verbose:
            if figure is None:
                figure=1
            ax = plt.figure(figure).get_axes()
            if ax:
                ax = ax[0]
            else:
                ax = plt.figure(figure).add_subplot(111)
            t = np.arange(0., self.T * x.size, self.T)
            ax.plot(t,x)
            ax.set_xlabel('Time')
            ax.set_ylabel('X')
            ax.grid('on')
            plt.show(block=False)
            
        return x

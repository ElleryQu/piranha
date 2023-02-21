import random
import math

q = 7340033         # 23bit
l = 20
bound = 2 ** l

def divideC(x, d):
    # r = random.randint(0, q-1)
    r = q - 1
    z = (r + x) % q
    e0 = wrap(r, z)
    e1 = int(r%d <= z%d)    # 1 - e1
    res = int(z/d) - int(r/d) + e0 * int(q/d) + e1 - 1
    
    nr = r % d
    nx = x % d 
    nq = q % d
    nz = z % d
    print(
        
f'''------------------------------------
Test Zone:
{e0=}, {e1=}
{nr=}, {nx=}, {nq=}, {nz=}: {int((nr+nx+nq)/d)} vs off1: {1-e1}
divideC output: {res}, but the true value is {int(x/bound)}
...what about divideC in R?\t\t{divideC_R(x-1, d)}
---------------------------------------''')
    return res

def divideC_R(x, d):
    # r = random.randint(0, q-1)
    r = q - 1
    z = r + x 
    e0 = wrap(r, z)
    e1 = int(r%d <= z%d)    # 1 - e1
    res = int(z/d) - int(r/d) + e0 * int(q/d) + e1 - 1
    
    return res
    
def wrap(r, z):
    return int(z < (q-1)/2)*int(r > (q-1)/2)

x = bound - 1

divideC(x, bound)

x = bound
divideC(x, bound)
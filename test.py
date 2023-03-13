import random

p = 2138816513
precision = 8

def random_p(x, p=p):
    return random.randint(0, p)

def flp2fxp(Rx, precision=precision):
    return int(Rx * (1 << precision))

def fxp2flp(rx, precision=precision):
    return rx / (1 << precision)

def get_share(x, p=p):
    x0 = random_p(x, p)
    x1 = (x - x0) % p
    return x0, x1

def fxp_to_fp(rx, p=p):
    return rx % p

def fp_to_fxp(x, p=p):
    return x - (x//((p+1)/2)) * p

def divide_d(x, d, p=p):
    return (x//d - (x//((p+1)/2)) * p//d + p) % p


Rx, Ry = -0b10, 0b11
rx, ry = flp2fxp(Rx), flp2fxp(Ry)

x, y = fxp_to_fp(rx), fxp_to_fp(ry)
# x0, y0 = random_p(x), random_p(y)
# x1, y1 = (x - x0) % p, (y - y0) % p
xy = (x * y) % p
rxy_ = fp_to_fxp(xy)
print(f"{rx*ry} -> {xy} -> {rxy_} -> {fxp2flp(rxy_)} vs {rx*ry/(1 << precision)}")

d = int(1 << precision)
epsilon = 1e-2
tr, fa = 0, 0
games = 10000
for _ in range(games):
    xy0, xy1 = get_share(xy)
    xy0dd, xy1dd = divide_d(xy0, d), divide_d(xy1, d)
    xydd = (xy0dd + xy1dd) % p
    # xydd = divide_d(xy, d)
    rxydd_ = fp_to_fxp(xydd)
    if (Rx*Ry-fxp2flp(rxydd_))**2 > epsilon:
        tr += 1
    else:
        fa += 1
# print(f"{xy0=}, {xy1=}")
# print(f"x//d = {xydd} -> {rxydd_} -> {fxp2flp(rxydd_)} vs {Rx*Ry}")
print(f"tr: {tr},\tfa: {fa}, rate: {tr/games}")
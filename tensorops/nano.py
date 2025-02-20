class N:
    c = None

    def __init__(s, v, rg=1):
        s.v = float(v)
        s.g = 0
        s.p = []
        s.c = []
        s.rg = rg
        NC.c and NC.c.add(s)

    def __sub__(s, o):
        return S(s, o if isinstance(o, N) else N(o))

    def __pow__(s, o):
        return P(s, o if isinstance(o, N) else N(o))

    def __add__(s, o):
        return A(s, o if isinstance(o, N) else N(o))

    def __mul__(s, o):
        return M(s, o if isinstance(o, N) else N(o))

    def co(s):
        pass

    def gg(s):
        pass

    def zg(s):
        s.g = 0


class NC:
    c = None

    def __init__(s):
        s.n = []

    def __enter__(s):
        s.p = NC.c
        NC.c = s
        return s

    def __exit__(s, et, ev, tb):
        NC.c = s.p

    def add(s, n):
        s.n.append(n)


class S(N):
    def __init__(s, n1, n2):
        super().__init__(0)
        s.n1 = n1
        s.n2 = n2
        s.p = [n1, n2]
        [p.c.append(s) for p in s.p]

    def co(s):
        s.v = s.n1.v - s.n2.v

    def gg(s):
        s.n1.g += s.g
        s.n2.g -= s.g


class P(N):
    def __init__(s, n1, n2):
        super().__init__(0)
        s.n1 = n1
        s.n2 = n2
        s.p = [n1, n2]
        [p.c.append(s) for p in s.p]

    def co(s):
        s.v = s.n1.v**s.n2.v

    def gg(s):
        s.n1.g += s.g * s.n2.v * (s.n1.v ** (s.n2.v - 1))


class A(N):
    def __init__(s, n1, n2):
        super().__init__(0)
        s.n1 = n1
        s.n2 = n2
        s.p = [n1, n2]
        [p.c.append(s) for p in s.p]

    def co(s):
        s.v = s.n1.v + s.n2.v

    def gg(s):
        s.n1.g += s.g
        s.n2.g += s.g


class M(N):
    def __init__(s, n1, n2):
        super().__init__(0)
        s.n1 = n1
        s.n2 = n2
        s.p = [n1, n2]
        [p.c.append(s) for p in s.p]

    def co(s):
        s.v = s.n1.v * s.n2.v

    def gg(s):
        s.n1.g += s.g * s.n2.v
        s.n2.g += s.g * s.n1.v


def f(ns):
    [n.co() for n in ns]


def b(ns):
    [n.gg() for n in ns[::-1] if n.rg]


def z(ns):
    [n.zg() for n in ns[::-1]]


with NC() as n:
    t, m, x, c = N(0.5), N(1), N(2, rg=False), N(3)
    y = m * x + c
    l = (y - t) ** N(2)
    f(n.n)
    for step in range(10):
        f(n.n)
        l.g = 1
        b(n.n)
        lr = 0.05
        m.v -= lr * m.g
        c.v -= lr * c.g
        z(n.n)
        print(f"loss:{l.v:.3f}")
print(f"m:{m.v:.3f} c:{c.v:.3f} loss:{l.v:.3f}")

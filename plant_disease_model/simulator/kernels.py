
class Kernel:
    def __init__(self, kernel, name):
        self.kernel = kernel
        self.name = name


def get_kernel_from_name(name):
    id_string = name.split('_')[0]
    if id_string == "cauchy":
        kernel = get_cauchy_kernel_from_string(name)
    elif id_string == "power":
        kernel = get_power_law_kernel_from_string(name)
    elif id_string == "no":
        kernel = get_no_spread_kernel()
    elif id_string == "constant":
        kernel = get_constant_kernel_from_string(name)
    else:
        assert 0
    return kernel


# Returns a function which can be used for aerial spread
def get_cauchy_kernel(alpha, gamma):
    def f(d):
        return gamma / (1 + d / alpha ** 2)
    name = "cauchy_alp_" + str(alpha) + "_gamma_" + str(gamma)
    return Kernel(f, name)


def get_cauchy_kernel_from_string(name):
    part = name.split('_')
    assert(part[0] == "cauchy")
    assert(part[1] == "alp")
    alpha = float(part[2])
    assert(part[3] == "gamma")
    gamma = float(part[4])
    return get_cauchy_kernel(alpha, gamma)


def get_power_law_kernel(b):
    def f(d):
        return d ** -(b + 1)
    return Kernel(f, "power_law_" + str(b))


def get_power_law_kernel_from_string(name):
    part = name.split('_')
    assert(part[0] == "power")
    assert(part[1] == "law")
    b = int(part[2])
    return get_power_law_kernel(b)


def get_constant_kernel_from_string(name):
    part = name.split('_')
    assert(part[0] == "constant")
    b = float(part[1])
    return get_constant_kernel(b)


def get_constant_kernel(b):
    def f(_):
        return b
    name = f"constant_{b}"
    return Kernel(f, name)


def get_no_spread_kernel():
    def f(_):
        return 0
    name = "no_spread"
    return Kernel(f, name)



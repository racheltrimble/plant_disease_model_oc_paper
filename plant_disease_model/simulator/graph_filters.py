class GraphFilter:
    def __init__(self, filter_func, params: list):
        self.filter_func = filter_func
        self.name = filter_func.__name__
        self.params = params

    def to_string(self):
        out = self.name
        for i in self.params:
            out += " " + str(i)
        return out

    @classmethod
    def from_string(cls, input_string):
        if input_string is None:
            return None

        params = input_string.split(' ')
        filter_name = params.pop(0)
        if filter_name == "initial_i_equals":
            assert(len(params) == 1)
            return get_initial_i_equals(params[0])
        elif filter_name == "initial_i_greater_than_zero":
            assert(len(params) == 0)
            return get_initial_i_greater_than_zero()
        elif filter_name == "iteration_equals":
            assert(len(params) == 1)
            return get_iteration_equals(int(params[0]))
        else:
            print("Bad string passed to graph filter initialisation")
            print(input_string)
            assert 0


def get_initial_i_equals(i):
    def initial_i_equals(x):
        return x.loc[x['t'] == 0.0]['nI'] == float(i)
    return GraphFilter(initial_i_equals, [i])


def get_initial_i_greater_than_zero():
    def initial_i_greater_than_zero(x):
        return x.loc[x['t'] == 0.0]['nI'] > 0
    return GraphFilter(initial_i_greater_than_zero, [])


# This is a bit of a misuse but makes it consistent with other filters.
def get_iteration_equals(n):
    def iteration_equals(x):
        return x.loc[x['t'] == 0.0]['test_iteration'] == n
    return GraphFilter(iteration_equals, n)

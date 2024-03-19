from control_ablations.generic_targets import OCIO
from pathlib import Path


# There is no real concept of iterations but it is more consistent to implement
# with everything as iteration 0 rather than making a different structure for these
# types of tests e.g. Allows reuse of eval code.
class PDOCIO(OCIO):
    def deterministic_df_path(self):
        return self.get_iteration_dir() / Path("state_df.csv")

    def deterministic_action_df_path(self):
        return self.get_iteration_dir() / Path("action_df.csv")

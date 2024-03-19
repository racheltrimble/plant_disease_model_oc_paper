from control_ablations.ablation_infra import BasePlotBlock
import subprocess
from pathlib import Path

class GitTagPlotBlock(BasePlotBlock):
    def __init__(self, _1, io, _2):
        self.io = io

    def plot(self):
        path = self.io.get_tag_file_path()
        path.write_text(self.get_git_revision_short_hash())

    @staticmethod
    def get_git_revision_short_hash() -> str:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=Path(__file__).resolve().parent).decode('ascii').strip()

    @staticmethod
    def get_trigger():
        return "git_tag"
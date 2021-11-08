import os
import subprocess

from tailor.common.testing import TailorTestCase
from tailor.perturbations.simple_perturbation import SimplePerturbation


class TestPerturbations(TailorTestCase):
    def test_perturbation(self):
        step = SimplePerturbation()
        result = step.run(inputs=["Hi this is a test", "Sample input text"])
        assert result[0] == "hi this is a test!"
        assert result[1] == "sample input text!"

    def test_deterministic_experiment(self):
        cmd = [
            "tango",
            "run",
            str(self.FIXTURES_ROOT / "experiment.jsonnet"),
            "-i",
            "tailor.perturbations.simple_perturbation",
            "-d",
            str(self.TEST_DIR),
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0
        assert len(os.listdir(self.TEST_DIR / "step_cache")) == 1

        # Running again shouldn't create any more directories.
        result = subprocess.run(cmd)
        assert result.returncode == 0
        assert len(os.listdir(self.TEST_DIR / "step_cache")) == 1

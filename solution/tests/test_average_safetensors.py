import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from solution.src.average_safetensors import average


class AverageSafetensorsTest(unittest.TestCase):
    def test_two_and_three_point_average(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = []
            for index, value in enumerate((1.0, 3.0, 8.0)):
                model = root / f"m{index}"
                model.mkdir()
                save_file({"w": torch.tensor([value]), "i": torch.tensor([2])}, model / "model.safetensors")
                for name in ("config.json", "merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json"):
                    (model / name).write_text("{}")
                models.append(model)
            two = average(models[:2], root / "two")
            self.assertAlmostEqual(load_file(root / "two/model.safetensors")["w"].item(), 2.0)
            three = average(models, root / "three")
            self.assertAlmostEqual(load_file(root / "three/model.safetensors")["w"].item(), 4.0)
            self.assertEqual(two["tensor_keys"], 2)
            self.assertEqual(three["tensor_keys"], 2)


if __name__ == "__main__":
    unittest.main()

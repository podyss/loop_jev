import json
import tempfile
import unittest
from pathlib import Path

from loop_jev.evaluation.latent_kv_summary import summarize


class LatentKVSummaryTest(unittest.TestCase):
    def test_scores_matched_examples_and_loop_ablation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mode in ("kv3", "ce3", "kv1", "ce1"):
                loops = 3 if mode.endswith("3") else 1
                checkpoint = root / f"jev-latent24-{mode}-final-ce-kl"
                (checkpoint / "eval").mkdir(parents=True)
                (checkpoint / "experiment.json").write_text(json.dumps({
                    "status": "complete", "loop_count": loops, "latent_slots": 24,
                    "kv_loss_weight": 0.1 if mode.startswith("kv") else 0.0,
                }))
                scores = (True, True) if mode == "kv3" else (True, False)
                loop_file = checkpoint / "eval" / f"gsm-mc-test-loop{loops}.jsonl"
                loop_file.write_text("".join(json.dumps({
                    "id": str(i), "correct_prediction": correct, "disable_looping": False,
                }) + "\n" for i, correct in enumerate(scores)))
                if loops > 1:
                    (checkpoint / "eval" / "gsm-mc-test-no-loop.jsonl").write_text(
                        "".join(json.dumps({
                            "id": str(i), "correct_prediction": i == 0,
                            "disable_looping": True,
                        }) + "\n" for i in range(2)),
                    )
            summary = summarize(root)
            self.assertEqual(summary["examples"], 2)
            self.assertEqual(summary["results"]["kv3"]["looped"]["correct"], 2)
            self.assertEqual(summary["results"]["kv3"]["no_loop"]["correct"], 1)
            self.assertEqual(summary["deltas"]["kv_gain_at_3_loops_pp"], 50.0)
            self.assertEqual(summary["deltas"]["kv3_same_checkpoint_loop_gain_pp"], 50.0)


if __name__ == "__main__":
    unittest.main()

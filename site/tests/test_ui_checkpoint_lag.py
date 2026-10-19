#!/usr/bin/env python3
"""Regression tests for serving-vs-trainer lag copy in the AGILLM UI."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "agillm_completion.html").read_text(encoding="utf-8")

HELPER_NAMES = (
    "finitePositiveStep",
    "trainerReferenceStep",
    "formatServingVsTrainerCopy",
)


def extract_helpers(html: str) -> str:
    chunks = []
    for name in HELPER_NAMES:
        start = html.find(f"function {name}(")
        if start < 0:
            raise AssertionError(f"missing helper {name}")
        depth = 0
        i = html.find("{", start)
        while i < len(html):
            if html[i] == "{":
                depth += 1
            elif html[i] == "}":
                depth -= 1
                if depth == 0:
                    chunks.append(html[start : i + 1])
                    break
            i += 1
        else:
            raise AssertionError(f"unclosed helper {name}")
    stub = (
        "function checkpointStep(value) {\n"
        "  const m = String(value || '').match(/step0*(\\d+)/i);\n"
        "  return m ? parseInt(m[1], 10) : null;\n"
        "}\n"
    )
    return stub + "\n".join(chunks)


class ServingLagPresentationTests(unittest.TestCase):
    def test_refresh_computes_serving_matches_and_honest_integrity(self):
        self.assertIn("const servingMatches = !!(health && health.ready === true && liveServingStep === Number(checkpoint.step));", HTML)
        self.assertIn('integrity: servingMatches ? "live GB10 checkpoint matched to serving backend"', HTML)
        self.assertIn('"latest GB10 save observed; serving checkpoint differs"', HTML)
        self.assertIn("fetch(\"/api/checkpoint\"", HTML)
        self.assertIn("trainer_live_step", HTML)

    def test_placeholders_are_not_ambiguous_serving_ellipsis(self):
        self.assertIn("checking serving checkpoint…", HTML)
        self.assertIn("Serving checkpoint: checking live backend…", HTML)
        self.assertNotIn('id="ckptCompact">serving…', HTML)
        self.assertNotIn('id="agi43-ckpt-lbl">checkpoint…', HTML)

    def test_fail_visible_classes_exist(self):
        self.assertIn(".checkpoint-status.behind", HTML)
        self.assertIn(".checkpoint-led.behind", HTML)
        self.assertIn("#agi43-ckpt-pill.behind", HTML)
        self.assertIn("behind by", HTML)

    def test_formatter_copy_matches_requested_lag_wording(self):
        js = extract_helpers(HTML)
        harness = """
const assert = (cond, msg) => { if (!cond) { console.error(msg); process.exit(1); } };
const behind = formatServingVsTrainerCopy(2865113, 2912770, {servingMatches:false});
assert(behind === 'serving 2,865,113 · trainer 2,912,770 · behind by 47,657', behind);
const current = formatServingVsTrainerCopy(2865113, 2865113, {servingMatches:true});
assert(current === 'serving 2,865,113 · trainer current', current);
const ahead = formatServingVsTrainerCopy(2912770, 2865113, {servingMatches:false});
assert(ahead === 'serving 2,912,770 · trainer 2,865,113 · serving ahead by 47,657', ahead);
const live = trainerReferenceStep({trainer_live_step:2913856, last_saved_step:2912770, file:'pretrain_step02912770.pt'});
assert(live === 2913856, String(live));
console.log('ok');
"""
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
            fh.write(js + "\n" + harness)
            path = fh.name
        result = subprocess.run(["node", path], check=True, capture_output=True, text=True)
        self.assertEqual(result.stdout.strip(), "ok")


if __name__ == "__main__":
    unittest.main()

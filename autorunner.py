import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / 'configs'

attacks = [
  'apgd',
  'cw',
  'deepfool',
  'fgsm',
  'pgd',
]

for attack in attacks:
  with open(CONFIG_PATH / 'autorun.yaml') as f:
    d = yaml.safe_load(f)

  d['adversarial']['name'] = attack
  with open(CONFIG_PATH / 'autorun.yaml', 'w') as f:
    d = yaml.safe_dump(d, f)

  subprocess.run('uv run src/main.py -c autorun', shell=True)

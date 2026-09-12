"""Give the local comparison runs readable names without modifying artifacts."""
import json
from pathlib import Path
names={'drone-ai-detail':'AI terrain · 24 views','drone-ai':'AI terrain · 12 views',
       'drone-calibration':'Classical · fixed calibration','drone-broad':'Diagnostic · broad matching',
       'drone-first':'Diagnostic · initial attempt'}
for name,title in names.items():
    path=Path('outputs')/name/'run_manifest.json'
    if path.exists():
        data=json.loads(path.read_text())
        data['display_name']=title
        path.write_text(json.dumps(data,indent=2))

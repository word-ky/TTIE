import json
from pathlib import Path
EXPECTED_BINDINGS=json.loads((Path(__file__).with_name("expected_bindings.json")).read_text(encoding="utf-8"))

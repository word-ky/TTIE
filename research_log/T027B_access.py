"""Task-scoped image-read audit shared by execution harnesses, not exporters."""
import sys
from pathlib import Path

import cv2


def install(paths):
    allowed = {Path(p).resolve() for p in paths}
    decoded, denied = [], []
    original = cv2.imread

    def check(path):
        p = Path(path).resolve()
        if p not in allowed:
            denied.append(str(p))
            raise PermissionError('Image outside T027-B smoke allowlist: ' + str(p))
        return p

    def read(path, *a, **kw):
        p = check(path)
        decoded.append(str(p))
        return original(str(p), *a, **kw)

    def audit(event, args):
        if event == 'open' and isinstance(args[0], str):
            p = Path(args[0])
            if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif'}:
                mode = args[1]
                if not isinstance(mode, str) or not any(c in mode for c in 'wax'):
                    check(p)

    cv2.imread = read
    sys.addaudithook(audit)
    return decoded, denied

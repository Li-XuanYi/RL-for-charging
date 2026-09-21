"""Resolve project-local evidence after copying this bundle to Linux/Windows."""
from pathlib import Path
import paths


def project_path(value):
    text=str(value).replace('\\','/')
    # Recorded Windows paths are provenance, not executable server locations.
    # Only known project-owned directories may be relocated.
    for anchor in ('revision_results/','revision_gpu/','revision_experiments/'):
        offset=text.find(anchor)
        if offset>=0:
            candidate=(paths.ROOT/text[offset:]).resolve()
            if not candidate.is_relative_to(paths.ROOT.resolve()):raise ValueError('Path escapes project')
            return candidate
    candidate=Path(text)
    if not candidate.is_absolute():candidate=paths.ROOT/candidate
    candidate=candidate.resolve()
    if not candidate.is_relative_to(paths.ROOT.resolve()):raise ValueError('Expected a path inside the project')
    return candidate

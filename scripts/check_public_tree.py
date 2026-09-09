"""Conservative public-tree checks. Reports locations, never credential values."""
from pathlib import Path
import ast
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
if sys.version_info < (3, 11):
    sys.exit('Run this check with the required Python 3.11 environment.')
SKIP={'.git','.venv','private','results','logs','data','cutouts','resources','__pycache__','.snakemake'}
errors=[]
count=0
for p in ROOT.rglob('*'):
    rel=p.relative_to(ROOT)
    if any(x in SKIP for x in rel.parts):continue
    if p.is_symlink():
        if not p.resolve().is_relative_to(ROOT):errors.append((str(rel),'external symlink'))
        continue
    if not p.is_file():continue
    count+=1
    if p.suffix in {'.lic','.pem','.key'} or p.name in {'.env','config.api.yaml'}:
        errors.append((str(rel),'private file'))
    if p.stat().st_size>95*1024*1024:errors.append((str(rel),'oversized Git file'))
    if p.suffix not in {'.py','.sh','.sbatch','.yaml','.yml','.json','.toml','.md','.txt'}:continue
    text=p.read_text(errors='replace')
    if p.suffix=='.py':
        try:ast.parse(text)
        except SyntaxError as e:errors.append((str(rel),f'Python syntax line {e.lineno}'))
    for i,line in enumerate(text.splitlines(),1):
        if re.search(r'(?i)(api[_-]?key|wlssecret|wlsaccessid|access[_-]?token|password|eia)\s*[=:]\s*[\"\x27]?[A-Za-z0-9_-]{24,}',line):
            errors.append((str(rel),f'possible credential line {i}'))
        if re.search(r'gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{25,}|^-----BEGIN .*PRIVATE KEY',line):
            errors.append((str(rel),f'possible credential line {i}'))
        if re.search('/n' + r'fs/(stak|hpc)/|/Us' + r'ers/[^/]+/|~/hpc' + '-share',line):
            errors.append((str(rel),f'personal absolute path line {i}'))
for path,reason in errors:print(path,reason)
print(f'Checked {count} files; {len(errors)} findings. This is a heuristic scan, not a guarantee.')
sys.exit(bool(errors))

"""Export verified reduced source data; raw checkpoints are external archives."""
from pathlib import Path
import json
import shutil
import subprocess
import sys

from .build import sha


def export_data(output,manifest=None):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    manifest=Path(manifest) if manifest else Path(__file__).parent/'paper_data/manifest.json'
    if not manifest.exists():raise FileNotFoundError('Supply the paper reduced-data manifest; no data are invented or silently downloaded.')
    spec=json.loads(manifest.read_text());references=[]
    for item in spec['files']:
        relative=Path(item['path'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Portable relative bundle paths required.')
        path=(manifest.parent/relative).resolve()
        if not path.is_relative_to(manifest.parent.resolve()):raise ValueError('Bundle source resolves outside its root.')
        if sha(path)!=item['sha256']:raise ValueError('Changed paper source: '+str(path))
        references.append((path,relative))
    output.mkdir(parents=True)
    for source,relative in references:
        target=output/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    shutil.copy2(manifest,output/'manifest.json')
    return output/'manifest.json'


def reproduce_figures(output,data=None):
    """Run the portable source-bound figure script; no raw trajectories or MD."""
    data=Path(data).resolve() if data else Path(__file__).parent/'paper_data'
    manifest=data/'manifest.json'
    if not manifest.exists():raise FileNotFoundError('A verified portable paper-data bundle is required.')
    for item in json.loads(manifest.read_text())['files']:
        relative=Path(item['path']);path=(data/relative).resolve()
        if relative.is_absolute() or '..' in relative.parts or not path.is_relative_to(data.resolve()):
            raise ValueError('Portable relative bundle paths required.')
        if sha(path)!=item['sha256']:raise ValueError('Changed paper bundle member: '+str(relative))
    runner=data/'reproduce_figures.py'
    if not runner.exists():raise FileNotFoundError('Portable figure runner is not bundled.')
    subprocess.run([sys.executable,str(runner),'--data',str(data),'--output',str(Path(output).resolve())],check=True)
    return Path(output)

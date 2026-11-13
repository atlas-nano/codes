"""Validated chemistry and explicitly nonequilibrium starting packings.

The supported parameterizations are the PVDF and PDMS models used by the paper.
SMILES identifies the repeat graph; it never supplies force-field parameters.
Other linear polymers require imported, explicitly parameterized LAMMPS data;
this package does not assign unsupported force fields from SMILES. Large builds
belong in compute allocations.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shutil
import subprocess

import numpy as np


SMILES = {'PVDF': '[*:1]CC(F)(F)[*:2]', 'PDMS': '[*:1]O[Si](C)(C)[*:2]'}
MASSES = {'H': 1.008, 'C': 12.011, 'F': 18.998, 'O': 15.999, 'Si': 28.086}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_repeat(smiles=None, monomer_file=None, polymer=None):
    """Identify a two-attachment repeat without interpreting atom count as identity."""
    if smiles and monomer_file:
        raise ValueError('Choose a SMILES string or one monomer file.')
    if monomer_file:
        path=Path(monomer_file)
        if path.suffix.lower() in ('.smi','.smiles','.txt'):
            smiles=path.read_text().strip().splitlines()[0].split()[0]
        else:
            try: from rdkit import Chem
            except ImportError as error: raise ImportError('Monomer MOL/SDF reading requires pip install .[structures].') from error
            mol=Chem.MolFromMolFile(str(path),removeHs=False)
            if mol is None: raise ValueError('Cannot read monomer MOL/SDF.')
            smiles=Chem.MolToSmiles(mol)
    if not smiles:
        if polymer not in SMILES: raise ValueError('Supply a repeat SMILES/monomer file or supported polymer name.')
        smiles=SMILES[polymer]
    name=None
    try:
        from rdkit import Chem
        mol=Chem.MolFromSmiles(smiles)
        if mol is None: raise ValueError('Invalid repeat SMILES.')
        dummies=[a for a in mol.GetAtoms() if a.GetAtomicNum()==0]
        if len(dummies)!=2 or any(a.GetDegree()!=1 for a in dummies):
            raise ValueError('A linear repeat needs exactly two singly bonded attachment dummy atoms [*:1]/[*:2].')
        elements=Counter(a.GetSymbol() for a in mol.GetAtoms() if a.GetAtomicNum() not in (0,1))
        attachment_elements=sorted(a.GetNeighbors()[0].GetSymbol() for a in dummies)
        all_single=all(b.GetBondTypeAsDouble()==1 for b in mol.GetBonds())
        neutral_standard=all(a.GetFormalCharge()==0 and a.GetNumRadicalElectrons()==0 and a.GetIsotope()==0 for a in mol.GetAtoms())
        carbons=[a for a in mol.GetAtoms() if a.GetSymbol()=='C']
        fluorines_per_carbon=sorted(sum(n.GetSymbol()=='F' for n in a.GetNeighbors()) for a in carbons)
        if neutral_standard and all_single and elements==Counter(C=2,F=2) and attachment_elements==['C','C'] and fluorines_per_carbon==[0,2]:
            name='PVDF'
        methyls_on_Si=all(a.GetDegree()==1 and a.GetNeighbors()[0].GetSymbol()=='Si' for a in carbons)
        if neutral_standard and all_single and elements==Counter(C=2,Si=1,O=1) and attachment_elements==['O','Si'] and methyls_on_Si:
            name='PDMS'
    except ImportError:
        simple=re.sub(r'\[\*:\d+\]','[*]',smiles.replace(' ',''))
        if simple in ('[*]CC(F)(F)[*]','[*]C(F)(F)C[*]'): name='PVDF'
        if simple in ('[*]O[Si](C)(C)[*]','[*][Si](C)(C)O[*]'): name='PDMS'
        if name is None:
            raise ImportError('General repeat-graph parsing requires pip install .[structures].')
    if polymer and polymer not in ('custom',name):
        raise ValueError('The repeat graph does not match the requested supported polymer.')
    return dict(polymer=name or 'custom',repeat_smiles=smiles,
                force_field_available=(name in SMILES),connection_sites=2)


def _unit(v):
    v=np.asarray(v,float);norm=np.linalg.norm(v)
    if not np.isfinite(norm) or norm<1e-12: raise ValueError('Degenerate starting geometry.')
    return v/norm


def _basis(axis):
    axis=_unit(axis);ref=np.array([0.,0.,1.]) if abs(axis[2])<.85 else np.array([1.,0.,0.])
    e1=_unit(np.cross(axis,ref));return e1,np.cross(axis,e1)


def _coil(n,center,rng,box_length):
    """Unconfined periodic coils; .6/.2/.2 torsions initialize, not equilibrate."""
    points=[np.zeros(3)];axis=_unit([rng.gauss(0,1) for _ in range(3)]);previous=None
    angle=math.pi-math.radians(112.7)
    for _ in range(n-1):
        if previous is None: direction=axis
        else:
            e1=_unit(-previous+np.dot(previous,axis)*axis) if abs(np.dot(previous,axis))<.999 else _basis(axis)[0]
            e2=np.cross(axis,e1);phi=rng.choices([math.pi,math.pi/3,-math.pi/3],[.6,.2,.2])[0]
            direction=math.cos(angle)*axis+math.sin(angle)*(math.cos(phi)*e1+math.sin(phi)*e2)
        points.append(points[-1]+1.529*direction);previous,axis=axis,_unit(direction)
    return [tuple(p+np.asarray(center)) for p in points]


def _substituents(backbone,index,count,rng):
    p=np.asarray(backbone[index])
    if count==3:
        neighbor=backbone[1] if index==0 else backbone[-2];axis=_unit(np.asarray(neighbor)-p)
        e1,e2=_basis(axis);phase=rng.uniform(0,2*math.pi)
        return [tuple(-axis/3+math.sqrt(8/9)*(math.cos(phase+2*j*math.pi/3)*e1+
                    math.sin(phase+2*j*math.pi/3)*e2)) for j in range(3)]
    u=_unit(np.asarray(backbone[index-1])-p);v=_unit(np.asarray(backbone[index+1])-p)
    bisector=_unit(u+v);normal=_unit(np.cross(u,v))
    return [tuple(-bisector/math.sqrt(3)+sign*math.sqrt(2/3)*normal) for sign in (-1,1)]


def data_sections(path):
    names=['Masses','Pair Coeffs','Bond Coeffs','Angle Coeffs','Dihedral Coeffs',
           'Atoms','Bonds','Angles','Dihedrals','Impropers','Velocities']
    result={n:[] for n in names};current=None;bounds={}
    for raw in Path(path).read_text().splitlines():
        line=raw.split('#',1)[0].strip();fields=line.split()
        if len(fields)==4 and fields[2:] in (['xlo','xhi'],['ylo','yhi'],['zlo','zhi']):
            bounds[fields[2][0]]=tuple(map(float,fields[:2]))
        if line in result: current=line
        elif line and current:
            if re.match(r'^\d+\s',line): result[current].append(fields)
            else: current=None
    result['bounds']=bounds;return result


def validate_data(path,polymer=None,repeats=None,chains=None):
    """Check typed graph, connected molecules, per-chain charge and cap conventions."""
    d=data_sections(path);atoms=d['Atoms'];bonds=d['Bonds']
    if not atoms or not bonds or not all(d[k] for k in ['Masses','Pair Coeffs','Bond Coeffs','Angle Coeffs','Dihedral Coeffs']):
        raise ValueError('Typed full-style data with all physical coefficient sections is required.')
    if d['Impropers']:raise ValueError('This publication writer supports the paper OPLS functional forms without impropers; supply a reviewed writer for other forms.')
    ids=[int(a[0]) for a in atoms]
    if len(set(ids))!=len(ids): raise ValueError('Duplicate atom IDs.')
    by_id={int(a[0]):a for a in atoms};adj=defaultdict(set)
    for b in bonds:
        i,j=map(int,b[2:4])
        if i not in by_id or j not in by_id or i==j: raise ValueError('Dangling/self bond.')
        if by_id[i][1]!=by_id[j][1]: raise ValueError('Intermolecular covalent bond in linear-polymer packing.')
        adj[i].add(j);adj[j].add(i)
    molecule_ids=sorted({int(a[1]) for a in atoms});counts=[];charges=[]
    for mid in molecule_ids:
        members={i for i,a in by_id.items() if int(a[1])==mid};seen=set();todo=[min(members)]
        while todo:
            i=todo.pop()
            if i in seen: continue
            seen.add(i);todo.extend(adj[i]-seen)
        if seen!=members: raise ValueError('Molecule ID contains disconnected chemistry.')
        charges.append(sum(float(by_id[i][3]) for i in members));counts.append(len(members))
        if polymer=='PVDF':
            n=int(repeats)-1;expected=Counter({1:1,2:n,3:2*n+3,4:1,5:n,6:2*n+3})
            if Counter(int(by_id[i][2]) for i in members)!=expected: raise ValueError('PVDF cap/repeat typed composition mismatch.')
            carbon={i for i in members if int(by_id[i][2]) in (1,2,4,5)}
            ends=[i for i in carbon if len(adj[i]&carbon)==1]
            if len(ends)!=2 or {int(by_id[i][2]) for i in ends}!={1,4}:
                raise ValueError('PVDF CH3/CF3 backbone cap connectivity mismatch.')
        if polymer=='PDMS':
            u=int(repeats)+1;expected=Counter({1:2*u+4,2:6*u+12,3:u,4:u+1})
            if Counter(int(by_id[i][2]) for i in members)!=expected: raise ValueError('PDMS M-D_n-M composition mismatch.')
            silicon=[i for i in members if int(by_id[i][2])==4]
            ends=[i for i in silicon if sum(int(by_id[j][2])==3 for j in adj[i])==1]
            if len(ends)!=2 or any(sum(int(by_id[j][2])==1 for j in adj[i])!=3 for i in ends):
                raise ValueError('PDMS terminal trimethylsiloxy connectivity mismatch.')
    if chains is not None and len(molecule_ids)!=int(chains): raise ValueError('Wrong number of connected chains.')
    if max(abs(q) for q in charges)>1e-6: raise ValueError('Nonneutral chain; parameterization requires explicit review.')
    if set(d['bounds'])!={'x','y','z'} or any(hi<=lo for lo,hi in d['bounds'].values()): raise ValueError('Invalid orthorhombic periodic cell.')
    return dict(atoms=len(atoms),bonds=len(bonds),angles=len(d['Angles']),dihedrals=len(d['Dihedrals']),
                chains=len(molecule_ids),atoms_per_chain=counts,charge_per_chain_e=charges,
                data_sha256=sha(path),physical_equilibrium_established=False)


def build_pvdf(output,repeats,chains,seed,density=.65):
    from .assets import pvdf as model
    n=int(repeats)-1
    if n<1 or chains<1 or density<=0: raise ValueError('PVDF needs at least two nominal VDF units and positive chains/density.')
    model.N_REPEAT_PAIRS=n;model.generate_backbone=_coil;model.substituent_dirs=_substituents
    mass=(2*n+2)*12.011+(2*n+3)*(1.008+18.998)
    length=(chains*mass*1.66053906660/density)**(1/3)
    rng=random.Random(seed);centers=[tuple(rng.uniform(0,length) for _ in range(3)) for _ in range(chains)]
    atoms,_,carbon_ids=model.build_atoms(centers,length,seed)
    bonds,angles,dihedrals=model.build_topology(atoms,carbon_ids)
    target=output/'initial.data';model.write_data(target,atoms,bonds,angles,dihedrals,length)
    lines=target.read_text().split('\n',1);target.write_text(f'DMAx PVDF: {chains} chains CH3-[CF2-CH2]_{n}-CF3\n'+lines[1])
    return dict(polymer='PVDF',nominal_repeat_units=repeats,internal_pairs=n,caps='CH3/CF3',
                initial_density_g_cm3=density,box_length_A=length,
                initializer='Unconfined periodic random coil; C-C1.529A, CCC112.7deg, initial t/g+/g- .6/.2/.2',
                parameter_asset_sha256=sha(model.__file__),geometry_seed=seed,**validate_data(target,'PVDF',repeats,chains))


def build_pdms(output,repeats,chains,seed,density=.1,packmol='packmol',uff_iterations=2000):
    try:
        from rdkit import Chem,rdBase
        from rdkit.Chem import AllChem
    except ImportError as error: raise ImportError('PDMS building requires RDKit: pip install .[structures].') from error
    from .assets import pdms as model
    executable=shutil.which(packmol) or (str(Path(packmol).resolve()) if Path(packmol).is_file() else None)
    if executable is None: raise FileNotFoundError('Packmol is required; supply --packmol /path/to/packmol.')
    u=int(repeats)+1
    if repeats<1 or chains<1 or density<=0: raise ValueError('Positive PDMS repeat/chains/density required.')
    atoms_per_chain=10*u+17
    if atoms_per_chain*chains>99999: raise ValueError('Packmol/PDB atom capacity exceeded; use a higher-capacity packing backend.')
    conformers=[];paths=[]
    for index in range(chains):
        actual_seed=(int(seed)+index*1009)%2147483646+1
        heavy=Chem.MolFromSmiles(model.paper_pdms_smiles(u))
        params=AllChem.ETKDGv3();params.randomSeed=actual_seed;params.numThreads=1;params.useRandomCoords=True
        if AllChem.EmbedMolecule(heavy,params)!=0: raise ValueError('Heavy-first PDMS embedding failed.')
        mol=Chem.AddHs(heavy,addCoords=True)
        if not AllChem.UFFHasAllMoleculeParams(mol): raise ValueError('UFF initializer lacks parameters.')
        status=AllChem.UFFOptimizeMolecule(mol,maxIters=uff_iterations)
        path=output/f'chain_{index:04d}.pdb';Chem.MolToPDBFile(mol,str(path));paths.append(path)
        conformers.append(dict(seed=actual_seed,UFF_status=int(status),UFF_converged=(status==0),pdb_sha256=sha(path)))
    formula=dict(C=2*u+4,H=6*u+12,O=u,Si=u+1)
    mass=sum(MASSES[k]*v for k,v in formula.items());length=(mass*chains/6.02214076e23/density*1e24)**(1/3)
    packed=output/'packed.pdb';inp=output/'packmol.inp'
    lines=['tolerance 2.0','nloop 20',f'seed {seed}','filetype pdb',f'output {packed.resolve()}']
    for path in paths: lines += [f'structure {path.resolve()}',' number 1',f' inside box 1 1 1 {length-1:.9f} {length-1:.9f} {length-1:.9f}','end structure']
    inp.write_text('\n'.join(lines)+'\n')
    with inp.open() as stream,(output/'packmol.log').open('x') as log:
        result=subprocess.run([executable],stdin=stream,stdout=log,stderr=subprocess.STDOUT,timeout=300)
    if result.returncode or not packed.exists() or 'Success!' not in (output/'packmol.log').read_text():
        raise ValueError('Packmol did not succeed; preserve the failed output instead of accepting forced coordinates.')
    topology=model.parse_pdb(packed)
    if len(topology.atoms)!=chains*atoms_per_chain: raise ValueError('Packmol atom count changed.')
    # Bond order is preserved and rechecked by the typed writer/graph validator.
    box=output/'box.data';box.write_text('\n'.join(f'0 {length:.12f} {a}lo {a}hi' for a in 'xyz')+'\n')
    stats=model.write_lammps_data(packed,output/'initial.data',box_data=box,strict=True,neutralize=True)
    if any(stats['unsupported'].values()): raise ValueError('Unsupported PDMS bonded chemistry.')
    positions=np.array([[a.x,a.y,a.z] for a in topology.atoms.values()]);molecules=np.arange(len(positions))//atoms_per_chain
    minimum=np.inf
    for first in range(0,len(positions),64):
        delta=positions[first:first+64,None,:]-positions[None,:,:];delta-=np.rint(delta/length)*length
        dist=np.sum(delta*delta,axis=2);dist[molecules[first:first+64,None]==molecules[None,:]]=np.inf
        minimum=min(minimum,float(np.sqrt(dist.min())))
    geometry_bound=math.sqrt(4-.01)-math.sqrt(3)*.001
    if chains>1 and minimum<geometry_bound: raise ValueError('Periodic interchain packing check failed; increase packing optimization explicitly.')
    return dict(polymer='PDMS',internal_D_units=repeats,siloxane_O_links=u,caps='trimethylsiloxy M-D_n-M',
                initial_density_g_cm3=density,box_length_A=length,RDKit_version=rdBase.rdkitVersion,
                conformers=conformers,UFF_max_iterations=uff_iterations,parameter_asset_sha256=sha(model.__file__),
                packmol_executable_sha256=sha(executable),packmol_seed=seed,periodic_interchain_min_A=minimum if chains>1 else None,
                packing_bound_A=geometry_bound,packing_bound_provenance='Paper Packmol21.2.1 precision.01A2 and .001A PDB rounding; geometry only',
                charge_correction_e=stats['neutralization_correction_per_atom'],**validate_data(output/'initial.data','PDMS',repeats,chains))


def build_packing(config,output,packing=1,packmol='packmol'):
    output=Path(output)
    if output.exists(): raise FileExistsError('A new build directory is required; previous attempts are evidence.')
    output.mkdir(parents=True)
    p=config['polymer'];seed=int(config.get('seed',20261113))+int(packing)*10007
    material=p['name'];repeats=int(p['repeat_units']);chains=int(p['chains'])
    if p.get('parameterized_data'):
        if config['packings']!=1:
            raise ValueError('One supplied coordinate file is one packing; provide separate studies for genuinely independent parameterized coordinates.')
        shutil.copy2(p['parameterized_data'],output/'initial.data')
        result=validate_data(output/'initial.data',material if material in SMILES else None,repeats if material in SMILES else None,chains)
        result.update(parameter_provider='explicit_user_parameterized_data',parameterized_source_sha256=sha(p['parameterized_data']))
    elif material=='PVDF': result=build_pvdf(output,repeats,chains,seed,p.get('initial_density_g_cm3',.65))
    elif material=='PDMS': result=build_pdms(output,repeats,chains,seed,p.get('initial_density_g_cm3',.1),packmol)
    else:
        if not p.get('parameterized_data'): raise ValueError('Other chemistries require explicit parameterized_data; SMILES alone does not define OPLS parameters.')
        shutil.copy2(p['parameterized_data'],output/'initial.data');result=validate_data(output/'initial.data',chains=chains)
    result.update(packing=int(packing),independent_packing_seed=seed,
                  state='starting_packing_requires_soft_push_off_and_physical_conditioning',
                  entanglement_equilibrium_established=False,configuration=config)
    (output/'build_manifest.json').write_text(json.dumps(result,indent=2)+'\n');return result

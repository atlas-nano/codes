import re
import os
import sys
import subprocess
import numpy as np
import random
from math import log10

def randomNumber(digits):
    """
    Function utilized to generate a random number for modifying the velocity seed for each calculation.

    Arguments:
    digits -- Number of digits in the random number to be generated.
    """
    lower = 10**(digits-1)
    upper = 10**digits - 1
    return random.randint(lower, upper)

def frequency_dirname_parser(frequency):
    """
    Function utilized to parse the directory names of .

    Arguments:
    digits -- Number of digits in the random number to be generated.
    """
    freq_range_dict = {"Hz": 1e0, "KHz": 1e3, "MHz": 1e6, "GHz": 1e9, "THz": 1e12, "PHz": 1e15, "EHz": 1e18}
    for freq_mag, freq in freq_range_dict.items():
        dirname_val = frequency / freq
        if 1 <= dirname_val < 1000:
            return "{}{}".format(dirname_val, freq_mag)

def submit_slurm_files(dir_path, avoid_containing_log=True):
    """
    Recursively submit all SLURM files found in a directory and its subdirectories,
    but do so inside the directories where the SLURM files are found.

    Arguments:
    dir_path -- A string representing the path of the directory to search for SLURM files.
    avoid_containing_log -- A boolean flag to avoid submitting files in directories containing a .log file.
    """

    # Iterate over each item in the given directory
    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)

        # If the item is a directory, recursively call this function on it
        if os.path.isdir(item_path):
            submit_slurm_files(item_path, avoid_containing_log)

        # If the item is a file with a ".slurm" extension, submit it to SLURM
        elif os.path.isfile(item_path) and item.endswith(".slurm"):
            # Check if avoid_containing_log flag is True and if a ".log" file exists in the same directory
            if avoid_containing_log and any(f.endswith(".log") for f in os.listdir(dir_path)):
                print(f"Skipping {item_path} as it is in a directory containing a .log file.")
            else:
                # Use the subprocess module to run the sbatch command on the SLURM file
                # But do so inside the directory where the SLURM file is found
                slurm_dir = os.path.dirname(item_path)
                subprocess.run(["sbatch", os.path.basename(item_path)], cwd=slurm_dir)

class LammpsInputGenerator:
    """
    Base LAMMPS input generator. Utilizes Prof. Tod A. Pascal's createLammpsInput C++ code to generate LAMMPS input files.
    Attributes
    ----------
    structure_file : BGF|MSI|MOL2
        The input structure for the calculation. Can also take the formats below with listed limitations.
        PDB|PQR|XYZ - connections will be generated automatically (if no CONECT tables).
            NOTE: Forcefield types are assumed to be the same as the atom names
        CHARMM_PSF - Also need to specify CHARMM CRD file (in quotes) or have xxx.crd file in same folder
        AMBER_PRMTOP - Also need to specify AMBER CRD|RST file (in quotes) or have xxx.crd|xxx.rst7 file in same folder
        GROMACS_GRO - Also need to specify TOP|XPLOR file (in quotes) xxx.top|xxx.xplor file in the same folder
    calc_dir: str
        Path to where input files should be written.
    forcefield : paths to 1 or more Cerius2|Polygraf|ITP|CHARMM_PRM|GROMACS_TOP formatted forcefields (str)
        Valid entries are
        AMBER91/96/99/03/19 - the AMBER 1991/1996/1999/2003/2019 forcefield for proteins and DNA
        GAFF - the General AMBER forcefield for small molecules
        CHARMM - the CHARMM par_all27_prot_na Protein/Nucleic Acid forcefield
        CHARMM_LIPID - the CHARMM par_all27_prot_lipid Protein/Lipid force field
        DREIDING - The 1990 DREIDING forcefield with F3C waters
        MESODNA - The DNA Meso scale forcefield v.6.0
        REAX - The REAX force field. You may have to specify a location with -r
        REXPON_WAT - The RexPoN-water force field
        --or-- you can specify your own forcefield location
        NOTE: you can specify multiple forcefields by enclosing them in ""
        NOTE: If specifying multiple forcefields with the same atom types, the atom type values will come from the last matching forcefield.
    suffix : str
        When specified, the program will generate in.[suffix] and data.[suffix] as the files. If not specified, the output will be in.lammps, data.lammps
    reax_rexpon_file : str
        Specifies either the ReaxFF  or RexPoN forcefield file
    qeq : str
        Specifies either the QEq/PQEq/Drude parameters set
    options : str
        Controls various options in input file. Valid entries include:
            "amoeba_monopole" - switch from full multipole to monopole only amoeba
            "2D or dimension 2" - for 2D simulations
            "compress_parms" - compress parameters to eliminate multiples with the same values.
            "finite" - for 0D (isolated) simulations
            "ewald vdw" - calculate long range vdw using ewald summations. Only valid for lj and exp6 potentials.
            "no shake|shake solute/solvent" - shake constraints on the hydrogens are turned on by default. This turns it off or applies it to the solute or solvent only
            "no_labels" - Specfies whether to supress writing atom type based label for the coefficients. Default yes
            "nobonds:" 'atom selection' - Delete all bonds involving selected atoms. If no atom selection give, then will delete ALL bonds!!
            "sort_fftypes" - Sort the fftypes alphabetically and renumber accordingly. Default 0: fftypes sorted based on encountered order in structure file
            "write_inputfile_coeffs" - Specifies whether to write the coefficients in the input file, instead of the data file. Default 0
            "write_inputfile_type_charges" - Specifies whether to write the charge of the atom types from the forcefield (if provided) in the forcefield. Default 0
            "write_inputfile_pair_off_diag" - Specifies whether to write the off diagonal components to the input file. By default, this is done
            "fep:" 'atom selection'. Write simulation parameters to perform a FEP simulation on atom selection(s)
            "qeq" - dynamically determine the charge during dynamics using the QEQ charge equilibration scheme. See the -x option to specify the parameter set, else the default set will be used
            "pqeq" - dynamically determine the partial atomic charge during dynamics using the PQEq scheme. See the -x option to specify the parameter set, else the default set will be used
                NOTE: The charges are repesented as gaussian distributions (not point charges) and associated with the coul/pqeq pair style.
            "charge x" - overall charge on system with pqeq. Default 0
            "electrode:" 'atom selection_1' ('atom_selection_2')" - for QEq simulations, this invokes the ECHEMDID method, based on the Chi parameter of the specified forcefield. The format is top electrode, bottom electrode. If only is specified then the fftype is for both. Same for conp simulations (see above)
                    "conp" - Activate the CONP options for electrochemical cell simulations. Must be use in conjunction with the 'electrode' flag
                    "fixedQ" - Activate the fixed charge electrode option. Must be used in conjunction with the 'electrode' flag
            "polarizable:" 'atom selection(s)' [adiabatic/drude/thole/pqeq]" - Turns on shell polarization options. The atom selection(s) should be enclosed in quotes and is based on the usual selection criteria.The shell polarization options are:
                            adiabatic: adiabatic core/shell model,
                            drude: drude induced dipole,
                            thole: drude induced dipole with thole short range screening,
                            pqeq: represent the electrostatics between the atoms using overlap of gaussian types orbitals, as opposed to point dipoles in the other options.
            "rigid: 'atom selection(s)'" - Treat the specified atoms (and their associated molecules) as rigid bodies during dynamics
            "include_file:" - Include a file with LAMMPS code after the data_read line. Use for further customization
    template : str
        Specifies the type of input file to create. See /home/tpascal/scripts/dat/LAMMPS (#change here) for a list. Current options include "2pt 2pt_f3c 2pt_tip4 anneal atom.eng battery biogroup bulk carbonates ced cnt cntWat cnt_h2o cof compress electrode entropy equil equil_new fep full gpcr heat ift md melt mesodna min mof nano110 nano_indent_x nano_indent_y nve polymer pre_fep pressureflow prod pulse reaxff rimse rimse_solvent solv solv.2pt solv.2pt.rigidSolu solv.atom.eng solv.atom.eng.rigidSolu solv.rigidSolu solvation solvation_f3c test thermal_conductivity thz-pulse viscosity water " or you can specify your own input file
    account: str
        The sueprcomputer allocation to be used. If none is provided, will raise an error since an account must be provided for simulations to run.
    nodes: int
        Number of nodes to be used in the supercomputer.
    ntasks_per_node: int
        Number of cpus per node to be used in the supercomputer.
    partition: str
        The queue in which to run the job.
    time: str
        The maximum amount of time that the simulation can run for.
    temperature: float
        The temperature (K) in which the simulation should be ran at.
    pressure: float
        The pressure (atm) in which the simulation should be ran at.
    """
    def __init__(self, structure_file, calc_dir, forcefield="UFF", suffix=None, reax_rexpon_file="", qeq="", options="", template="", account=None, nodes=1, ntasks_per_node=16, partition='shared', time='24:0:0', temperature=300, pressure=1):
        self.structure_file = structure_file
        self.calc_dir = calc_dir
        self.forcefield = forcefield
        if suffix == None:
            self.suffix = os.path.splitext(os.path.basename(self.structure_file))[0]
        else:
            self.suffix = suffix
        self.reax_rexpon_file = reax_rexpon_file
        self.qeq = qeq
        self.options = options
        self.template = template
        if account == None:
            raise Exception("An account must be provided for simulations to run. If you wish to only see the inputs generated, write any string for account and try again.")
        self.account = account
        self.nodes = nodes
        self.ntasks_per_node = ntasks_per_node
        self.partition = partition
        self.time = time
        self.temperature = temperature
        self.pressure = pressure

    def createLammpsInput(self, **kwargs):
        """
        Method for generating the LAMMPS input files.
        Parameters
        ----------
        omit_prints : bool
            Specifies whether the user wants to see the print statements from the C++ code that is executed to create the LAMMPS input files. Useful for debugging file generation issues.
        """
        structure_file = kwargs.get('structure_file', self.structure_file)
        calc_dir = kwargs.get('calc_dir', self.calc_dir)
        forcefield = kwargs.get('forcefield', self.forcefield)
        suffix = kwargs.get('suffix', self.suffix)
        reax_rexpon_file = kwargs.get('reax_rexpon_file', self.reax_rexpon_file)
        qeq = kwargs.get('qeq', self.qeq)
        options = kwargs.get('options', self.options)
        template = kwargs.get('template', self.template)
        temperature = kwargs.get('temperature', self.temperature)
        omit_prints = kwargs.get('omit_prints', True)
        random_seed = kwargs.get("random_seed", False)
        os.chdir(calc_dir)
        if not omit_prints:
            subprocess.call('/expanse/lustre/projects/csd626/tpascal/scripts/createLammpsInput.pl -b {} -f {} -s {} -r {} -q {} -o {} -t {}'.format(self.structure_file, self.forcefield, self.suffix, self.reax_rexpon_file, self.qeq, self.options, self.template), shell=True)
        else:
            with open('/dev/null', 'w') as devnull:
                subprocess.call('/expanse/lustre/projects/csd626/tpascal/scripts/createLammpsInput.pl -b {} -f {} -s {} -r {} -q {} -o {} -t {}'.format(self.structure_file, self.forcefield, self.suffix, self.reax_rexpon_file, self.qeq, self.options, self.template), shell=True, stdout=devnull, stderr=devnull)
        subprocess.call('cp /expanse/lustre/projects/csd626/ricardosb/master_files/slurm_script.master {}.lammps.slurm'.format(self.suffix), shell=True)
        if random_seed:
            subprocess.call("sed -i 's/0.0 12345678/{} {}/g' in.*".format(temperature, randomNumber(8)), shell=True)

    def slurm_modifier(self, **kwargs):
        account = kwargs.get('account', self.account)
        nodes = kwargs.get('nodes', self.nodes)
        ntasks_per_node = kwargs.get('ntasks_per_node', self.ntasks_per_node)
        partition = kwargs.get('partition', self.partition)
        time = kwargs.get('time', self.time)
        temperature = kwargs.get('temperature', self.temperature)
        pressure = kwargs.get('pressure', self.pressure)
        sed_string = "sed -i -e 's/master_prefix/{}/g ; s/master_account/{}/g ; s/master_nodes/{}/g ; s/master_ntasks_per_node/{}/g ; s/master_time/{}/g ; s/master_temp/{}/g ; s/master_partition/{}/g ; s/master_press/{}/g' *.slurm".format(self.suffix, account, nodes, ntasks_per_node, time, temperature, partition, pressure)
        subprocess.call(sed_string, shell=True)
        

class DMAInputGenerator(LammpsInputGenerator):
    def createLammpsInputDMA(self, frequency, temperature=300, pressure=1, timestep_unit_fs=1, oscillation_amplitude_percentage=0.04, numcycles=5, datapoints_per_cycle=500, tilt_analysis=False, nve=False, stressdir=None, master_inputs=None, **kwargs):
        temperature = kwargs.get('structure_file', self.temperature)
        pressure = kwargs.get("pressure", self.pressure)
        calc_dir = kwargs.get("calc_dir", self.calc_dir)
        stressdir = kwargs.get("stressdir", None)
        random_seed = kwargs.get("random_seed", False)
        os.chdir(calc_dir)
        timestep = timestep_unit_fs
        period = (1 / frequency) / (timestep * 1e-15)
        if not master_inputs:
            self.createLammpsInput(**{"calc_dir": calc_dir, "temperature": temperature, "random_seed": random_seed})
            self.slurm_modifier(**{"temperature": temperature, "pressure": pressure})
            subprocess.call("rm *singlepoint", shell=True)
            in_line = int(subprocess.check_output("grep -n 'NVT production dynamics' in.* | cut -d : -f 1", shell=True))
            subprocess.call("head -n {} in.* > in_head".format(in_line), shell=True)
            subprocess.call("cat in_head /expanse/lustre/projects/csd626/ricardosb/master_files/in.master_mech_loss > in.{}".format(self.suffix), shell=True)
            subprocess.call("rm *in_head*", shell=True)
        else:
            subprocess.call('cp {} in.{}'.format(master_inputs["in"], self.suffix), shell=True)
            subprocess.call('cp {} data.{}'.format(master_inputs["data"], self.suffix), shell=True)
            # For the next command line to work, the lines (variable             input string in.*) and (variable             sname string *) must have the * replaced by the word 'master'
            subprocess.call("sed -i 's/master/{}/g' in.*".format(self.suffix), shell=True)
            # Finding magnitude and direction of longest lattice vector and determining the oscillation amplitude amd direction
        grepCRYSTX = subprocess.check_output("grep 'CRYSTX' {}".format(self.structure_file), shell=True).decode("utf-8").split()
        latparam_dict = {"x": float(grepCRYSTX[1]), "y": float(grepCRYSTX[2]), "z": float(grepCRYSTX[3])}
        biggestlatparam = max(latparam_dict.values())
        biggestlatdir = max(latparam_dict, key=latparam_dict.get)
        dirlist = list(latparam_dict.keys())
        if stressdir != None:
            changing_var1 = stressdir
            changing_var2 = stressdir
        else:
            changing_var1 = biggestlatdir
            changing_var2 = dirlist[0] + dirlist[1]
        if tilt_analysis == False:
            dirlist.remove(changing_var1)
            subprocess.call("sed -i -e 's/stressdir/{}/g ; s/dir1/{}/g ; s/dir2/{}/g' in.*".format(changing_var1, dirlist[0], dirlist[1]), shell=True)
        else:
            subprocess.call("sed -i -e 's/stressdir wiggle osc_amp freq_input dir1 volume dir2 volume/{} wiggle osc_amp freq_input/g' in.*".format(changing_var2), shell=True)
        if nve:
            subprocess.call("sed -i -e 's/fix                  2 all nvt temp ${rtemp} ${rtemp} 100.0 tloop 10 ploop 10/fix                  2 all nve/g' in.*", shell=True)
        subprocess.call("sed -i 's/osc_amp/{}/g' in.*".format(biggestlatparam * float(oscillation_amplitude_percentage)), shell=True)
        subprocess.call("sed -i 's/freq_input/{}/g' in.*".format(round(period)), shell=True)
        dump_frequency = period / datapoints_per_cycle
        tdamp = 100
        # Adjusting the calculation timestep to adapt for high frequencies
        self.in_modifier_DMA(period, dump_frequency, timestep, tdamp, numcycles)
        
    def in_modifier_DMA(self, period, dump_frequency, timestep, tdamp, numcycles):
        runtime_multiplier = 1
        while dump_frequency < 1:
            timestep /= 10
            dump_frequency *= 10
            runtime_multiplier *= 10
            tdamp /= 2
        runtime = round(period * runtime_multiplier * int(numcycles))
        dump_frequency = round(dump_frequency)
        subprocess.call("sed -i 's/prod_timestep/{}/g' in.*".format(timestep), shell=True)
        # Determining the runtime to fit the desired number of cycles
        subprocess.call("sed -i 's/runtime/{}/g' in.*".format(runtime), shell=True)
        # Determining the dump frequency to fit N points per cycle
        subprocess.call("sed -i 's/dump_freq/{}/g' in.*".format(dump_frequency), shell=True)
        # Increasing the maximum number of neighbors since it seems to be a problem for some supercells
        subprocess.call("sed -i 's/check yes/check yes one 3000/g' in.*", shell=True)
        # Reducing the temperature update time to avoid issues at high frequencies
        subprocess.call("sed -i 's/tdamp/{}/g' in.*".format(tdamp), shell=True)        

    def noise_filter_CLI(self, min_frequency, numcycles, datapoints_per_cycle=500, timestep=1, tdamp=100, **kwargs):
        calc_dir = kwargs.get('calc_dir', self.calc_dir)
        temperature = kwargs.get('temperature', self.temperature)
        pressure = kwargs.get('pressure', self.pressure)
        os.chdir(calc_dir)
        period = (1 / min_frequency) / 1e-15
        dump_frequency = period / datapoints_per_cycle
        if not "noise_filter_run" in os.listdir(calc_dir):
            os.mkdir("noise_filter_run")
        os.chdir("noise_filter_run")
        self.createLammpsInput(**{"calc_dir": calc_dir, "temperature": temperature})
        # Calculate the maximum runtime from all frequencies observed
        runtime = round(period * numcycles)
        subprocess.call("rm *singlepoint", shell=True)
        in_line = int(subprocess.check_output("grep -n 'NVT production dynamics' in.* | cut -d : -f 1", shell=True))
        subprocess.call("head -n {} in.* > in_head".format(in_line), shell=True)
        subprocess.call("cat in_head /expanse/lustre/projects/csd626/ricardosb/master_files/in.master_noise_filter_run > in.{}".format(self.suffix), shell=True)
        subprocess.call("rm *in_head*", shell=True)
        self.in_modifier_DMA(period, dump_frequency, timestep, tdamp, numcycles)
        self.slurm_modifier(**{"temperature": temperature, "pressure": pressure})
        os.chdir("../")

    def error_analysis_DMA(self, frequencies=[25e9], num_error_calcs=5, standard_numcycles=5, noise_filter_run=False, **kwargs):
        oscillation_amplitude_percentage = kwargs.get('oscillation_amplitude_percentage', 0.04)
        numcycles = kwargs.get('numcycles', 10)
        temperature = kwargs.get('structure_file', self.temperature)
        pressure = kwargs.get("pressure", self.pressure)
        calc_dir = kwargs.get("calc_dir", self.calc_dir)
        stressdir = kwargs.get("stressdir", None)
        os.chdir(calc_dir)
        if not "frequencies" in os.listdir("."):
            os.mkdir("frequencies")
        os.chdir("frequencies")
        if noise_filter_run:
            self.noise_filter_CLI(min(frequencies), **{ "calc_dir": os.getcwd(), "temperature": temperature, "pressure": pressure})
        for freq in frequencies:
            for i in range(num_error_calcs):
                prefix = "{}_mech_loss_{}K_{}_{}".format(self.suffix, temperature, frequency_dirname_parser(freq), i)
                print("Creating inputs for " + prefix)
                if not "{}_{}".format(frequency_dirname_parser(freq), i) in os.listdir("."):
                    os.mkdir("{}_{}".format(frequency_dirname_parser(freq), i))
                os.chdir("{}_{}".format(frequency_dirname_parser(freq), i))
                self.createLammpsInputDMA(freq, oscillation_amplitude_percentage=oscillation_amplitude_percentage, numcycles=numcycles, temperature=temperature, pressure=pressure, **{"calc_dir": os.getcwd(), "stressdir": stressdir, "random_seed": True})
                os.chdir("../")


class DMAConvAnalysisInputGenerator(DMAInputGenerator):
    """
    LAMMPS input generator for the convergence analysis of parameters for Dynamic Mechanical Analysis.
    """
    def numCyclesConvAnalysisCLI(self, std_freq_Hz=25e9, custom_numcycles=None, standard_numcycles=5, bulk_modulus_GPa=None, strain_direction=None, noise_filter_run=True, **kwargs):
        """
        Method for generating the LAMMPS input files for a runtime convergence for Dynamic Mechanical Analysis.
        Parameters
        ----------
        std_freq : float
            Standard frequency (in Hz) to run all simulations. Default to 25GHz.
        custom_numcycles: int
            Custom number of cycles to test for convergence.
        bulk_modulus_GPa: float
            Estimated Bulk Modulus of the observed structure. If provided, will be used to optimize the selected strain range to better fit the material's crystallinity.
        strain_direction: str
            Strain direction of the simulation. If not provided, will default to the longest axis of the unit cell.
        noise_calc: bool
            Whether to add a calculation without any strain to remove noise in post-processing.
        """
        calc_dir = kwargs.get('calc_dir', self.calc_dir)
        temperature = kwargs.get('temperature', self.temperature)
        pressure = kwargs.get('pressure', self.pressure)
        os.chdir(calc_dir)
        if custom_numcycles:
            numcycles = custom_numcycles
        else:
            numcycles = standard_numcycles
        if bulk_modulus_GPa:
            if bulk_modulus_GPa < 100:
                numcycles = 10
            else:
                numcycles = 5
        if not "frequencies" in os.listdir("."):
            os.mkdir("frequencies")
        os.chdir("frequencies")
        if noise_filter_run:
            self.noise_filter_CLI(std_freq_Hz, numcycles, **{ "calc_dir": ".", "temperature": temperature, "pressure": pressure})
        freq_dirname = frequency_dirname_parser(std_freq_Hz)
        os.mkdir(freq_dirname)
        os.chdir(freq_dirname)
        os.mkdir("{}cycles".format(numcycles))
        os.chdir("{}cycles".format(numcycles))
        self.createLammpsInputDMA(std_freq_Hz, numcycles=numcycles, temperature=temperature, **{"calc_dir": ".", "stressdir": strain_direction})

    def strainSizeConvAnalysisCLI(self, std_freq_Hz=25e9, custom_strain_range=None, bulk_modulus_GPa=None, strain_direction=None, standard_numcycles=5, standard_numstrains=10, noise_filter_run=True, crystalline=False, **kwargs):
        """
        Method for generating the LAMMPS input files for a strain size convergence for Dynamic Mechanical Analysis. The strain percentages are proportional to the size of the unit cell in the direction of the strain
        Parameters
        ----------
        std_freq : float
            Standard frequency (in Hz) to run all simulations. Default to 25GHz.
        custom_strain_range: list
            Custom list of strains to test for convergence. Values must be in percentage (Ex.: 1%, 2% -> strain_range=[1, 2]). If not provided, will default to standard selection.
        bulk_modulus_GPa: float
            Estimated Bulk Modulus of the observed structure. If provided, will be used to optimize the selected strain range to better fit the material's crystallinity.
        strain_direction: str
            Strain direction of the simulation. If not provided, will default to the longest axis of the unit cell.
        """
        calc_dir = kwargs.get('calc_dir', self.calc_dir)
        temperature = kwargs.get('temperature', self.temperature)
        pressure = kwargs.get('pressure', self.pressure)
        os.chdir(self.calc_dir)
        if not "frequencies" in os.listdir("."):
            os.mkdir("frequencies")
        os.chdir("frequencies")
        if noise_filter_run:
            self.noise_filter_CLI(std_freq_Hz, standard_numcycles, **{ "calc_dir": ".", "temperature": temperature, "pressure": pressure})
        freq_dirname = frequency_dirname_parser(std_freq_Hz)
        os.mkdir(freq_dirname)
        os.chdir(freq_dirname)
        if crystalline:
            strain_range_pc = np.array([round(i, 3) for i in np.logspace(log10(0.001), log10(10), num=standard_numstrains, endpoint=True, base=10)])
        else:
            strain_range_pc = np.array([round(i, 3) for i in np.logspace(log10(0.1), log10(100), num=standard_numstrains, endpoint=True, base=10)])
        if custom_strain_range:
            strain_range_pc = custom_strain_range
        if bulk_modulus_GPa:
            if bulk_modulus_GPa < 100:
                strain_range_pc = np.array([10, 15, 20, 25, 30, 35])
            else:
                strain_range_pc = np.array([0.01, 0.1, 1, 2, 5, 10])
        for strain in strain_range_pc:
            os.mkdir("{}pc_strain".format(strain))
            os.chdir("{}pc_strain".format(strain))
            self.createLammpsInputDMA(std_freq_Hz, numcycles=standard_numcycles, oscillation_amplitude_percentage=strain/100, temperature=temperature, **{"calc_dir": ".", "stressdir": strain_direction})
            os.chdir("../")

class DMAMasterCurveInputGenerator(DMAInputGenerator):
    def MasterCurveCLI(self, polymer_glass_temperature, standard_numcycles, standard_strain_pc, num_frequencies=20, num_temperatures=10, temperature_interval=10, custom_frequency_range=None, custom_temperature_range=None, noise_filter_run=True, **kwargs):
        """
        Method for generating the LAMMPS input files for generating a mechanical loss master curve.
        ----------
        polymer_glass_temperature : float
            Glass temperature transition of the studied polymer.
        standard_numcycles: int
            Standard number of periods to run all simulations.
        standard_strain_pc: float
            Standard strain to run all simulations. Values must be in percentage (Ex.: 1%, 2% -> strain_range=[1, 2]).
        strain_direction: str
            Strain direction of the simulation. If not provided, will default to the longest axis of the unit cell.
        """
        temperature = kwargs.get('structure_file', self.temperature)
        pressure = kwargs.get("pressure", self.pressure)
        calc_dir = kwargs.get("calc_dir", self.calc_dir)
        stressdir = kwargs.get("stressdir", None)
        os.chdir(calc_dir)
        if not "temperatures" in os.listdir("."):
            os.mkdir("temperatures")
        os.chdir("temperatures")
        if custom_frequency_range:
            frequency_range = custom_freq_range
        else:
            frequency_range = [round(i, 3) for i in np.logspace(log10(10e9), log10(500e9), num=num_frequencies, endpoint=True, base=10)]
        if custom_temperature_range:
            temperature_range = custom_temperature_range
        else:
            temperature_range = np.arange(num_temperatures) * temperature_interval
            temperature_range -= (num_temperatures // 2) * temperature_interval
            temperature_range += polymer_glass_temperature
        for temp in temperature_range:
            if not "{}K".format(temp) in os.listdir("."):
                os.mkdir("{}K".format(temp))
            os.chdir("{}K".format(temp))
            if noise_filter_run:
                self.noise_filter_CLI(min(frequency_range), standard_numcycles, **{ "calc_dir": ".", "temperature": temp, "pressure": pressure})
            for freq in frequency_range:
                prefix = "{}_mech_loss_{}K_{}".format(self.suffix, temp, frequency_dirname_parser(freq))
                print("Creating inputs for " + prefix)
                if not "{}".format(frequency_dirname_parser(freq)) in os.listdir("."):
                    os.mkdir("{}".format(frequency_dirname_parser(freq)))
                os.chdir("{}".format(frequency_dirname_parser(freq)))
                self.createLammpsInputDMA(freq, oscillation_amplitude_percentage=standard_strain_pc/100, numcycles=standard_numcycles, temperature=temp, **{"calc_dir": ".", "stressdir": stressdir})
                os.chdir("../")
            os.chdir("../")


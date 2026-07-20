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
        os.chdir(calc_dir)
        if not omit_prints:
            subprocess.call('/expanse/lustre/projects/csd626/tpascal/scripts/createLammpsInput.pl -b {} -f {} -s {} -r {} -q {} -o {} -t {}'.format(self.structure_file, self.forcefield, self.suffix, self.reax_rexpon_file, self.qeq, self.options, self.template), shell=True)
        else:
            with open('/dev/null', 'w') as devnull:
                subprocess.call('/expanse/lustre/projects/csd626/tpascal/scripts/createLammpsInput.pl -b {} -f {} -s {} -r {} -q {} -o {} -t {}'.format(self.structure_file, self.forcefield, self.suffix, self.reax_rexpon_file, self.qeq, self.options, self.template), shell=True, stdout=devnull, stderr=devnull)
        subprocess.call('cp $MASTER_FILES/slurm_script.master {}.lammps.slurm'.format(self.suffix), shell=True)
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

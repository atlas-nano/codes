class DMAConvAnalysisInputGenerator(DMAInputGenerator):
    """
    LAMMPS input generator for the convergence analysis of parameters for Dynamic Mechanical Analysis.
    """
    def numCyclesConvAnalysisCLI(self, std_freq_Hz=25e9, custom_numcycles=None, bulk_modulus_GPa=None, strain_direction=None, noise_filter_run=True, **kwargs):
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
        if not "frequencies" in os.listdir("."):
            os.mkdir("frequencies")
        os.chdir("frequencies")
        if noise_filter_run:
            self.noise_filter_CLI(**{ "calc_dir": ".", "temperature": temperature, "pressure": pressure})
        freq_dirname = frequency_dirname_parser(std_freq_Hz)
        os.mkdir(freq_dirname)
        os.chdir(freq_dirname)
        numcycles = 5
        if custom_numcycles:
            strain_range_pc = custom_numcycles
        if bulk_modulus_GPa:
            if bulk_modulus_GPa < 100:
                numcycles = 10
            else:
                numcycles = 5
        os.mkdir("{}cycles".format(numcycles))
        os.chdir("{}cycles".format(numcycles))
        self.createLammpsInputDMA(std_freq_Hz, numcycles=numcycles, temperature=temperature, **{"calc_dir": ".", "stressdir": strain_direction})

    def strainSizeConvAnalysisCLI(self, std_freq_Hz=25e9, custom_strain_range=None, bulk_modulus_GPa=None, strain_direction=None, standard_numcycles=5, noise_filter_run=True, **kwargs):
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
            self.noise_filter_CLI(**{ "calc_dir": ".", "temperature": temperature, "pressure": pressure})
        freq_dirname = frequency_dirname_parser(std_freq_Hz)
        os.mkdir(freq_dirname)
        os.chdir(freq_dirname)
        strain_range_pc = np.array([1, 2, 4, 8, 16, 32])
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
    def MasterCurveCLI(self, polymer_glass_temperature, standard_numcycles, standard_strain, num_frequencies=20, num_temperatures=10, temperature_interval=10, custom_frequency_range=None, noise_filter_run=True, **kwargs):
        """
        Method for generating the LAMMPS input files for generating a mechanical loss master curve.
        ----------
        polymer_glass_temperature : float
            Glass temperature transition of the studied polymer.
        standard_numcycles: int
            Standard number of periods to run all simulations.
        standard_strain: float
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
        temperature_range = np.arange(num_temperatures) * temperature_interval
        temperature_range -= (num_temperatures // 2) * temperature_interval
        temperature_range += polymer_glass_temperature
        for temp in temperature_range:
            if not "{}K".format(temp) in os.listdir("."):
                os.mkdir("{}K".format(temp))
            os.chdir("{}K".format(temp))
            if noise_filter_run:
                self.noise_filter_CLI(**{ "calc_dir": ".", "temperature": temp, "pressure": pressure})
            for freq in frequency_range:
                prefix = "{}_mech_loss_{}K_{}".format(self.suffix, temp, frequency_dirname_parser(freq))
                print("Creating inputs for " + prefix)
                if not "{}".format(frequency_dirname_parser(freq)) in os.listdir("."):
                    os.mkdir("{}".format(frequency_dirname_parser(freq)))
                os.chdir("{}".format(frequency_dirname_parser(freq)))
                self.createLammpsInputDMA(freq, oscillation_amplitude_percentage=standard_strain, numcycles=standard_numcycles, temperature=temp, **{"calc_dir": ".", "stressdir": stressdir})
                os.chdir("../")
            os.chdir("../")

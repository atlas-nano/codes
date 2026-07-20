class DMAInputGenerator(LammpsInputGenerator):
    def createLammpsInputDMA(self, frequency, temperature=300, pressure=1, timestep_unit_fs=1, oscillation_amplitude_percentage=0.04, numcycles=10, datapoints_per_cycle=500, tilt_analysis=False, nve=False, stressdir=None, master_inputs=None, **kwargs):
        temperature = kwargs.get('structure_file', self.temperature)
        pressure = kwargs.get("pressure", self.pressure)
        calc_dir = kwargs.get("calc_dir", self.calc_dir)
        stressdir = kwargs.get("stressdir", None)
        os.chdir(calc_dir)
        timestep = timestep_unit_fs
        period = (1 / frequency) / (timestep * 1e-15)
        if not master_inputs:
            self.createLammpsInput(**{"calc_dir": calc_dir, "temperature": temperature})
            self.slurm_modifier(**{"temperature": temperature, "pressure": pressure})
            subprocess.call("rm *singlepoint", shell=True)
            in_line = int(subprocess.check_output("grep -n 'NVT production dynamics' in.* | cut -d : -f 1", shell=True))
            subprocess.call("head -n {} in.* > in_head".format(in_line), shell=True)
            subprocess.call("cat in_head $MASTER_FILES/in.master_mech_loss > in.{}".format(self.suffix), shell=True)
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

    def noise_filter_CLI(self, **kwargs):
        calc_dir = kwargs.get('calc_dir', self.calc_dir)
        os.chdir(calc_dir)
        temperature = kwargs.get('temperature', self.temperature)
        pressure = kwargs.get('pressure', self.pressure)
        if not "noise_filter_run" in os.listdir(calc_dir):
            os.mkdir("noise_filter_run")
        os.chdir("noise_filter_run")
        self.createLammpsInput(**{"calc_dir": calc_dir, "temperature": temperature})
        subprocess.call("sed -i 's/run                  5000000 # run for 15 ns/run                  500000 # run for 0.5 ns/g' in.*", shell=True)
        self.slurm_modifier(**{"temperature": temperature, "pressure": pressure})
        subprocess.call("rm *singlepoint", shell=True)
        os.chdir("../")

    def error_analysis_DMA(self, frequencies=[25e9], num_error_calcs=5, noise_filter_run=False, **kwargs):
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
            self.noise_filter_CLI(**{ "calc_dir": ".", "temperature": temperature, "pressure": pressure})
        for freq in frequencies:
            for i in range(num_error_calcs):
                prefix = "{}_mech_loss_{}K_{}_{}".format(self.suffix, temperature, frequency_dirname_parser(freq), i)
                print("Creating inputs for " + prefix)
                if not "{}_{}".format(frequency_dirname_parser(freq), i) in os.listdir("."):
                    os.mkdir("{}_{}".format(frequency_dirname_parser(freq), i))
                os.chdir("{}_{}".format(frequency_dirname_parser(freq), i))
                self.createLammpsInputDMA(freq, oscillation_amplitude_percentage=oscillation_amplitude_percentage, numcycles=numcycles, temperature=temperature, pressure=pressure, **{"calc_dir": ".", "stressdir": stressdir})
                os.chdir("../")



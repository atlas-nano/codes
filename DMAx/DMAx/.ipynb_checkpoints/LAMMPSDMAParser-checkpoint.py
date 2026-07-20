import re
import os
import subprocess
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from mastercurves import MasterCurve
from mastercurves.transforms import Multiply
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def extract_tot_eng(input_dir_path, output_file_path=None, plot=False, label=None, invert_sign=False):
    # Search for files with .log extension in the input directory
    log_files = [f for f in os.listdir(input_dir_path) if f.endswith('.log')]

    # Check if there are any .log files in the input directory
    if not log_files:
        raise ValueError('No .log files found in the input directory')

    # Use only the first .log file found in the directory
    input_file_path = os.path.join(input_dir_path, log_files[0])

    # Use a regular expression pattern to find all occurrences of TotEng and extract the associated value
    tot_eng_values = []
    tot_eng_values_floats = []
    energy_labels = []
    with open(input_file_path, "r") as input_file:
        count = 0
        for line in input_file:
            minimization, NVT_heat, NPT, cell_deform = 'print                "500 steps CG Minimization"', 'print                "NVT dynamics to heat system"', 'print                "NPT dynamics with an isotropic pressure of 1atm."', 'print                "deforming cell"'
            if minimization in line:
                count = 1
            if NVT_heat in line:
                count = 1
            if NPT in line:
                count = 1
            if cell_deform in line:
                count = 1
            if "TotEng   =" in line:
                if count == 1:
                    try:
                        tot_eng_values.append(line.split()[2])
                        tot_eng_values_floats.append(float(line.split()[2]))
                        energy_labels.append(1)
                        count = 0
                    except:
                        print(line.split()[2])
                        continue
                else:
                    try:
                        tot_eng_values.append(line.split()[2])
                        tot_eng_values_floats.append(float(line.split()[2]))
                        energy_labels.append(0)
                    except:
                        print(line.split()[2])
                        continue
        
    # Determine the output file path
    if output_file_path is None:
        output_file_path = os.path.join(os.path.dirname(input_file_path), 'TotEng.txt')

    if output_file_path is not None:
        # Open the output file for writing
        with open(output_file_path, 'w') as output_file:
            # Write the extracted TotEng values to the output file
            for value in tot_eng_values:
                output_file.write(value + '\n')
    
    # Plot the energy
    if plot:
        matching_indices = [i for i, val in enumerate(energy_labels) if val == 1]
        matching_values = [tot_eng_values_floats[i] for i in matching_indices]
        labels = ['Minimization', 'NVT Heat', 'NPT Start', 'NPT Deform']
        plt.title("Total Energy vs Runtime for NVT Heat / NPT Steps")
        plt.xlabel("Step")
        plt.ylabel("Total Energy (kcal/mole)")
        plt.ticklabel_format(style="scientific", scilimits=(0,4))
        plt.scatter(range(len(tot_eng_values_floats)), tot_eng_values_floats, label=label, s=1)
        plt.scatter(matching_indices, matching_values, c='y', marker='o', s=50)
        for i, label in zip(matching_indices, labels):
            plt.text(i, tot_eng_values_floats[i], label, ha='center', va='bottom', fontsize=8)
    return tot_eng_values_floats

def frequency_dirname_parser(frequency):
    freq_range_dict = {"Hz": 1e0, "KHz": 1e3, "MHz": 1e6, "GHz": 1e9, "THz": 1e12, "PHz": 1e15, "EHz": 1e18}
    for freq_mag, freq in freq_range_dict.items():
        dirname_val = frequency / freq
        if 1 <= dirname_val < 1000:
            return "{}{}".format(dirname_val, freq_mag)

def sort_dict_by_frequency(dict_to_sort):
    # Define a dictionary mapping frequency unit suffixes to their corresponding multipliers
    unit_multipliers = {'Hz': 1, 'kHz': 1e3, 'MHz': 1e6, 'GHz': 1e9, 'THz': 1e12}
    
    # Create a list of tuples where each tuple contains the numerical value of the frequency and the original key-value pair
    sorted_list = sorted([(float(k[:-3]) * unit_multipliers[k[-3:]], k, v) for k, v in dict_to_sort.items()])
    
    # Create a new dictionary with the sorted key-value pairs
    sorted_dict = {k: v for _, k, v in sorted_list}
    
    return sorted_dict


def plot_error_bars(*data_dicts):
    fig_width = len(data_dicts) * 5 # set width of figure based on number of data_dicts
    fig, axs = plt.subplots(1, len(data_dicts), figsize=(fig_width, 5), squeeze=False) # create one subplot for each data_dict
    yaxis_list = ["Storage Modulus", "Loss Modulus", "Loss Tangent"]
    for i, data_dict in enumerate(data_dicts):
        x_values = list(data_dict.keys())
        y_values = list(data_dict.values())

        for j, y in enumerate(y_values):
            mean = sum(y) / len(y)
            stdev = (sum([(val-mean)**2 for val in y]) / len(y))**0.5
            max_error = max(y) - min(y)
            axs[0, i].errorbar(x_values[j], mean, yerr=stdev, capsize=5, capthick=1, elinewidth=1, marker='s', markersize=5)
            axs[0, i].text(x_values[j], mean + stdev + 0.1*max_error, f"Max Error: {max_error:.4f}", ha='center', fontsize=6)

            # Add data points to error bars
            for k, val in enumerate(y):
                axs[0, i].plot([x_values[j]], [val], 'ko', markersize=2)

        axs[0, i].set_xlabel('Frequencies')
        axs[0, i].set_ylabel(yaxis_list[i])
    plt.suptitle("Storage Modulus, Loss Modulus, and Loss Tangent Error Analysis")
    plt.show()

class DMADataParser:
    def __init__(self, calc_dir):
        self.calc_dir = calc_dir
    
    def pressureparser(self, dielectric=False, noise_filter_run=False, output_xy=True, plot=False, plot_title="Pressure vs runtime", pressdir=None, apply_gaussian_filter=False, gaussian_sigma=6, force_pressure_remove=True, **kwargs):
        simulation_dir = kwargs.get("simulation_dir", self.calc_dir)
        # Function to generate a pressure.txt file with human-readable values for the pressures in all directions over the runtime of the simulation
        press_dict = {"x": "Pxx", "y": "Pyy", "z": "Pzz"}
        os.chdir(simulation_dir)
        if pressdir != None:
            pressdir = pressdir
            if dielectric or noise_filter_run:
                pressure_mult = 0.101325
            else:
                pressure_mult = -0.101325
        else:
            if dielectric:
                grepPressDir = subprocess.check_output("grep 'Field direction = ' in.*", shell=True).decode("utf-8").split()
                pressdir = grepPressDir[5]
                pressure_mult = 0.101325
            elif noise_filter_run:
                pressdir = None
                pressure_mult = 0.101325
            else:
                grepPressDir = subprocess.check_output("grep 'wiggle' in.*", shell=True).decode("utf-8").split()
                pressdir = grepPressDir[5]
                pressure_mult = -0.101325
        if "output.txt" not in os.listdir() or force_pressure_remove:
            log_file = [i for i in os.listdir(simulation_dir) if i[-3:] == "log"][0]
            with open(log_file, 'r') as file:
                lines = file.readlines()
            start_NVT_prod = [i for i, line in enumerate(lines) if 'Step          Time          TotEng         KinEng          Temp          PotEng' in line][0]
            end_NVT_prod = [i for i, line in enumerate(lines) if 'Loop time of' in line][-1]
            if start_NVT_prod and end_NVT_prod:
                NVT_prod_lines = lines[start_NVT_prod:end_NVT_prod]
                NVT_prod_lines = [line for line in NVT_prod_lines if "SHAKE" not in line and "Bond" not in line]
                NVT_prod_data = [line.split() for line in NVT_prod_lines]
                try:
                    NVT_prod_df = pd.DataFrame(NVT_prod_data[1:], columns=NVT_prod_data[0])
                except:
                    print("The simulation {} did not run successfully".format(simulation_dir))
                    return None, None
                NVT_prod_df.to_csv('output.txt', sep='\t', index=False)
        runtime = np.array(NVT_prod_df["Time"].astype(float))
        if noise_filter_run:
            pressures = [np.array(NVT_prod_df[pressdir].astype(float)) * pressure_mult for pressdir in press_dict.values()] #in MPa
            pressure = np.mean(pressures, axis=0)
        else:
            pressure = np.array(NVT_prod_df[press_dict[pressdir]].astype(float)) * pressure_mult #in MPa
        if apply_gaussian_filter:
            pressure = gaussian_filter1d(pressure, gaussian_sigma)
        if plot:
            plt.title(plot_title)
            plt.xlabel("Runtime (fs)")
            plt.ylabel("Pressure (MPa)")
            plt.ticklabel_format(style="scientific", scilimits=(0,4))
            plt.scatter(runtime, pressure, s=1)
        if output_xy:
            return runtime, pressure
    
    def fit_sin(self, runtime, pressure, dielectric=False, plot_pressure=False, plot_title="Pressure vs runtime (+ Scipy Fitting)", fit_color='red', original_signal_color='green', do_prints=False, **kwargs):
        '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
        simulation_dir = kwargs.get("simulation_dir", self.calc_dir)
        os.chdir(simulation_dir)
        if runtime is None and pressure is None:
            return {"Fit Curve": None, "Predicted Frequency": None, "Real Phi": None, "Loss Tangent": None, "Storage Modulus": None, "Loss Modulus": None, "MAE": None, "RMSE": None, "R2": None}      
        runtime = np.array(runtime)
        pressure = np.array(pressure)
        ff = np.fft.fftfreq(len(runtime), (runtime[1]-runtime[0]))   # assume uniform spacing
        if dielectric:
            guess_freq = 2 * 1 / float(subprocess.check_output("grep 'periodfs' in.*", shell=True).decode("utf-8").split()[3])
        else:
            guess_freq = 1 / float(subprocess.check_output("grep 'wiggle' in.*", shell=True).decode("utf-8").split()[8])
        guess_w = 2 * np.pi * guess_freq
        guess_amp = np.std(pressure) * 2.**0.5
        guess_offset = np.mean(pressure)
        guess = np.array([guess_amp, 0., guess_offset])

        def sinfunc(t, A, p, c):  return A * np.sin(guess_w * t + p) + c
        popt, pcov = sp.optimize.curve_fit(sinfunc, runtime, pressure, p0=guess)
        A, p, c = popt
        fitfunc = lambda t: A * np.sin(guess_w * t + p) + c
        y = fitfunc(runtime)
        if plot_pressure:
            plt.scatter(runtime, pressure, s=1, label="Pressure data")
            plt.ticklabel_format(style="scientific", scilimits=(0,4))
            plt.title(plot_title)
            plt.xlabel("Runtime (fs)")
            plt.ylabel("Pressure (MPa)")
            plt.plot(runtime, y, fit_color, label="Scipy fitted curve")
            plt.plot(runtime, A * np.sin(guess_w * runtime) + c, original_signal_color, label="Original Field Applied", linestyle="--")
            plt.legend()
        sin_offset = p
        mae, rmse, r2 = mean_absolute_error(pressure, y), mean_squared_error(pressure, y, squared=False), r2_score(pressure, y)
        if do_prints:
            print('Scipy Fitted Parameters:')
            #print(tpe_best)
            print("Scipy Angular Frequency (w): " + str(guess_w))
            print("Scipy Offset (a0): " + str(c))
            print("Scipy Amplitude (a1): " + str(A))
            print("Scipy Predicted Frequency (f): " + str(guess_w / (2 * np.pi) * 1e6) + " GHz")
            print("Scipy Real 'phi': " + str(sin_offset))
            print("Scipy Loss Tangent is: " + str(np.tan(sin_offset)))
            print("MAE, RMSE, R2: {}, {}, {}".format(mae, rmse, r2))
        return {"Original Curve": pressure, "Fit Curve": y, "Predicted Frequency": guess_w / (2 * np.pi) * 1e6, "Real Phi (rad)": sin_offset, "Real Phi (deg)": sin_offset * 180 / np.pi, "Loss Tangent": np.tan(sin_offset), "Storage Modulus": np.cos(sin_offset), "Loss Modulus": np.sin(sin_offset), "MAE": mae, "RMSE": rmse, "R2": r2}        

class DMAErrorDataParser(DMADataParser):
    def error_parser(self, add_savgol_filter=False, add_gaussian_filter=False):
        os.chdir(self.calc_dir)
        os.chdir("frequencies")
        if "noise_filter_run" in os.listdir("."):
            os.chdir("noise_filter_run")
            runtime_nfr, pressure_nfr = self.pressureparser(".", add_savgol_filter=add_savgol_filter, add_gaussian_filter=add_gaussian_filter)
            os.chdir("../")
            freq_directories = os.listdir(".").pop("noise_filter_run")
        else:
            freq_directories = os.listdir(".")
        unique_frequencies = [i.split("_")[0] for i in freq_directories]
        unique_frequencies = set(unique_frequencies)
        SM_dict = sort_dict_by_frequency({i:[] for i in unique_frequencies})
        LM_dict = sort_dict_by_frequency({i:[] for i in unique_frequencies})
        LT_dict = sort_dict_by_frequency({i:[] for i in unique_frequencies})
        for freq in freq_directories:
            frequency = freq.split("_")[0]
            os.chdir(freq)
            runtime, pressure = self.pressureparser(add_savgol_filter=add_savgol_filter, add_gaussian_filter=add_gaussian_filter, **{"simulation_dir": os.getcwd()})
            curve_fitting = self.fit_sin(runtime, pressure, **{"simulation_dir": os.getcwd()})
            SM_dict[frequency].append(curve_fitting["Storage Modulus"])
            LM_dict[frequency].append(curve_fitting["Loss Modulus"])
            LT_dict[frequency].append(curve_fitting["Loss Tangent"])
            os.chdir("../")
        return SM_dict, LM_dict, LT_dict
    
    def error_plotter(self, SM_dict, LM_dict, LT_dict, title="DMA Error Margin Analysis"):
        plot_error_bars(SM_dict, LM_dict, LT_dict)
            
class DMAMasterCurveDataParser(DMADataParser):
    def master_curve_parser(self, add_savgol_filter=False, add_gaussian_filter=False):
        final_dict_storage, final_dict_loss, final_dict_tangent = {}, {}, {}
        os.chdir(self.calc_dir)
        os.chdir("temperatures")
        for temp_dir in os.listdir("."):
            temp = float(re.sub("[^0-9.]", "", temp_dir))
            final_dict_storage[temp], final_dict_loss[temp], final_dict_tangent[temp] = {}, {}, {}
            os.chdir(temp_dir)
            #if "noise_filter_run" in os.listdir("."):
             #   os.chdir("noise_filter_run")
              #  print(os.listdir("."))
               # runtime_nfr, pressure_nfr, calc_dir_nfr = self.pressureparser(".", pressdir="x", add_savgol_filter=add_savgol_filter, add_gaussian_filter=add_gaussian_filter)
                #os.chdir("../")
            freq_directories = os.listdir(".")
            if "noise_filter_run" in freq_directories:
                freq_directories.remove("noise_filter_run")
            #else:
            for freq_dir in freq_directories:
                freq = float(re.sub("[^0-9.]", "", freq_dir)) * 1e9
                os.chdir(freq_dir)
                try:
                    runtime, press = self.pressureparser(apply_savgol_filter=add_savgol_filter, apply_gaussian_filter=add_gaussian_filter, **{"simulation_dir": os.getcwd()})
                    results = self.fit_sin(runtime, press, **{"simulation_dir": os.getcwd()})
                    final_dict_storage[temp][freq] = results["Storage Modulus"]
                    final_dict_loss[temp][freq] = results["Loss Modulus"]
                    final_dict_tangent[temp][freq] = results["Loss Tangent"]
                    os.chdir("../")
                except:
                    final_dict_storage[temp][freq] = 0.99
                    final_dict_loss[temp][freq] = 0.1
                    final_dict_tangent[temp][freq] = 0.2
                    os.chdir("../")
            os.chdir("../")
        df_storage, df_loss, df_tangent = pd.DataFrame(final_dict_storage), pd.DataFrame(final_dict_loss), pd.DataFrame(final_dict_tangent)
        df_storage.columns.name, df_loss.columns.name, df_tangent.columns.name = 'Temperature (K)', 'Temperature (K)', 'Temperature (K)'
        df_storage.index.name, df_loss.index.name, df_tangent.index.name = 'Frequency (Hz)', 'Frequency (Hz)', 'Frequency (Hz)'
        df_storage, df_loss, df_tangent = df_storage.sort_index(axis=0), df_loss.sort_index(axis=0), df_tangent.sort_index(axis=0)  # sort rows
        df_storage, df_loss, df_tangent = df_storage.sort_index(axis=1), df_loss.sort_index(axis=1), df_tangent.sort_index(axis=1)  # sort columns
        return df_storage, df_loss, df_tangent
    
    def master_curve_plotter(self, *data):
        for data in data:
            x_data = [np.array(data.index) for i in data.columns]
            y_data = [np.array(data[i]) for i in data.columns]
            x_data = [np.log(xi) for xi in x_data]
            y_data = [np.log(ci) for ci in y_data]
            states = [i for i in np.array(data.columns)]
            mc = MasterCurve(fixed_noise = 0)
            mc.add_data(x_data, y_data, states)
            mc.add_htransform(Multiply())
            mc.add_vtransform(Multiply())
            mc.superpose()
            fig1, ax1, fig2, ax2, fig3, ax3 = mc.plot(colormap = lambda i: plt.cm.Blues_r(i/1.5))
            ax1.set_xscale('log')
            ax1.set_yscale('linear')
            ax2.set_xscale('log')
            ax2.set_yscale('linear')
            ax3.set_xscale('log')
            ax3.set_yscale('linear')
            
class ConvAnalysisParser(DMADataParser):
    
    def runtime_conv_analysis(self, threshold=0.1, add_gaussian_filter=False, gaussian_sigma=6):
        os.chdir(self.calc_dir)
        os.chdir("frequencies")
        if "noise_filter_run" in os.listdir("."):
            os.chdir("noise_filter_run")
            os.chdir("../")
            freq_directories = os.listdir(".")
            freq_directories.remove("noise_filter_run")
        else:
            freq_directories = os.listdir(".")
            
        for freq_dir in freq_directories:
            os.chdir(freq_dir) # Changing into strain directory 
            os.chdir(os.listdir(".")[0])
            num_cycles = []
            LT_list = []
            diff_list = []
            optimalcycles_lst = []
            timesteps, pressure = self.pressureparser(**{"simulation_dir": os.getcwd()})
            period = int(subprocess.check_output("grep 'wiggle' in.*", shell=True).decode("utf-8").split()[8])
            dump_freq = int(subprocess.check_output("grep 'thermo' in.*", shell=True).decode("utf-8").split()[-1])
            runtime = int(subprocess.check_output("grep 'run' in.*", shell=True).decode("utf-8").split()[-1])
            calc_cycles =  int(runtime / period)
            datapoints_per_cycle = round(period / dump_freq)
            final_result = []
            old_results = 0
            for cycles in range(1, calc_cycles + 1):
                sliced_runtime, sliced_pressure = timesteps[:datapoints_per_cycle * cycles], pressure[:datapoints_per_cycle * cycles]
                results = self.fit_sin(sliced_runtime, sliced_pressure, **{"simulation_dir": os.getcwd()})["Loss Tangent"]
                if old_results == 0 or abs(results - old_results) >= 0.1:
                    old_results = results
                else:
                    final_result.append(cycles)
                LT_list.append(results)
                num_cycles.append(cycles)
            plt.subplot(1,2,1)
            plt.plot(num_cycles, LT_list)
            plt.title("Loss Tangent vs. Total Cycles")
            plt.xlabel('Number of Cycles ')
            plt.ylabel('Loss Tangent')
            print("Optimal Number of Cycles", final_result[0])
            os.chdir("../")    
    
    def strainsize_conv_analysis(self, threshold=0.1, add_gaussian_filter=False, gaussian_sigma=6):
        os.chdir(self.calc_dir)
        os.chdir("frequencies")
        if "noise_filter_run" in os.listdir("."):
            freq_directories_strain = os.listdir(".")
            freq_directories_strain.remove("noise_filter_run")
        else:
            freq_directories_strain = (os.listdir("."))
            #Since, there is going to only one frequency value
        for frequency in freq_directories_strain:
            os.chdir(frequency)
            #Getting the list of strain subdirectories within frequency directory
            strain_lst = []
            LT_strainlst = []
            LM_strainlst = []
            SM_strainlst = []
            R2_strainlst = []
            RMSE_strainlst = []
            original_strainlst = []
            for subdir in os.listdir("."):
                strain_val = float(subdir[:-9])
                strain_lst.append(strain_val)
                os.chdir(subdir)
                timesteps_strain, pressure_strain = self.pressureparser(**{"simulation_dir": os.getcwd()}, apply_gaussian_filter=add_gaussian_filter, gaussian_sigma=gaussian_sigma)
                results_strain = self.fit_sin(timesteps_strain, pressure_strain, **{"simulation_dir": os.getcwd()})
                SM_strainlst.append(results_strain["Storage Modulus"])
                LM_strainlst.append(results_strain["Loss Modulus"])
                LT_strainlst.append(results_strain["Loss Tangent"])
                R2_strainlst.append(results_strain["R2"])
                RMSE_strainlst.append(results_strain["RMSE"])
                #original_strainlst.append(results_strain["Original Curve"])
               # print(strain_lst, R2_strainlst)
                os.chdir("../")

        # Sort the SM, LM, and LT data in ascending order
        sorted_data = sorted(zip(strain_lst, SM_strainlst, LM_strainlst, LT_strainlst, R2_strainlst, RMSE_strainlst))
        strain_lst, SM_strainlst, LM_strainlst, LT_strainlst, R2_strainlst, RMSE_strainlst = map(list, zip(*sorted_data))
        for i in range(len(strain_lst)):
            if LT_strainlst[i] == None:
                strain_lst[i] = None
        def remove_none_elements(*lists):
            filtered_lists = []
            for lst in lists:
                filtered_lists.append([element for element in lst if element is not None])
            return tuple(filtered_lists)
        strain_lst, LT_strainlst, LM_strainlst, SM_strainlst, R2_strainlst, RMSE_strainlst = remove_none_elements(strain_lst, LT_strainlst, LM_strainlst, SM_strainlst, R2_strainlst, RMSE_strainlst)
        
        
        # Create the figure and subplots
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex=True, figsize=(5, 10))

        # Plot the data on each subplot
        ax1.plot(strain_lst, SM_strainlst, '--o')
        ax2.plot(strain_lst, LM_strainlst, '--o')
        ax3.plot(strain_lst, LT_strainlst, '--o')
        ax4.plot(strain_lst, R2_strainlst, '--o')
        ax5.plot(strain_lst, RMSE_strainlst, '--o')

        # Set labels and titles for each subplot
        ax1.set_ylabel('Storage Modulus')
        ax2.set_ylabel('Loss Modulus')
        ax3.set_ylabel('Loss Tangent')
        ax4.set_ylabel('R2 Score')
        ax5.set_ylabel('RMSE')
        ax5.set_xlabel('Strain (%)')
        ax5.set_xscale('log')

        # Print out the best strain based on R2 score and RMSE
        def find_lowest_ranked_value(x, r2, rmse):
            """
            Finds the value in the list "x" that has the lowest ranking sum of its corresponding R2 and RMSE values.

            Arguments:
            x -- List of values.
            r2 -- List of R2 values (same length as x).
            rmse -- List of RMSE values (same length as x).

            Returns:
            The value in x that has the lowest ranking sum of its corresponding R2 and RMSE values.
            """
            ranks = []
            for i in range(len(x)):
                r2_rank = len(r2) - sorted(r2).index(r2[i])
                rmse_rank = sorted(rmse).index(rmse[i])
                ranks.append(r2_rank + rmse_rank)
            min_rank_idx = ranks.index(min(ranks))
            return x[min_rank_idx]
        
        print("Best strain size: {}".format(find_lowest_ranked_value(strain_lst, R2_strainlst, RMSE_strainlst)))

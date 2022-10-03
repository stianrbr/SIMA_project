##
from ReadTimeSeriesResults import unpackBinaryResults, readSIMO_resultstext, getchannelNumbers, unpackSIMOresults
from utility import Betz, Cp
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
try:
    os.mkdir("Results")
except FileExistsError:
    pass

plt.style.use("seaborn")
plt.rcParams["figure.figsize"] = (15, 4)

# ============================================================================#
# Inputs
SIMAfol_Rotor = "C:\\DNV\\Workspaces\\SIMA\\Constant Wind\\RotorOnly12MW\\Initial1\\1231-20220930084514\\"
SIMAfol_Full = "C:\\DNV\\Workspaces\\SIMA\\Constant Wind\\INO_WINDMOOR12MW\\Initial1\\1220-20220929151532\\"
# Total simulation time and storage time step

dt = 0.1 # Storage time step
SimulationTime = 15300 # Simulation time in second
NtRiflexForce = SimulationTime/dt  # number of stored time steps on the force file (=simulationTime/dt)
hullBody = 1  # this is the number of the hull body (order in the SIMA inputs)



# ---------------------------------------------RIFLEX results---------------------------------------------#
# Forces
# Path + file name
ForcefileName_Rotor = SIMAfol_Rotor + 'sima_witurb.bin'
A_Rotor = unpackBinaryResults(ForcefileName_Rotor, NtRiflexForce, 0)

ForcefileName_Full = SIMAfol_Full + 'sima_witurb.bin'
A_Full = unpackBinaryResults(ForcefileName_Full, NtRiflexForce, 0)

# Time vector
time_RIFLEX_Rotor = A_Rotor[1]
time_RIFLEX_Full = A_Rotor[1]

# NB! Forces from RIFLEX are in kN!

# Extract forces from tower and blade roots, assuming that force storage is
# specified with tower base, then tower top, then mooring lines

TowerBaseAxial_Rotor = A_Rotor[2]  # Axial force
TowerBaseTors_Rotor = A_Rotor[3]  # Torsional moment
TowerBaseBMY_Rotor = A_Rotor[4]  # bending moment in local y
TowerBaseBMZ_Rotor = A_Rotor[6]  # bending moment in local z
TowerBaseShearY_Rotor = A_Rotor[8]  # shear force in local y
TowerBaseShearZ_Rotor = A_Rotor[10]  # shear force in local z

TowerTopAxial_Rotor = A_Rotor[12]
TowerTopTors_Rotor = A_Rotor[13]
TowerTopBMY_Rotor = A_Rotor[15]  # end 2
TowerTopBMZ_Rotor = A_Rotor[17]
TowerTopShearY_Rotor = A_Rotor[19]
TowerTopShearZ_Rotor = A_Rotor[21]

bl1Axial_Rotor = A_Rotor[22]
bl1Tors_Rotor = A_Rotor[23]
bl1BMY_Rotor = A_Rotor[24]
'''bl1BMZ           = A[26]
bl1ShearY        = A[28]
bl1ShearZ        = A[30]

bl2Axial         = A[32]
bl2Tors          = A[33]
bl2BMY           = A[34]
bl2BMZ           = A[36]
bl2ShearY        = A[38]
bl2ShearZ        = A[40]'''

TowerBaseAxial_Full = A_Full[2]  # Axial force
TowerBaseTors_Full = A_Full[3]  # Torsional moment
TowerBaseBMY_Full = A_Full[4]  # bending moment in local y
TowerBaseBMZ_Full = A_Full[6]  # bending moment in local z
TowerBaseShearY_Full = A_Full[8]  # shear force in local y
TowerBaseShearZ_Full = A_Full[10]  # shear force in local z

TowerTopAxial_Full = A_Full[12]
TowerTopTors_Full = A_Full[13]
TowerTopBMY_Full = A_Full[15]  # end 2
TowerTopBMZ_Full = A_Full[17]
TowerTopShearY_Full = A_Full[19]
TowerTopShearZ_Full = A_Full[21]

bl1Axial_Full = A_Full[22]
bl1Tors_Full = A_Full[23]
bl1BMY_Full = A_Full[24]
'''bl1BMZ           = A[26]
bl1ShearY        = A[28]
bl1ShearZ        = A[30]

bl2Axial         = A[32]
bl2Tors          = A[33]
bl2BMY           = A[34]
bl2BMZ           = A[36]
bl2ShearY        = A[38]
bl2ShearZ        = A[40]'''

# wind turbine results
WTfileName_Rotor = SIMAfol_Rotor + 'sima_witurb.bin'

A_Rotor = unpackBinaryResults(WTfileName_Rotor, 0, 26)

time_WT_Rotor = A_Rotor[1]  # it is possible that this time vector differs from t
omega_Rotor = A_Rotor[2] * np.pi / 180  # convert from deg/s to rad/s
rpm_Rotor = A_Rotor[3]
genTq_Rotor = A_Rotor[4]
genPwr_Rotor = A_Rotor[5]
azimuth_Rotor = A_Rotor[6]
HubWindX_Rotor = A_Rotor[7]
HubWindY_Rotor = A_Rotor[8]
HubWindZ_Rotor = A_Rotor[9]
AeroForceX_Rotor = A_Rotor[10]
AeroMomX_Rotor = A_Rotor[13]
Bl1Pitch_Rotor = A_Rotor[16]
Bl2Pitch_Rotor = A_Rotor[17]
Bl3Pitch_Rotor = A_Rotor[18]

WTfileName_Full = SIMAfol_Full + 'sima_witurb.bin'

A_Full = unpackBinaryResults(WTfileName_Full, 0, 26)

time_WT_Full = A_Full[1]  # it is possible that this time vector differs from t
omega_Full = A_Full[2] * np.pi / 180  # convert from deg/s to rad/s
rpm_Full = A_Full[3]
genTq_Full = A_Full[4]
genPwr_Full = A_Full[5]
azimuth_Full = A_Full[6]
HubWindX_Full = A_Full[7]
HubWindY_Full = A_Full[8]
HubWindZ_Full = A_Full[9]
AeroForceX_Full = A_Full[10]
AeroMomX_Full = A_Full[13]
Bl1Pitch_Full = A_Full[16]
Bl2Pitch_Full = A_Full[17]
Bl3Pitch_Full = A_Full[18]

# ---------------------------------------------SIMO results---------------------------------------------#
# #Path + file name
fileName_Rotor = SIMAfol_Rotor + 'results.tda'
fileNametxt_Rotor = SIMAfol_Rotor + 'results.txt'
nchan_Rotor, nts_Rotor, dt_simo_Rotor, chanNames_Rotor = readSIMO_resultstext(fileNametxt_Rotor)

fileName_Full = SIMAfol_Full + 'results.tda'
fileNametxt_Full = SIMAfol_Full + 'results.txt'
nchan_Full, nts_Full, dt_simo_Full, chanNames_Full = readSIMO_resultstext(fileNametxt_Full)


AA_Rotor = unpackSIMOresults(fileName_Rotor, nts_Rotor)
# % Determine which channels to read for the platform motions, wave elevation
chanMotions_Rotor, chanWave_Rotor, chanAcc_Rotor = getchannelNumbers(chanNames_Rotor, hullBody)

time_SIMO_Rotor = AA_Rotor[1, :]
# summarize data in matrix
PlatMotions_Rotor = AA_Rotor[chanMotions_Rotor, :]
wave_Rotor = AA_Rotor[chanWave_Rotor, :]

AA_Full = unpackSIMOresults(fileName_Full, nts_Full)
# % Determine which channels to read for the platform motions, wave elevation
chanMotions_Full, chanWave_Full, chanAcc_Full = getchannelNumbers(chanNames_Full, hullBody)

time_SIMO_Full = AA_Full[1, :]
# summarize data in matrix
PlatMotions_Full = AA_Full[chanMotions_Full, :]
wave_Full = AA_Full[chanWave_Full, :]

platform_surge = PlatMotions_Full[0]
platform_roll = PlatMotions_Full[3]
platform_pitch = PlatMotions_Full[4]

# --------------------------------------------end read section------------------------------------------#
# ------------------------------------------------------------------------------------------------------#
##
# Start and end values are hardcoded from inspection of dataset
starts_Rotor = [627, 2400, 3450, 4650, 5760, 6960, 8140, 9350, 10550, 11760, 12950, 14140]
ends_Rotor = [2000, 3280, 4490, 5690, 6890, 8060, 9270, 10460, 11660, 12870, 14060, 15260]

starts_Full = [627, 2400, 3450, 4650, 5860, 7100, 8340, 9450, 10750, 11860, 13050, 14240]
ends_Full = [2000, 3180, 4390, 5590, 6790, 7960, 9170, 10560, 11560, 12770, 13960, 15160]
starts_Rotor = starts_Full
ends_Rotor = ends_Full

# Taking mean of SIMA results for the identified intervals
rpm_mean_rotor = np.array([np.mean(rpm_Rotor[int(start / dt):int(end / dt)]) for start, end in zip(starts_Rotor, ends_Rotor)])
thrust_mean_rotor = np.array([np.mean(AeroForceX_Rotor[int(start / dt):int(end / dt)]) for start, end in zip(starts_Rotor, ends_Rotor)])
wind_speed_mean_rotor = np.array([np.mean(HubWindX_Rotor[int(start / dt):int(end / dt)]) for start, end in zip(starts_Rotor, ends_Rotor)])
genTq_mean_rotor =np.array([np.mean(genTq_Rotor[int(start / dt):int(end / dt)]) for start, end in zip(starts_Rotor, ends_Rotor)])
genPwr_mean_rotor =np.array([np.mean(genPwr_Rotor[int(start / dt):int(end / dt)]) for start, end in zip(starts_Rotor, ends_Rotor)])
Bl1Pitch_mean_rotor = np.array([np.mean(Bl1Pitch_Rotor[int(start / dt):int(end / dt)]) for start, end in zip(starts_Rotor, ends_Rotor)])
Bl2Pitch_mean_rotor = np.array([np.mean(Bl2Pitch_Rotor[int(start / dt):int(end / dt)]) for start, end in zip(starts_Rotor, ends_Rotor)])
Bl3Pitch_mean_rotor = np.array([np.mean(Bl3Pitch_Rotor[int(start / dt):int(end / dt)]) for start, end in zip(starts_Rotor, ends_Rotor)])

rpm_mean_Full = np.array([np.mean(rpm_Full[int(start / dt):int(end / dt)]) for start, end in zip(starts_Full, ends_Full)])
thrust_mean_Full = np.array([np.mean(AeroForceX_Full[int(start / dt):int(end / dt)]) for start, end in zip(starts_Full, ends_Full)])
wind_speed_mean_Full = np.array([np.mean(HubWindX_Full[int(start / dt):int(end / dt)]) for start, end in zip(starts_Full, ends_Full)])
genTq_mean_Full =np.array([np.mean(genTq_Full[int(start / dt):int(end / dt)]) for start, end in zip(starts_Full, ends_Full)])
genPwr_mean_Full =np.array([np.mean(genPwr_Full[int(start / dt):int(end / dt)]) for start, end in zip(starts_Full, ends_Full)])
Bl1Pitch_mean_Full = np.array([np.mean(Bl1Pitch_Full[int(start / dt):int(end / dt)]) for start, end in zip(starts_Full, ends_Full)])
Bl2Pitch_mean_Full = np.array([np.mean(Bl2Pitch_Full[int(start / dt):int(end / dt)]) for start, end in zip(starts_Full, ends_Full)])
Bl3Pitch_mean_Full = np.array([np.mean(Bl3Pitch_Full[int(start / dt):int(end / dt)]) for start, end in zip(starts_Full, ends_Full)])

rpm_stddev_Full = np.array([np.std(rpm_Full[int(start / dt):int(end / dt)],ddof=1) for start, end in zip(starts_Full, ends_Full)])
thrust_stddev_Full = np.array([np.std(AeroForceX_Full[int(start / dt):int(end / dt)],ddof=1) for start, end in zip(starts_Full, ends_Full)])
wind_speed_stddev_Full = np.array([np.std(HubWindX_Full[int(start / dt):int(end / dt)],ddof=1) for start, end in zip(starts_Full, ends_Full)])
genTq_stddev_Full =np.array([np.std(genTq_Full[int(start / dt):int(end / dt)],ddof=1) for start, end in zip(starts_Full, ends_Full)])
genPwr_stddev_Full =np.array([np.std(genPwr_Full[int(start / dt):int(end / dt)],ddof=1) for start, end in zip(starts_Full, ends_Full)])
Bl1Pitch_stddev_Full = np.array([np.std(Bl1Pitch_Full[int(start / dt):int(end / dt)],ddof=1) for start, end in zip(starts_Full, ends_Full)])
Bl2Pitch_stddev_Full = np.array([np.std(Bl2Pitch_Full[int(start / dt):int(end / dt)],ddof=1) for start, end in zip(starts_Full, ends_Full)])
Bl3Pitch_stddev_Full = np.array([np.std(Bl3Pitch_Full[int(start / dt):int(end / dt)],ddof=1) for start, end in zip(starts_Full, ends_Full)])

fast = pd.read_excel("Wind_files\\WM12MW_OpenFAST.xlsx", engine="openpyxl")
wind1VelX_fast = fast["Wind1VelX"].to_numpy()[1:]
RtAeroFxh_fast = fast["RtAeroFxh"].to_numpy()[1:]
Rotor_speed_fast = fast["RotSpeed"].to_numpy()[1:]
BldPitch_fast = fast["BldPitch1"].to_numpy()[1:]
GenTq_fast = fast["GenTq"].to_numpy()[1:]
GenSpeed_fast = fast["GenSpeed"].to_numpy()[1:]
GenPwr_fast = fast["GenPwr"].to_numpy()[1:]

##
"""
Plotting of the RPM for the step wind case.
"""

plt.plot(time_WT_Rotor, rpm_Rotor, linewidth=1, c="blue", label="Fixed", zorder=2)
plt.plot(time_WT_Full, rpm_Full, linewidth=0.6 , c="black", label="Floating", zorder=1)
for i, (start, end) in enumerate(zip(starts_Rotor, ends_Rotor)):
    if i ==0:
        plt.vlines(x=time_WT_Rotor[int(start / dt)], ymin=np.mean(rpm_Rotor[int(start / dt):int(end / dt)]) - 1, ymax=np.mean(rpm_Rotor[int(start / dt):int(end / dt)]) + 1, linestyles="dotted", colors="green", label="Start steady state")
        plt.vlines(x=time_WT_Rotor[int(end / dt)], ymin=np.mean(rpm_Rotor[int(start / dt):int(end / dt)]) - 1, ymax=np.mean(rpm_Rotor[int(start / dt):int(end / dt)]) + 1, linestyles="dotted", colors="black", label="End steady state")
        plt.hlines(y=np.mean(rpm_Rotor[int(start / dt):int(end / dt)]), xmin=time_WT_Rotor[int(start / dt)], xmax=time_WT_Rotor[int(end / dt)], colors="red",
                   linestyles="dashed", label="Mean")
    else:
        plt.vlines(x=time_WT_Rotor[int(start / dt)], ymin=np.mean(rpm_Rotor[int(start / dt):int(end / dt)]) - 1, ymax=np.mean(rpm_Rotor[int(start / dt):int(end / dt)]) + 1, linestyles="dotted", colors="green")
        plt.vlines(x=time_WT_Rotor[int(end / dt)], ymin=np.mean(rpm_Rotor[int(start / dt):int(end / dt)]) - 1, ymax=np.mean(rpm_Rotor[int(start / dt):int(end / dt)]) + 1, linestyles="dotted", colors="black")
        plt.hlines(y=np.mean(rpm_Rotor[int(start / dt):int(end / dt)]), xmin=time_WT_Rotor[int(start / dt)], xmax=time_WT_Rotor[int(end / dt)], colors="red",
                   linestyles="dashed")

plt.xlabel("Time [s]")
plt.ylabel("Rotor speed [rpm]")
plt.legend(loc="best")
plt.title("Rotor RPM time history")
plt.tight_layout()
plt.savefig("Results\\rotor_speed_vs_time.png")
plt.show()
##
fig, ax = plt.subplots()
ax2 = ax.twinx()
ax2.plot(time_WT_Rotor[0:-2], HubWindX_Rotor[0:-2], linewidth=0.8, c="darkorange", label="Wind speed")
ax.plot(time_WT_Rotor[0:-2], AeroForceX_Rotor[0:-2], linewidth=1, c="blue", label="Fixed", zorder=2)
ax.plot(time_WT_Full[0:-2], AeroForceX_Full[0:-2], linewidth=0.6 , c="black", label="Floating", zorder=1)
for i, (start, end) in enumerate(zip(starts_Rotor, ends_Rotor)):
    if i ==0:
        ax.vlines(x=time_WT_Rotor[int(start / dt)], ymin=np.mean(AeroForceX_Rotor[int(start / dt):int(end / dt)]) - 1, ymax=np.mean(AeroForceX_Rotor[int(start / dt):int(end / dt)]) + 1, linestyles="dotted", colors="green", label="Start steady state")
        ax.vlines(x=time_WT_Rotor[int(end / dt)], ymin=np.mean(AeroForceX_Rotor[int(start / dt):int(end / dt)]) - 1, ymax=np.mean(AeroForceX_Rotor[int(start / dt):int(end / dt)]) + 1, linestyles="dotted", colors="black", label="End steady state")
        ax.hlines(y=np.mean(AeroForceX_Rotor[int(start / dt):int(end / dt)]), xmin=time_WT_Rotor[int(start / dt)], xmax=time_WT_Rotor[int(end / dt)], colors="red",
                   linestyles="dashed", label="Mean")
    else:
        ax.vlines(x=time_WT_Rotor[int(start / dt)], ymin=np.mean(AeroForceX_Rotor[int(start / dt):int(end / dt)]) - 1, ymax=np.mean(AeroForceX_Rotor[int(start / dt):int(end / dt)]) + 1, linestyles="dotted", colors="green")
        ax.vlines(x=time_WT_Rotor[int(end / dt)], ymin=np.mean(AeroForceX_Rotor[int(start / dt):int(end / dt)]) - 1, ymax=np.mean(AeroForceX_Rotor[int(start / dt):int(end / dt)]) + 1, linestyles="dotted", colors="black")
        ax.hlines(y=np.mean(AeroForceX_Rotor[int(start / dt):int(end / dt)]), xmin=time_WT_Rotor[int(start / dt)], xmax=time_WT_Rotor[int(end / dt)], colors="red",
                   linestyles="dashed")

ax.set_xlabel("Time [s]")
ax.set_ylabel("Force [kN]")
ax2.grid()
ax2.annotate("8 m/s", xy=(3700, 8), xycoords="data", fontsize=10)
ax2.annotate("Rated:\n10.6 m/s", xy=(6100, 10.6), xycoords="data", fontsize=10)
ax2.annotate("18 m/s", xy=(10900, 18), xycoords="data", fontsize=10)
ax2.set_ylabel("Wind speed [m/s]")
ax.legend(loc="upper left")
ax2.legend(loc="center left")
fig.suptitle("Thrust force time history")
plt.tight_layout()
plt.savefig("Results\\thrust_vs_time.png")
plt.show()

##
"""
Plotting of mean RPM versus wind speed
"""

plt.plot(wind_speed_mean_rotor, rpm_mean_rotor, 'r-o', label="SIMA - Fixed")
plt.plot(wind_speed_mean_Full, rpm_mean_Full, 'k-o', linewidth=0.8, label="SIMA - Floating")
plt.plot(wind1VelX_fast, Rotor_speed_fast, "b-X", label="FAST")
plt.errorbar(x=wind_speed_mean_Full, y=rpm_mean_Full, yerr=rpm_stddev_Full, color="black", fmt="none", label="Std.dev.")
plt.xlabel("Wind speed [m/s]")  # Endre denne til incomming?
plt.ylabel("Rotor speed [rpm]")
plt.legend(loc="best")
plt.title("Mean rotor speed vs. wind speed")
plt.tight_layout()
plt.savefig("Results\\mean_rotor_speed_vs_velocity.png")
plt.show()

##
"""
Plotting of mean thrust versus wind speed
"""

plt.plot(wind_speed_mean_rotor, thrust_mean_rotor, 'r-o', label="SIMA - Fixed")
plt.plot(wind_speed_mean_Full, thrust_mean_Full, 'k-o', linewidth=0.8, label="SIMA - Floating")
plt.plot(wind1VelX_fast, RtAeroFxh_fast/1000, "b-X", label="FAST")
plt.xlabel("Wind speed [m/s]")
plt.ylabel("Thrust force [kN]")
plt.legend(loc="best")
plt.title("Mean thrust vs. wind speed")
plt.tight_layout()
plt.savefig("Results\\mean_thrust_vs_velocity.png")
plt.show()

##
"""
Plotting of mean torque versus wind speed
"""

plt.plot(wind_speed_mean_rotor, genTq_mean_rotor, 'r-o', label="SIMA")
plt.plot(wind_speed_mean_Full, genTq_mean_Full, 'k-o', linewidth=0.8, label="SIMA")
plt.plot(wind1VelX_fast, GenTq_fast, "b-X", label="FAST")
plt.xlabel("Wind speed [m/s]")
plt.ylabel("Torque [kNm]")
plt.legend(loc="best")
plt.title("Mean torque vs. wind speed")
plt.tight_layout()
plt.savefig("Results\\mean_torque_vs_velocity.png")
plt.show()

##
"""
Plotting of mean power versus wind speed
"""

plt.plot(wind_speed_mean_rotor, genPwr_mean_rotor, 'r-o', label="SIMA - Fixed")
plt.plot(wind_speed_mean_Full, genPwr_mean_Full, 'k-o', linewidth=0.8, label="SIMA - Floating")
plt.plot(wind1VelX_fast, GenPwr_fast, "b-X", label="FAST")
plt.xlabel("Wind speed [m/s]")
plt.ylabel("Power [kW]")
plt.legend(loc="best")
plt.title("Mean power vs. wind speed")
plt.tight_layout()
plt.savefig("Results\\mean_power_vs_velocity.png")
plt.show()

##
"""
Extra: Plotting of mean power coeff versus wind speed
"""

plt.plot(wind_speed_mean_rotor, genPwr_mean_rotor * 1000 / Cp(rho_air=1.2, d=216.9, u=wind_speed_mean_rotor), 'r-o', label="SIMA")
plt.plot(wind1VelX_fast, GenPwr_fast*1000/Cp(rho_air=1.2, d=216.9, u=wind1VelX_fast), "b-X", label="FAST")
plt.plot(wind_speed_mean_rotor, Betz(rho_air=1.2, d=216.9, u=wind_speed_mean_rotor) / Cp(rho_air=1.2, d=216.9, u=wind_speed_mean_rotor), ls="dotted", c="black", label="Betz' limit")
plt.xlabel("Wind speed [m/s]")
plt.ylabel("Power coefficient [-]")
plt.legend(loc="best")
plt.title("Mean Cp vs. wind speed")
plt.tight_layout()
plt.savefig("Results\\mean_power_coeff_vs_velocity.png")
plt.show()

##
"""
Plotting of mean blade pitch versus mean wind speed
"""

plt.plot(wind_speed_mean_rotor, Bl1Pitch_mean_rotor, 'r-o', label="SIMA - Fixed")
plt.plot(wind_speed_mean_Full, Bl1Pitch_mean_Full, "k-o", linewidth=0.8, label="SIMA - Floating")
plt.plot(wind1VelX_fast, BldPitch_fast, "b-X", label="FAST")
plt.xlabel("Wind speed [m/s]")
plt.ylabel("Blade Pitch [deg]")
plt.legend(loc="best")
plt.title("Mean blade pitch vs. wind speed")
plt.tight_layout()
plt.savefig("Results\\mean_blade_pitch_vs_velocity.png")
plt.show()

##
"""
Plotting of floater displacement
"""
plt.plot(time_SIMO_Full, platform_surge*-1, "k", linewidth=0.8, label="Surge motion")
for i, (start, end) in enumerate(zip(starts_Rotor, ends_Rotor)):
    if i ==0:
        plt.vlines(x=time_SIMO_Full[int(start / dt)], ymin=np.mean(platform_surge[int(start / dt):int(end / dt)]*-1) - 1, ymax=np.mean(platform_surge[int(start / dt):int(end / dt)]*-1) + 1, linestyles="dotted", colors="green", label="Start steady state")
        plt.vlines(x=time_SIMO_Full[int(end / dt)], ymin=np.mean(platform_surge[int(start / dt):int(end / dt)]*-1) - 1, ymax=np.mean(platform_surge[int(start / dt):int(end / dt)]*-1) + 1, linestyles="dotted", colors="black", label="End steady state")
        plt.hlines(y=np.mean(platform_surge[int(start / dt):int(end / dt)]*-1), xmin=time_SIMO_Full[int(start / dt)], xmax=time_SIMO_Full[int(end / dt)], colors="red",
                   linestyles="dashed", label="Mean")
    else:
        plt.vlines(x=time_SIMO_Full[int(start / dt)], ymin=np.mean(platform_surge[int(start / dt):int(end / dt)]*-1) - 1, ymax=np.mean(platform_surge[int(start / dt):int(end / dt)]*-1) + 1, linestyles="dotted", colors="green")
        plt.vlines(x=time_SIMO_Full[int(end / dt)], ymin=np.mean(platform_surge[int(start / dt):int(end / dt)]*-1) - 1, ymax=np.mean(platform_surge[int(start / dt):int(end / dt)]*-1) + 1, linestyles="dotted", colors="black")
        plt.hlines(y=np.mean(platform_surge[int(start / dt):int(end / dt)]*-1), xmin=time_SIMO_Full[int(start / dt)], xmax=time_SIMO_Full[int(end / dt)], colors="red",
                   linestyles="dashed")
plt.title("Surge motion")
plt.xlabel("")
plt.legend(loc="best")
plt.savefig("Results\\surge.png")
plt.show()

plt.plot(time_SIMO_Full, platform_roll, "k", linewidth=0.8, label="Roll motion")
for i, (start, end) in enumerate(zip(starts_Rotor, ends_Rotor)):
    if i ==0:
        plt.vlines(x=time_SIMO_Full[int(start / dt)], ymin=np.mean(platform_roll[int(start / dt):int(end / dt)]) - 0.5, ymax=np.mean(platform_roll[int(start / dt):int(end / dt)]) + 0.5, linestyles="dotted", colors="green", label="Start steady state")
        plt.vlines(x=time_SIMO_Full[int(end / dt)], ymin=np.mean(platform_roll[int(start / dt):int(end / dt)]) - 0.5, ymax=np.mean(platform_roll[int(start / dt):int(end / dt)]*-1) + 0.5, linestyles="dotted", colors="black", label="End steady state")
        plt.hlines(y=np.mean(platform_roll[int(start / dt):int(end / dt)]), xmin=time_SIMO_Full[int(start / dt)], xmax=time_SIMO_Full[int(end / dt)], colors="red",
                   linestyles="dashed", label="Mean")
    else:
        plt.vlines(x=time_SIMO_Full[int(start / dt)], ymin=np.mean(platform_roll[int(start / dt):int(end / dt)]) - 0.5, ymax=np.mean(platform_roll[int(start / dt):int(end / dt)]) + 0.5, linestyles="dotted", colors="green")
        plt.vlines(x=time_SIMO_Full[int(end / dt)], ymin=np.mean(platform_roll[int(start / dt):int(end / dt)]) - 0.5, ymax=np.mean(platform_roll[int(start / dt):int(end / dt)]) + 0.5, linestyles="dotted", colors="black")
        plt.hlines(y=np.mean(platform_roll[int(start / dt):int(end / dt)]), xmin=time_SIMO_Full[int(start / dt)], xmax=time_SIMO_Full[int(end / dt)], colors="red",
                   linestyles="dashed")
plt.title("Roll motion")
plt.legend(loc="best")
plt.savefig("Results\\roll.png")
plt.show()

plt.plot(time_SIMO_Full, platform_pitch*-1, "k", linewidth=0.8, label="Pitch motion")
for i, (start, end) in enumerate(zip(starts_Rotor, ends_Rotor)):
    if i ==0:
        plt.vlines(x=time_SIMO_Full[int(start / dt)], ymin=np.mean(platform_pitch[int(start / dt):int(end / dt)]*-1) - 1, ymax=np.mean(platform_pitch[int(start / dt):int(end / dt)]*-1) + 1, linestyles="dotted", colors="green", label="Start steady state")
        plt.vlines(x=time_SIMO_Full[int(end / dt)], ymin=np.mean(platform_pitch[int(start / dt):int(end / dt)]*-1) - 1, ymax=np.mean(platform_pitch[int(start / dt):int(end / dt)]*-1) + 1, linestyles="dotted", colors="black", label="End steady state")
        plt.hlines(y=np.mean(platform_pitch[int(start / dt):int(end / dt)]*-1), xmin=time_SIMO_Full[int(start / dt)], xmax=time_SIMO_Full[int(end / dt)], colors="red",
                   linestyles="dashed", label="Mean")
    else:
        plt.vlines(x=time_SIMO_Full[int(start / dt)], ymin=np.mean(platform_pitch[int(start / dt):int(end / dt)]*-1) - 1, ymax=np.mean(platform_pitch[int(start / dt):int(end / dt)]*-1) + 1, linestyles="dotted", colors="green")
        plt.vlines(x=time_SIMO_Full[int(end / dt)], ymin=np.mean(platform_pitch[int(start / dt):int(end / dt)]*-1) - 1, ymax=np.mean(platform_pitch[int(start / dt):int(end / dt)]*-1) + 1, linestyles="dotted", colors="black")
        plt.hlines(y=np.mean(platform_pitch[int(start / dt):int(end / dt)]*-1), xmin=time_SIMO_Full[int(start / dt)], xmax=time_SIMO_Full[int(end / dt)], colors="red",
                   linestyles="dashed")
plt.title("Pitch motion")
plt.legend(loc="best")
plt.savefig("Results\\pitch.png")
plt.show()

surge_mean = np.array([np.mean(platform_surge[int(start / dt):int(end / dt)]) for start, end in zip(starts_Rotor, ends_Rotor)])
roll_mean = np.array([np.mean(platform_roll[int(start / dt):int(end / dt)]) for start, end in zip(starts_Rotor, ends_Rotor)])
pitch_mean = np.array([np.mean(platform_pitch[int(start / dt):int(end / dt)]) for start, end in zip(starts_Rotor, ends_Rotor)])

plt.plot(wind_speed_mean_rotor, surge_mean*-1, 'k-o')
plt.xlabel("Wind speed [m/s]")
plt.ylabel("Displacement [m]")
plt.legend(loc="best")
plt.title("Mean surge offset vs. wind speed")
plt.tight_layout()
plt.savefig("Results\\surge_vs_velocity.png")
plt.show()

plt.plot(wind_speed_mean_rotor, roll_mean, 'k-o')
plt.xlabel("Wind speed [m/s]")
plt.ylabel("Angular displacement [deg]")
plt.legend(loc="best")
plt.title("Mean roll offset vs. wind speed")
plt.tight_layout()
plt.savefig("Results\\roll_vs_velocity.png")
plt.show()

plt.plot(wind_speed_mean_rotor, pitch_mean*-1, 'k-o')
plt.xlabel("Wind speed [m/s]")
plt.ylabel("Angular displacement [deg]")
plt.legend(loc="best")
plt.title("Mean pitch offset vs. wind speed")
plt.tight_layout()
plt.savefig("Results\\pitch_vs_velocity.png")
plt.show()

debug=True
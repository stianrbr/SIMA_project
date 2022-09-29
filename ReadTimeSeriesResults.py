# -*- coding: utf-8 -*-
"""
ReadTimeDomainResults
"""

import struct
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import os
from statistics import mean
import pandas as pd
import openpyxl




plt.style.use("seaborn")
plt.rcParams["figure.figsize"] = (15, 4)

# ============================================================================#
# Inputs
SIMAfol = "C:\\DNV\\Workspaces\\SIMA\\Project work 3\\RotorOnly12MW\\Initial1\\1231-20220927115332\\"

# Total simulation time and storage time step
NtRiflexForce = 153000  # number of stored time steps on the force file (=simulationTime/dt)
hullBody = 1  # this is the number of the hull body (order in the SIMA inputs)


# ============================================================================#
# ============================================================================#

def unpackBinaryResults(fileName, Nt, Nchan):
    # Read .bin file
    # forfilebin is the name of the binary file
    # Nt is the number of time steps (if 0, then Nchan is needed)
    # Nchan is an optional argument - unused if Nt>0

    with open(fileName, mode='rb') as file:
        fileContent = file.read()

    # Reshape results in file to a 2D array 'A', with one result component per row and one time step per column
    numRecords = int((len(fileContent) / 4))
    unpacked = struct.unpack("f" * numRecords, fileContent)
    if Nt > 1:
        Nchan = len(np.asarray(unpacked)) / Nt
    else:
        Nt = len(np.asarray(unpacked)) / Nchan
    # print(Nchan,Nt,numRecords)
    A = np.reshape(np.asarray(unpacked), (int(Nchan), int(Nt)), order='F')

    return A


def readSIMO_resultstext(filename):
    # read the SIMO text

    chanNames = [];
    nchan = 0;
    nts = 0;
    dt = 0;
    with open(filename, 'r') as f:

        # read the header

        for ii in range(0, 6):
            tline = f.readline()
        # number of samples
        tline = f.readline()
        regexp = r"(\d+)"
        d = np.fromregex(StringIO(tline), regexp, [('num', np.int64)])
        nts = int(d['num'])  # number of time steps
        tline2 = f.readline()
        tsplit = tline2.split(' ')
        dt = float(tsplit[8])  # time step
        tline = (f.readline()).split(' ')
        tstart = float(tline[7])
        tline = (f.readline()).split(' ')
        tend = tline[9]

        for tline in f.readlines():
            if tline[0] != '*':
                nchan = nchan + 1
                chanNames.append(tline.split())

    return nchan, nts, dt, chanNames


def getchannelNumbers(chanNames, B1):
    chanMotions = 0
    ind1 = 0
    chanWave = 0
    chanAcc = 0
    nameMotion = 'B%dr29c1' % B1

    for ii in range(0, len(chanNames)):
        xstr = chanNames[ii][0]
        x = xstr.find(nameMotion)
        if x > -1:
            ind1 = ii
            chanMotions = ind1 + np.arange(0, 6)

    ind1 = 0
    nameWave = 'Totalwaveelevation';
    for ii in range(0, len(chanNames)):
        xstr = chanNames[ii][0]
        x = xstr.find(nameWave)
        if x > -1:
            ind1 = ii
            chanWave = ind1

    ind1 = 0
    nameAcc = 'B%dr30c1' % B1
    for ii in range(0, len(chanNames)):
        xstr = chanNames[ii][0]
        x = xstr.find(nameAcc)
        if x > -1:
            ind1 = ii
            chanAcc = ind1

    return chanMotions, chanWave, chanAcc


def unpackSIMOresults(fileName, nts):
    with open(fileName, mode='rb') as file:
        fileContent = file.read()
    numRecords = int((len(fileContent) / 4))
    cols = len(np.asarray(struct.unpack("f" * numRecords, fileContent))) / nts
    A = np.transpose(np.reshape(np.asarray(struct.unpack("f" * numRecords, fileContent)), (nts, int(cols)), order='F'))

    return A

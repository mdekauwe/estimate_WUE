#!/usr/bin/env python

"""
Estimate ecosystem WUE

That's all folks.
"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (17.08.2018)"
__email__ = "mdekauwe@gmail.com"

import os
import sys
import glob
import multiprocessing as mp
import numpy as np
import xarray as xr
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd

class EstimateWUE(object):

    def __init__(self, flux_dir=None, output_dir=None, met_dir=None,
                 flux_subset=None, data_source=None, mpi=None, num_cores=None):

        self.flux_dir = flux_dir
        self.output_dir = output_dir
        self.met_dir = met_dir
        self.flux_subset = flux_subset
        self.data_source = data_source
        self.num_cores = num_cores

        # W/m2 = 1000 (kg/m3) * 2.45 (MJ/kg) * 10^6 (J/kg) * 1 mm/day * \
        #        (1/86400) (day/s) * (1/1000) (mm/m)
        # 2.45 * 1E6 W/m2 = kg/m2/s or mm/s
        self.WM2_TO_KG_M2_S = 1.0 / ( 2.45 * 1E6 )
        self.KG_TO_G = 1000.0
        self.MOL_TO_MMOL = 1000.0
        self.G_TO_MOL_H20 = 1.0 / 18.0
        self.HPA_TO_KPA = 0.1
        self.KPA_TO_PA = 1000.0
        self.SEC_TO_HR = 3600.0
        self.SEC_TO_HLFHR = 1800.0
        self.UMOL_TO_MOL = 0.000001
        self.MOL_C_TO_GRAMS_C = 12.

        self.out_cols = ["Site","Year","WUE"]

    def main(self):

        (flux_files, met_files) = self.initialise_stuff()

        # Setup multi-processor jobs
        if self.num_cores is None: # use them all!
            self.num_cores = mp.cpu_count()
        chunk_size = int(np.ceil(len(flux_files) / float(self.num_cores)))
        pool = mp.Pool(processes=self.num_cores)
        queue = mp.Queue() # define an output queue

        processes = []
        for i in range(self.num_cores):
            start = chunk_size * i
            end = chunk_size * (i + 1)
            if end > len(flux_files):
                end = len(flux_files)

            # setup a list of processes that we want to run
            p = mp.Process(target=self.worker,
                           args=(queue, flux_files[start:end],
                                 met_files[start:end], ))
            processes.append(p)

        # Run processes
        for p in processes:
            p.start()

        # OS pipes are not infinitely long, so the queuing process can get
        # blocked when using the put function - a "deadlock". The following
        # code gets around this issue

        # Get process results from the output queue
        results = []
        while True:
            running = any(p.is_alive() for p in processes)
            while not queue.empty():
                results.append(queue.get())
            if not running:
                break

        # Exit the completed processes
        for p in processes:
            p.join()

        return results

    def worker(self, output, flux_files, met_files):

        df_out = pd.DataFrame(columns=self.out_cols)

        for flux_fname, met_fname in zip(flux_files, met_files):
            site = os.path.basename(flux_fname).split(".")[0][0:6]
            years = os.path.basename(flux_fname).split(".")[0][7:16]
            source = os.path.basename(flux_fname).split(".")[0][17:].\
                                                  split("_")[0]

            print(site)

            (df_flx, df_met) = self.get_data(flux_fname, met_fname)
            (df_flx, df_met) = self.clean_vars(df_flx, df_met, source)

            df_s = df_flx.resample("D").sum()
            df_m = df_met.resample("D").mean()
            df = pd.concat([df_s, df_m], axis=1)
            wue = np.where(df.ET < 0.1, np.nan, df.GPP / df.ET * df.VPD)
            df["wue"] = wue
            df = df.resample("A").mean()
            df = df[~np.isnan(df.wue)]

            for i in range(len(df)):
                df_out = self.write_row(site, df, i, df_out)

        output.put(df_out)

    def write_row(self, site, df, i, df_out):

        row = pd.Series([site, df.index.year[i], df.wue[i]],
                        index=self.out_cols)
        result = df_out.append(row, ignore_index=True)

        return result

    def initialise_stuff(self):

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Run all the met files in the directory?
        if len(self.flux_subset) == 0 and self.data_source is None:

            flux_files = glob.glob(os.path.join(self.flux_dir, "*.nc"))
            met_files = glob.glob(os.path.join(self.met_dir, "*.nc"))

        # just a sub-section by theme, i.e. FLUXNET2015
        elif len(self.flux_subset) == 0 and self.data_source is not None:
            all_flux_files = glob.glob(os.path.join(self.flux_dir, "*.nc"))
            all_met_files = glob.glob(os.path.join(self.met_dir, "*.nc"))
            flux_files = []
            met_files = []
            for src in self.data_source:
                for i,j in zip(all_flux_files, all_met_files):
                    if src in i:
                        flux_files.append(i)
                        met_files.append(j)

        # A single file, or a list of files ...
        else:
            flux_files = [os.path.join(self.flux_dir, i) \
                            for i in self.flux_subset]
            met_files = []
            for i in self.flux_subset:
                source = os.path.basename(i).\
                            split(".")[0][17:].split("_")[0]

                if source == "FLUXNET2015" or source == "LaThuile":
                    new = i.replace("Flux", "Met")
                    met_files.append(os.path.join(self.met_dir, new))
                elif source == "OzFlux" :
                    new = i.replace("Flux", "Met", 2).\
                            replace("Met", "Flux", 1)
                    met_files.append(os.path.join(self.met_dir, new))

        return (flux_files, met_files)

    def get_data(self, flux_fname, met_fname):

        ds = xr.open_dataset(flux_fname)
        df_flx = ds.squeeze(dim=["x","y"], drop=True).to_dataframe()
        df_flx = df_flx.reset_index()
        df_flx = df_flx.set_index('time')

        ds = xr.open_dataset(met_fname)
        df_met = ds.squeeze(dim=["x","y"], drop=True).to_dataframe()
        df_met = df_met.reset_index()
        df_met = df_met.set_index('time')

        # only keep "daylight" hours
        df_flx = df_flx.between_time("06:00", "20:00")
        df_met = df_met.between_time("06:00", "20:00")

        return (df_flx, df_met)

    def clean_vars(self, df_flx, df_met, source):

        # Screen for measured and good gap-filled data
        if source == "OzFlux":
            # Measured = 0 ; 10 = instrument calibration correction,  good data
            df_met.where(np.logical_or(df_flx.Qle_qc == 0, df_flx.Qle_qc == 10),
                         inplace=True)
            df_met.where(np.logical_or(df_flx.GPP_qc == 0, df_flx.Qle_qc == 10),
                         inplace=True)

            # Measured = 0 ; 10 = instrument calibration correction,  good data
            df_flx.where(np.logical_or(df_flx.Qle_qc == 0, df_flx.Qle_qc == 10),
                         inplace=True)
            df_flx.where(np.logical_or(df_flx.GPP_qc == 0, df_flx.Qle_qc == 10),
                         inplace=True)

        elif source == "FLUXNET2015":
            # Measured = 0 ; good-quality gap-fill = 1
            df_met.where(np.logical_or(df_flx.Qle_qc == 0, df_flx.Qle_qc == 1),
                         inplace=True)

            # No QC for GPP, use NEE and RECO
            df_met.where(np.logical_or(df_flx.NEE_qc == 0, df_flx.NEE_qc == 1),
                         inplace=True)

            # Measured = 0 ; good-quality gap-fill = 1
            df_flx.where(np.logical_or(df_flx.Qle_qc == 0, df_flx.Qle_qc == 1),
                         inplace=True)

            # No QC for GPP, use NEE and RECO
            df_flx.where(np.logical_or(df_flx.NEE_qc == 0, df_flx.NEE_qc == 1),
                         inplace=True)

        elif source == "LaThuile":
            # Measured = 0 ; good-quality gap-fill = 1
            df_met.where(np.logical_or(df_flx.Qle_qc == 0, df_flx.Qle_qc == 1),
                         inplace=True)

            # Measured and good-quality gap fill
            df_met.where(df_flx.NEE_GPP_qcOK == 1, inplace=True)

            # Measured = 0 ; good-quality gap-fill = 1
            df_flx.where(np.logical_or(df_flx.Qle_qc == 0, df_flx.Qle_qc == 1),
                         inplace=True)

            # Measured and good-quality gap fill
            df_flx.where(df_flx.NEE_GPP_qcOK == 1, inplace=True)

        # Mask by met data.
        # Measured = 0 ; good-quality gap-fill = 1
        df_flx.where(np.logical_or(df_met.Rainf_qc == 0, df_met.Rainf_qc == 1),
                     inplace=True)
        df_flx.where(np.logical_or(df_met.VPD_qc == 0, df_met.VPD_qc == 1),
                     inplace=True)
        df_met.where(np.logical_or(df_met.Rainf_qc == 0, df_met.Rainf_qc == 1),
                     inplace=True)
        df_met.where(np.logical_or(df_met.VPD_qc == 0, df_met.VPD_qc == 1),
                     inplace=True)

        # Mask dew
        df_met.where(df_flx.Qle > 0., inplace=True)
        df_flx.where(df_flx.Qle > 0., inplace=True)

        # Convert units ...
        df_flx["ET"] = df_flx['Qle'] * self.WM2_TO_KG_M2_S
        diff = df_met.index.minute[1] - df_met.index.minute[0]
        if diff == 0:
            # hour gap i.e. Tumba
            df_met['Rainf'] = df_met.Rainf * self.SEC_TO_HR
            df_flx["GPP"] *= self.MOL_C_TO_GRAMS_C * self.UMOL_TO_MOL * \
                             self.SEC_TO_HR
            df_flx["ET"] *= self.SEC_TO_HLFHR
        else:
            # 30 min gap
            df_met['Rainf'] *= self.SEC_TO_HLFHR
            df_flx["GPP"] *= self.MOL_C_TO_GRAMS_C * self.UMOL_TO_MOL * \
                             self.SEC_TO_HLFHR
            df_flx["ET"] *= self.SEC_TO_HLFHR

        # kPa
        df_met['VPD'] *= self.HPA_TO_KPA

        # Drop the stuff we don't need
        df_flx = df_flx[['GPP','ET']]
        df_met = df_met[['Rainf','VPD']]

        df_met = df_met.reset_index()
        df_met = df_met.set_index('time')
        df_flx = df_flx.reset_index()
        df_flx = df_flx.set_index('time')

        (months, missing_gpp) = self.get_three_most_productive_months(df_flx)

        # filter three most productive months
        df_flx = df_flx[(df_flx.index.month == months[0]) |
                        (df_flx.index.month == months[1]) |
                        (df_flx.index.month == months[2])]
        df_met = df_met[(df_met.index.month == months[0]) |
                        (df_met.index.month == months[1]) |
                        (df_met.index.month == months[2])]

        # remove days after rain days...
        (df_flx, df_met) = self.filter_for_rain(df_flx, df_met, diff)

        return df_flx, df_met

    def filter_for_rain(self, df_flx, df_met, diff):
        idx = df_met[df_met.Rainf > 0.0].index.tolist()
        bad_dates = []
        for rain_idx in idx:
            bad_dates.append(rain_idx)
            if diff == 0:
                for i in range(24):
                    new_idx = rain_idx + dt.timedelta(minutes=60)
                    bad_dates.append(new_idx)
                    rain_idx = new_idx
            else:
                for i in range(48):
                    new_idx = rain_idx + dt.timedelta(minutes=30)
                    bad_dates.append(new_idx)
                    rain_idx = new_idx
        # There will be duplicate dates most likely so remove these.
        bad_dates = np.unique(bad_dates)

        # remove rain days...
        df_flx = df_flx[~df_flx.index.isin(bad_dates)]
        df_met = df_met[~df_met.index.isin(bad_dates)]

        return (df_flx, df_met)

    def get_three_most_productive_months(self, df):
        # filter three most productive months
        df_m = df.resample("M").mean()
        missing_gpp = False

        try:
            df_m = df_m.sort_values("GPP", ascending=False)[:3]
            months = df_m.index.month
        except KeyError:
            missing_gpp = True
            months = None

        return (months, missing_gpp)

if __name__ == "__main__":

    # ------------------------------------------- #
    flux_dir = "data/Flux/"
    met_dir = "data/Met/"
    output_dir = "outputs"
    num_cores = None
    ofname = os.path.join(output_dir, "WUE.csv")
    # if empty...run all the files in the met_dir
    flux_subset = ['US-Ha1_1991-2012_FLUXNET2015_Flux.nc',\
                   'FR-Pue_2000-2014_FLUXNET2015_Flux.nc'] #[]
    data_source = ["FLUXNET2015"]
    # ------------------------------------------- #

    W = EstimateWUE(flux_dir=flux_dir, output_dir=output_dir, met_dir=met_dir,
                    flux_subset=flux_subset, data_source=data_source,
                    num_cores=num_cores)
    results = W.main()

    # merge all the processor DF fits into one big dataframe
    df = pd.concat(results, ignore_index=True)
    if os.path.exists(ofname):
            os.remove(ofname)
    df.to_csv(ofname, index=False)

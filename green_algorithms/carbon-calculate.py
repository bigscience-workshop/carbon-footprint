import os
import json
from functools import lru_cache

import pandas as pd

from app import aggregate_input_values


def get_gpu(row):
    for gpu in ["V100_32GB", "V100_16GB", "A100_40GB"]:
        if row[gpu] > 0:
            return gpu


def parse_duration(duration):
    # ignore hyphen
    if "-" in duration:
        _, duration = duration.split("-")
    hours, minutes, _ = list(map(int, duration.split(":")))
    return hours, minutes


@lru_cache(maxsize=3)
def get_tdp(gpu):
    if gpu is None:
        return 200
    if "V100" in gpu:
        return 300
    if "A100" in gpu:
        return 250
    raise ValueError


def get_gpu_df():
    gpu_df = pd.read_csv("jz-logs_09.02.csv")
    gpu_df["V100_32GB"] = pd.to_numeric(gpu_df["V100_32GB"], downcast="float")
    gpu_df["V100_16GB"] = pd.to_numeric(gpu_df["V100_16GB"], downcast="float")
    gpu_df["A100_40GB"] = pd.to_numeric(gpu_df["A100_40GB"], downcast="float")
    gpu_df["CPU"].replace(("N/A", 0), inplace=True)
    gpu_df["CPU"] = pd.to_numeric(gpu_df["CPU"], downcast="float")
    gpu_df["RAM"].replace("M", "", regex=True, inplace=True)
    gpu_df["RAM"].replace("G", "000", regex=True, inplace=True)
    gpu_df["RAM"].replace("T", "000000", regex=True, inplace=True)
    gpu_df["RAM"] = pd.to_numeric(gpu_df["RAM"], downcast="float")
    gpu_df["RAM"] = gpu_df["RAM"] / 1000
    return gpu_df


def get_cores_dict():
    data_dir = os.path.join(os.path.abspath(""), "data")
    cpu_data = pd.read_csv(os.path.join(data_dir, "TDP_cpu.csv"), sep=",", skiprows=1)
    cpu_data.drop(["source"], axis=1, inplace=True)

    ### GPU ###
    gpu_data = pd.read_csv(os.path.join(data_dir, "TDP_gpu.csv"), sep=",", skiprows=1)
    gpu_data.drop(["source"], axis=1, inplace=True)

    # Dict of dict with all the possible models
    # e.g. {'CPU': {'Intel(R) Xeon(R) Gold 6142': 150, 'Core i7-10700K': 125, ...
    cores_dict = dict()
    cores_dict["CPU"] = pd.Series(
        cpu_data.TDP_per_core.values, index=cpu_data.model
    ).to_dict()
    cores_dict["GPU"] = pd.Series(
        gpu_data.TDP_per_core.values, index=gpu_data.model
    ).to_dict()
    return cores_dict


def build_carbon_df(gpu_df, cores_dict):
    outputs = []
    for _, row in gpu_df.iterrows():
        gpu = get_gpu(row)
        tdp_gpu = get_tdp(gpu)
        hours, minutes = parse_duration(row["duration"])
        output = aggregate_input_values(
            coreType="Both" if gpu is not None else "CPU",
            n_CPUcores=row["CPU"]/2,
            CPUmodel="Xeon Gold 6248",
            tdpCPU=cores_dict["CPU"]["Xeon Gold 6248"],
            n_GPUs=int(sum(row[["V100_32GB", "V100_16GB", "A100_40GB"]])),
            GPUmodel=gpu,
            tdpGPU=tdp_gpu,
            memory=row["RAM"],
            runTime_hours=hours,
            runTime_min=minutes,
            locationContinent="Europe",
            locationCountry="France",
            PUE=1.2,
            location="FR",
            serverContinent=None,
            server=None,
            tdpCPUstyle={"display": "none"},
            serverStyle={"display": "none"},
            providerStyle={"display": "none"},
            tdpGPUstyle={"display": "true"},
            locationStyle={"display": "true"},
            usageCPUradio=None,
            # assume max usage
            usageCPU=1,
            usageGPU=1,
            usageGPUradio=None,
            PUEradio="Yes",
            PSFradio=None,
            PSF=1,
            selected_platform="localServer",
            selected_provider=None,
            existing_state=None,
        )
        # for some reason the output is a string
        output = json.loads(output)
        # for some reason the output is nested in these keys
        output = output["response"]["props"]["data"]
        # add job id
        output["jobid"] = row.jobid
        output["jobname"] = row.jobname
        output["start_time"] = row.start_time
        output['end_time'] = row.end_time
        output['user'] = row.user
        outputs.append(output)

    carbon_df = pd.DataFrame(outputs)
    return carbon_df


def main():
    gpu_df = get_gpu_df()
    cores_dict = get_cores_dict()
    carbon_df = build_carbon_df(gpu_df, cores_dict)
    carbon_df.to_csv("carbon.csv")


if __name__ == "__main__":
    main()

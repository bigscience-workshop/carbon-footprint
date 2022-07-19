#!/usr/bin/env python
"""Retrieve user runs on JZ and print metrics in human or machine readable format"""

# Orginal script
from doctest import FAIL_FAST
import sys
import subprocess
import collections
import re
from enum import Enum, auto

class GpuType(Enum):
    V100_32GB = auto()
    V100_16GB = auto()
    A100_40GB = auto()
    A100_80GB = auto()

class Node:
    def __init__(self, gpu_type, num_gpus):
        self.gpu_type = gpu_type
        self.num_gpus = num_gpus

type2partitions = [(Node(GpuType.V100_32GB, 4), 'gpu_p1'),
                   (Node(GpuType.V100_32GB, 8), 'gpu_p2'),
                   (Node(GpuType.V100_16GB, 4), 'gpu_p3'),
                   (Node(GpuType.A100_40GB, 8), 'gpu_p4'),
                   (Node(GpuType.A100_80GB, 8), 'gpu_p5')]

node2type = {}
for node_type, partition in type2partitions:
    p = subprocess.run(f'/gpfslocalsys/slurm/current/bin/sinfo -N -p {partition} --Format=nodehost -h',
                       shell=True, encoding='utf8',
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for node in p.stdout.splitlines():
        node2type[node.strip()] = node_type


def find_node_type(node):
    try:
        return node2type[node]
    except:
        return None


def find_num_gpus_per_types(nodelist, alloctres):
    p = subprocess.run(f'/gpfslocalsys/slurm/current/bin/scontrol show hostnames {nodelist}',
                       shell=True, encoding='utf8',
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    nodes = p.stdout.splitlines()
    is_single_node = (len(nodes) == 1)

    if is_single_node:
        m = re.search('gpu=(\d+)', alloctres)
        num_gpus = int(m.group(1)) if m else 0

    num_gpus_per_types = collections.defaultdict(int)
    for node in nodes:
        node_type = find_node_type(node)
        if node_type:
            num_gpus_per_types[node_type.gpu_type] += num_gpus if is_single_node else node_type.num_gpus
    return num_gpus_per_types


# Modifications to show more info for each job
def split_alloctres(alloctres):
    line_splitted = [i.split('=') for i in alloctres.split(',')]
    d = {}
    for pair in line_splitted:
        if len(pair) < 2:
                    continue
        key = pair[0]
        value = pair[1]
        d[key] = (value)
    return d


show_headers = not ('-n' in sys.argv[1:] or '--noheader' in sys.argv[1:])
args = ['sacct'] + sys.argv[1:] + ['--format=jobid,elapsed,nodelist,alloctres,partition,qos,start,end,group,jobname,workdir', '-P', '-X', '-n']

p = subprocess.run(' '.join(args), shell=True, encoding='utf8',
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)

if show_headers:
    # Print human-readable output
    fmt_string = '{0:<9} {1:<9} {2:<9} {3:<9} {4:<9} {5:<9} {6:<9} {7:<9} {8:<9} {9:<9} {10:<9} {11:<10} {12:<10} {13:<19} {14:<19} {15:<19}'
    print(fmt_string.format('JobID', 'V100 32GB', 'V100 16GB', 'A100 40GB', 'A100 80GB', 'CPUs', 'RAM', 'Energy', 'Partition', 'Group', 'Elapsed', 'QoS', 'JobName', 'Start', 'End', 'Workdir'))
    print(('-' * 9 + ' ') * 11 + ('-' * 10 + ' ') * 2 + ('-' * 19 + ' ') * 2 + ('-' * 40 + ' '))

else:
    # Machine readable output
    fmt_string = '{0}|{1}|{2}|{3}|{4}|{5}|{6}|{7}|{8}|{9}|{10}|{11}|{12}|{13}|{14}|{15}'

for j in p.stdout.splitlines():
    job_id, elapsed, nodelist, alloctres, partition, qos, start, end, group, jobname, workdir = j.split(
        '|')

    num_gpus_per_types = find_num_gpus_per_types(nodelist, alloctres)

    alloc = split_alloctres(alloctres)

    for field in ['cpu', 'mem', 'energy']:
        if field not in alloc:
            alloc[field] = 'N/A'

    print(
        fmt_string.format(job_id,
                          num_gpus_per_types[GpuType.V100_32GB],
                          num_gpus_per_types[GpuType.V100_16GB],
                          num_gpus_per_types[GpuType.A100_40GB],
                          num_gpus_per_types[GpuType.A100_80GB],
                          alloc['cpu'],
                          alloc['mem'], 
                          alloc['energy'],
                          partition,
                          group,
                          elapsed,
                          qos,
                          jobname,
                          start,
                          end,
                          workdir))

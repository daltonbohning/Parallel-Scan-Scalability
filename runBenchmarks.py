#!/usr/bin/env python3
#
# runBenchmarks.py
#


import sys
import os
from os.path import dirname
from os.path import join
from statistics import median
import subprocess


rootDir = dirname(os.path.realpath(__file__))

numIterationsPerTest = 3 #The median values are taken


# Execute from the rootDir, piping output
def execute(*a):
    if len(a) == 1 and isinstance(a[0], str):
        #Single string argument, execute as shell=True
        args_ = a[0]
    else:
        #List of arguments, execute as shell=False
        args_ = []
        for i in a:
            if isinstance(i, str): args_.append(i)
            elif isinstance(i, list): args_ += i
            else: print("invalid argument: ", file=stderr)

    return subprocess.run(args=args_,
                          cwd=rootDir,
                          stdout=subprocess.PIPE,
                          shell=isinstance(args_, str))



def run():
    outputFile = open(join(rootDir, "benchmarks.csv"), "w+")
    outputFile.write("implementation,arraySize,sectionSize,totalTime(ms)\n")

    for target in ["brent_release"]:
        executable = "./brent-kung" if target == "brent_release" else "./openmp_inclusiveScan"
        for arraySize in [1024,2048,4096,8192]:
            for sectionSize in [1024,2048]:
                args_ = [target,"ARRAY_SIZE="+str(arraySize),"ARRAY_SIZE="+str(arraySize)]
                print("Making " + str(target))
                execute("make", target)

                #We capture these metrics
                totalTime = [0] * numIterationsPerTest

                for iteration in range(numIterationsPerTest):
                    proc = execute(executable)
                    for l in proc.stdout.decode().split("\n"):
                        if "Kernel Total (ms):" in l:
                            totalTime[iteration] = float(l.split(":")[1].strip())
                ###end for iteration
                
                outputFile.write(target + "," + str(arraySize) + "," + str(sectionSize) + "," + str(median(totalTime)) + "\n")
                
            ###end for sectionSize
        ###end for arraySize
    ###end for target

    outputFile.close()
###end run()


run()

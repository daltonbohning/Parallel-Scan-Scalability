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
    outputFile.write("implementation,arraySize,sectionSize,executionTime(ms),memoryTime(ms),totalTime(ms)\n")

    for target in ["brent_release"]:
        executable = "./brent-kung" if target == "brent_release" else "./openmp_inclusiveScan"
        for arraySize in list(map(lambda x: 2**x, range(8,29))): #from 2^8 to 2^28, by 2's
            for sectionSize in [1024,2048]:
                makeTarget = "make " + target + " ARRAY_SIZE="+str(arraySize) +" SECTION_SIZE="+str(sectionSize)
                print("Running \"" + makeTarget + "\": ", end="", flush=True)
                execute(makeTarget)
                
                #We capture these metrics
                execTime = [0] * numIterationsPerTest
                memTime = [0] * numIterationsPerTest
                totalTime = [0] * numIterationsPerTest

                for iteration in range(numIterationsPerTest):
                    print(".", end="", flush=True)
                    proc = execute(executable)
                    for l in proc.stdout.decode().split("\n"):
                        if "Kernel Execution (ms):" in l:
                            execTime[iteration] = float(l.split(":")[1].strip())
                        if "Kernel Memory (ms):" in l:
                            memTime[iteration] = float(l.split(":")[1].strip())
                        if "Kernel Total (ms):" in l:
                            totalTime[iteration] = float(l.split(":")[1].strip())
                ###end for iteration
                print("Done")
                
                outputFile.write(target + "," + str(arraySize) + "," + str(sectionSize) + "," + 
                                 str(median(execTime)) + "," +
                                 str(median(memTime)) + "," +
                                 str(median(totalTime)) +"\n")
                
            ###end for sectionSize
        ###end for arraySize
    ###end for target

    outputFile.close()
###end run()


run()

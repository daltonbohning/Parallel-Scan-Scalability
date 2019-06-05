#!/usr/bin/env python3
#
# runTests.py
#


import sys
import os
from os.path import dirname
from os.path import join
from statistics import median
import subprocess


rootDir = dirname(os.path.realpath(__file__))


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
    numFailures = 0
    totalTests = 0
    for target in ["brent_test"]:
        executable = "./brent-kung" if target == "brent_test" else "./openmp_inclusiveScan"
        for arraySize in list(map(lambda x: 2**x, range(8,21))): #from 2^8 to 2^20, by 2's
            for sectionSize in [1024,2048]:
                totalTests += 1

                makeTarget = "make " + target + " ARRAY_SIZE="+str(arraySize) +" SECTION_SIZE="+str(sectionSize)
                print("Running \"" + makeTarget + "\": ", end="", flush=True)
                execute(makeTarget)

                proc = execute(executable)
                for l in proc.stdout.decode().split("\n"):
                    if "ALL CORRECT!" in l:
                        print("OK!")
                    elif "FAIL!" in l:
                        numFailures += 1
                        print("FAIL!")
                        
            ###end for sectionSize
        ###end for arraySize
    ###end for target
    
    if numFailures == 0:
        print("Passed " + str(totalTests) + "/" + str(totalTests))
    else:
        print("Failed " + str(numFailures) + "/" + str(totalTests))
###end run()


run()


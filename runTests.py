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


#By default, run all tests. Individual versions can be specified
targetsToRun = ["brent_test", "openmp_test"]
if len(sys.argv) > 1:
    #Allow some shorthands, or the full names as above
    target = {
        "cuda": "brent_test",
        "brent": "brent_test",
        "omp": "openmp_test",
        "openmp": "openmp_test"
    }.get(sys.argv[1], sys.argv[1])
    if target not in targetsToRun:
        print("Invalid target: " + target)
        exit()
    targetsToRun = [target]


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
    numCorrect = 0
    numFailures = 0
    totalTests = 0

    print("Running " +  str(targetsToRun))

    for target in targetsToRun:
        executable = {
            "brent_test": "./brent-kung",
            "openmp_test": "./openmp_inclusiveScan"
        }.get(target)

        
        sectionSizes = {
            "brent_test": [1024,2048],
            "openmp_test": [2,4,8,16]
        }.get(target)


        for arraySize in list(map(lambda x: 2**x, range(8,29))): #from 2^8 to 2^28, by 2's
            for sectionSize in sectionSizes:
                totalTests += 1

                makeTarget = "make " + target + " ARRAY_SIZE="+str(arraySize) +" SECTION_SIZE="+str(sectionSize) + " NUM_THREADS="+str(sectionSize)
                print("Running \"" + makeTarget + "\": ", end="", flush=True)
                execute(makeTarget)

                proc = execute(executable)
                for l in proc.stdout.decode().split("\n"):
                    if "ALL CORRECT!" in l:
                        numCorrect += 1
                        print("OK!")
                    elif "FAIL!" in l:
                        numFailures += 1
                        print("FAIL!")
                        
            ###end for sectionSize
        ###end for arraySize
    ###end for target

    if numCorrect > 0:
        print("Passed " + str(totalTests) + "/" + str(totalTests))
    if numFailures > 0:
        print("Failed " + str(numFailures) + "/" + str(totalTests))
    
    numUnknown = totalTests - (numFailures + numCorrect)
    if numUnknown > 0:
        print("ERROR " + str(numUnknown) + "/" + str(totalTests))

###end run()


run()


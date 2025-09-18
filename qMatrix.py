#!/usr/bin/python3
import time
tStart = time.time()
tStartTotal = tStart

from qFunction import qMatrix, qMatrixUsingTree, qMatrixUsingTreeFast, projectFeatures, loadTableFromCsv, DATA_AND_SET_SIZES
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import sys



def showHelp():
  print(f"{sys.argv[0]} [--help]")
  print("          [-o outputTable]")
  print("          [-tree] [-tree-fast] [-log]")
  print("          [-op outputPointList] [-oi outputImage] [-cool-down float]")
  print("          [-i columnName] inputTable")
  print("")
  print("---[ General ]-----------------------------------------------------")
  print(" --help             : shows the help of the program.")
  print("")
  print("---[ Q-matrix generation ]-----------------------------------------")
  print(" -o outputTable     : writes the Q-matrix as CSV-file. If the file")
  print("                      name is \"-\" then the data will be print to stdout.")
  print("")
  print(" -tree              : uses the tree approach instead computing the")
  print("                      Q-matrix brute force.")
  print("")
  print(" -tree-fast         : uses the fast version of the tree approach")
  print("                      instead computing the Q-matrix brute force.")
  print("")
  print(" -log               : prints debug information after computing the data.")
  print("")
  print("---[ dependency projection ]---------------------------------------")
  print(" -op outputPointList: writes the position of the projected feature")
  print("                      points to a CSV-file. If the file name is \"-\"")
  print("                      then the data will be print to stdout.")
  print("")
  print(" -oi outputImage    : writes the projected feature points to an image-file")
  print("                      (PNG or PDF). If the file name is \"-\" then the")
  print("                      image is shown in a window.")
  print("")
  print(" -cool-down c       : The coolDown value for the projection phase.")
  print("                      Default is 0.4. The value is expected to be")
  print("                      0 <= c < 1. Smaller values will tend more to be")
  print("                      a line and the probability that bijective dependent")
  print("                      features land at the same point increases.")
  print("                      Greater values tend more to be a cloud.")
  print("")
  print("---[ data input ]--------------------------------------------------")
  print(" -i columnName      : Ignores the given column during the computation.")
  print("                      This parameter can be given multiple times.")
  print("")
  print(" -numbered          : Use own data loader for faster table access. (default)")
  print(" -pandas            : Use pandas data loader for comparability.")
  print("")
  print(" inputTable         : A file name of a CSV-file. This first line is")
  print("                      expected to be column names.")


def cleanupName(name):
  return (f"{name}"
    .replace(",", ";")
    .replace("\r\n", " ")
    .replace("\n\r", " ")
    .replace("\r", " ")
    .replace("\n", " ")
    )

def writeMatrix(matrix, columns, f):
  print("," + ",".join(columns), file=f)
  for row, c in zip(matrix, columns):
    print(",".join([c] + [f"{v}" for v in row]), file=f)
  
def writePoints(matrix, columns, f):
  print("Column,X,Y", file=f)
  for row, c in zip(matrix, columns):
    print(",".join([c] + [f"{v}" for v in row]), file=f)
  

def checkForOldFlags(name):
  replacements = [
    ( '--coolDown', '-cool-down' )
    , ( '--tree', '-tree' )
    , ( '--tree-fast', '-tree-fast' )
    , ( '--log', '-log' )
    ]

  for (o, n) in replacements:
    if name == o:
      print(f"The parameter '{o}' is old and will be removed soon.")
      print(f"Please use '{n}' instead.")

def timeStep(tStart, title):
  now = time.time()
  print(f"{title}: {now - tStart}s")
  return now

if __name__ == "__main__":
  inFileName = None
  outFileName = None
  outFileNameImage = None
  outFileNamePoints = None
  dropColumns = []
  doLog = False
  useTree = False
  useTreeFast = False
  useNumberedData = True
  coolDown = 0.4

  n = 1
  nextIsOutFile = False
  nextIsOutFileImage = False
  nextIsOutFilePoints = False
  nextIsColumnName = False
  nextIsCoolDown = False
  for a in sys.argv[1:]:
    if nextIsCoolDown:
      nextIsCoolDown = False
      coolDown = float(a)
      if coolDown < 0 or coolDown >= 1:
        print(f"coolDown is expected to be >= 0 and < 1 but {coolDown} was given.")
        exit(1)
      continue

    if nextIsOutFile:
      nextIsOutFile = False
      outFileName = a
      continue

    if nextIsOutFileImage:
      nextIsOutFileImage = False
      outFileNameImage = a
      continue

    if nextIsOutFilePoints:
      nextIsOutFilePoints = False
      outFileNamePoints = a
      continue

    if nextIsColumnName:
      nextIsColumnName = False
      dropColumns.append(a)
      continue

    checkForOldFlags(a)

    if a == '-o':
      nextIsOutFile = True
      continue

    if a == '-oi':
      nextIsOutFileImage = True
      continue

    if a == '-op':
      nextIsOutFilePoints = True
      continue

    if a == '-i':
      nextIsColumnName = True
      continue

    if a == '-numbered':
      useNumberedData = True
      continue

    if a == '-pandas':
      useNumberedData = False
      continue

    if a == "--coolDown" or a == '-cool-down':
      nextIsCoolDown = True
      continue

    if a == '--tree' or a == '-tree':
      useTree = True
      continue

    if a == '--tree-fast' or a == '-tree-fast':
      useTreeFast = True
      continue

    if a == '--log' or a == '-log':
      doLog = True
      continue

    if a == '--help' or a == '-h':
      showHelp()
      exit(0)

    inFileName = a

  if inFileName is None:
    showHelp()
    exit(1)

  tStart = timeStep(tStart, "Initialization")
  if useNumberedData:
    tl = loadTableFromCsv(inFileName, DATA_AND_SET_SIZES, dropColumns)
    columns = tl.heading
    data = tl.data
    del tl
  else:
    data = pd.read_csv(inFileName)
    if len(dropColumns) > 0:
      cs = list(data.columns)
      dc = [c for c in dropColumns if c in cs]
      data = data.drop(columns=dc)
      del cs
      del dc
      del dropColumns
    columns = [cleanupName(n) for n in list(data.columns)]
    data = np.array(data)

  tStart = timeStep(tStart, "Load data")
  if useTree:
    matrix, qf = qMatrixUsingTree(data, debug=doLog)
  elif useTreeFast:
    matrix, qf = qMatrixUsingTreeFast(data, debug=doLog)
  else:
    matrix = qMatrix(data)
    qf = None
  del data

  tStart = timeStep(tStart, "Compute Q-Matrix")
  
  if outFileName is not None:
    if outFileName != "-":
      with open(outFileName, "wt") as f:
        writeMatrix(matrix, columns, f)
    else:
      writeMatrix(matrix, columns, sys.stdout)

  if doLog and qf is not None:
    qf.statistics()

  if outFileNamePoints is not None or outFileNameImage is not None:
    tStart = time.time()
    projection = projectFeatures(matrix, coolDown=coolDown)    
    tStart = timeStep(tStart, "Projection")

    if outFileNamePoints is not None:
      if outFileNamePoints != "-":
        with open(outFileNamePoints, "wt") as f:
          writePoints(projection, columns, f)
      else:
        writePoints(projection, columns, sys.stdout)

    if outFileNameImage is not None:
      plt.scatter(projection[:,0], projection[:,1])
      plt.xlim(-110,110)
      plt.ylim(-110,110)
      if outFileNameImage != "-":
        plt.savefig(outFileNameImage)
      else:
        plt.show()

timeStep(tStartTotal, "Total")

from qFunction.qFunction import *
from qFunction.calcTree import qMatrixUsingTree, qMatrixUsingTreeFast
import pandas as pd
import numpy as np
import sys


def showHelp():
  print(f"{sys.argv[0]} [-o outputTable] [--help] [-i columnName] inputTable")


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
  


if __name__ == "__main__":
  inFileName = None
  outFileName = None
  dropColumns = []
  doLog = False
  useTree = False
  useTreeFast = False

  n = 1
  nextIsOutFile = False
  nextIsColumnName = False
  for a in sys.argv[1:]:
    if nextIsOutFile:
      nextIsOutFile = False
      outFileName = a
      continue

    if nextIsColumnName:
      nextIsColumnName = False
      dropColumns.append(a)
      continue

    if a == '-o':
      nextIsOutFile = True
      continue

    if a == '-i':
      nextIsColumnName = True
      continue

    if a == '--tree':
      useTree = True
      continue

    if a == '--tree-fast':
      useTreeFast = True
      continue

    if a == '--log':
      doLog = True
      continue

    if a == '[--help]':
      showHelp()
      exit(0)

    inFileName = a

  if inFileName is None:
    showHelp()
    exit(1)

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
  if useTree:
    matrix, qf = qMatrixUsingTree(data)
  elif useTreeFast:
    matrix, qf = qMatrixUsingTreeFast(data)
  else:
    matrix = qMatrix(data)
    qf = None
  del data
  
  if outFileName is not None:
    with open(outFileName, "wt") as f:
      writeMatrix(matrix, columns, f)
  else:
    writeMatrix(matrix, columns, sys.stdout)

  if doLog and qf is not None:
    qf.statistics()


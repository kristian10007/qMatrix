import numpy as np
import csv

JUST_DATA = 0
DATA_AND_SET_SIZES = 1
KEEP_ALL = 2

class TableLoader:
  def __init__(self, fileName, separator=','):
    self.reader = csv.reader(open(fileName), delimiter=separator)
    self.heading = None
    self.data = None
    self.sets = None
    self.ignoreColumns = []

  def ignoreColumn(self, name):
    self.ignoreColumns.append(name)
    

  def load(self, store=KEEP_ALL):
    if self.data is not None:
      return

    sets = None
    data = []
    for row in self.reader:
      if row is None:
        continue

      if self.heading is None:
        self.heading = [name for name in row if name not in self.ignoreColumns]
        sets = [(None if name in self.ignoreColumns else {}) for name in row]
        continue

      rowData = []
      nCols = len(sets)
      for n, v in enumerate(row):
        if n >= nCols:
          break

        if sets[n] is not None:
          v = f"{v}"
          if v in sets[n]:
            p = sets[n][v]
          else:
            p = len(sets[n])
            sets[n][v] = p

          rowData.append(p)

      data.append(np.array(rowData, dtype=int))

    data = np.array(data)
    self.data = data
    del data

    if store == DATA_AND_SET_SIZES:
      self.sets = [len(s) for s in sets if s is not None]
    elif store == KEEP_ALL:
      self.sets = []
      for col in sets:
        if col is None:
          continue
        mapping = [None for _ in col.keys()]
        for k in col.keys():
          mapping[col[k]] = k
        self.sets.append(mapping)
    del sets

        

def loadTableFromCsv(fileName, store=KEEP_ALL, ignoreColumns=None):
  tl = TableLoader(fileName)
  if ignoreColumns is not None:
    tl.ignoreColumns = ignoreColumns
  tl.load(store)
  return tl

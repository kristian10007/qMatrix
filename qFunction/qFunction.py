import pandas as pd
import numpy as np

def q_functionDf(df: pd.DataFrame, col_A_name: str, col_B_name: str) -> float:
    """
    Calculates the Q-score for two columns, measuring the logical dependency
    of col_B on col_A. A score of 0 indicates perfect functional dependency (A -> B).
    """
    if col_A_name not in df.columns or col_B_name not in df.columns:
        raise ValueError("One or both column names are not in the DataFrame.")

    unique_A = df[col_A_name].dropna().unique()
    unique_B = df[col_B_name].dropna().unique()

    num_unique_A = len(unique_A)
    num_unique_B = len(unique_B)

    if num_unique_A == 0 or num_unique_B <= 1:
        return 0.0

    num_observed_pairs = df[[col_A_name, col_B_name]].dropna().drop_duplicates().shape[0]

    numerator = num_observed_pairs - num_unique_A
    denominator = num_unique_A * (num_unique_B - 1)

    if denominator == 0:
        return 0.0

    return numerator / denominator


def q_functionNp(df: np.ndarray, col_A: int, col_B: int) -> float:
    """
    Calculates the Q-score for two columns, measuring the logical dependency
    of col_B on col_A. A score of 0 indicates perfect functional dependency (A -> B).
    """
    assert( isinstance(df, np.ndarray) )
    assert( len(df.shape) == 2 )
    
    if min(df.shape[0], df.shape[1]) < 1:
        raise ValueError("Array has to contain at least one element.")

    if min(col_A, col_B < 0) or max(col_A, col_B) >= df.shape[1]:
        raise ValueError("One or both column index are out of range.")

    if col_A == col_B:
      return 0.0

    num_unique_A = len(set(df[:,col_A]))
    num_unique_B = len(set(df[:,col_B]))

    if num_unique_A == 0 or num_unique_B <= 1:
        return 0.0

    num_observed_pairs = len(set(zip(df[:,col_A], df[:,col_B]))) 

    numerator = num_observed_pairs - num_unique_A
    denominator = num_unique_A * (num_unique_B - 1)

    if denominator == 0:
        return 0.0

    return numerator / denominator


def qMatrix(df: np.ndarray):
    """Optimized version of computing the Q-matrix.
    The expensive stuff is done at most (n ( n + 1 ) / 2) times.
    """
    assert( isinstance(df, np.ndarray) )
    assert( len(df.shape) == 2 )
    nFeatures = df.shape[1]
    m = np.zeros(shape=(nFeatures, nFeatures))

    setSizes = [ len(set(df[:, j])) for j in range(nFeatures) ]

    def q(i,j):
      if i == j or setSizes[i] < 1 or setSizes[j] < 1:
        return 0.0, 0.0

      nPairs = len(set(zip(df[:,i], df[:,j])))
      q1, q2 = 0.0, 0.0
      if setSizes[j] >= 2:
        q1 = (nPairs - setSizes[i]) / (setSizes[i] * (setSizes[j] - 1))
      if setSizes[i] >= 2:
        q2 = (nPairs - setSizes[j]) / (setSizes[j] * (setSizes[i] - 1))
      return q1, q2

    for i in range(nFeatures):
      for j in range(i + 1):
        q1, q2 = q(i, j)
        m[i, j] = q1
        m[j, i] = q2

    return m


class Q:
  def __init__(self, data):
    assert( isinstance(data, np.ndarray) )
    assert( len(data.shape) == 2 )
    assert( data.shape[0] >= 1 )
    assert( data.shape[1] >= 1 )

    self.data = data
    self.nFeatures = data.shape[1]

    self.qValues = np.zeros(shape=(self.nFeatures, self.nFeatures), dtype=float)

    for i in range(self.nFeatures):
      for j in range(self.nFeatures):
        if i != j:
          self.qValues[i,j] = None

    self.nRequested = 0
    self.nSetRequested = 0
    self.nPairRequested = 0
    self.nCalculated = 0
    self.nSetCalculated = 0
    self.nPairCalculated = 0

  def calc(self, i, j):
    self.nRequested += 1
    q = 0.0
    if i != j:
      self.nCalculated += 1
      a = self.setSize(i)
      b = self.setSize(j)
      if a >= 1 and b >= 2:
        r = self.pairSize(i,j)
        q = (r - a) / (a * (b - 1))

    if self.qValues[i,j] is None:
      self.qValues[i,j] = q

    return q

  def setSize(self, n):
    self.nSetRequested += 1
    if n < 0 or n >= self.nFeatures:
      return 0

    self.nSetCalculated += 1
    return len(set(self.data[:,n]))

  def pairSize(self, i, j):
    self.nPairRequested += 1
    if min(i, j) < 0 or max(i,j) >= self.nFeatures:
      return 0

    self.nPairCalculated += 1
    return len(set(zip(self.data[:, i], self.data[:, j])))

  def statistics(self):
    print(f"nRequested = {self.nRequested}")
    print(f"nSetRequested = {self.nSetRequested}")
    print(f"nPairRequested = {self.nPairRequested}")
    print(f"nCalculated = {self.nCalculated}")
    print(f"nSetCalculated = {self.nSetCalculated}")
    print(f"nPairCalculated = {self.nPairCalculated}")

    print(f"Optimized Q-matrix would be:")
    print(f"nCalculated = {(self.nFeatures * (self.nFeatures - 1)) / 2}")
    print(f"nSetCalculated = {self.nFeatures}")
    print(f"nPairCalculated = {(self.nFeatures * (self.nFeatures - 1)) / 2}")






class Qfast:
  def __init__(self, data):
    assert( isinstance(data, np.ndarray) )
    assert( len(data.shape) == 2 )
    assert( data.shape[0] >= 1 )
    assert( data.shape[1] >= 1 )

    self.data = data
    self.nFeatures = data.shape[1]
    self.setSizes = np.zeros(shape=(self.nFeatures, ), dtype=int)
    self.qValues = np.zeros(shape=(self.nFeatures, self.nFeatures), dtype=float)

    for i in range(self.nFeatures):
      self.setSizes[i] = -1
      for j in range(self.nFeatures):
        if i != j:
          self.qValues[i,j] = None

    self.nRequested = 0
    self.nSetRequested = 0
    self.nPairRequested = 0
    self.nCalculated = 0
    self.nSetCalculated = 0
    self.nPairCalculated = 0

  def calc(self, i, j):
    self.nRequested += 1
    if np.isnan(self.qValues[i, j]):
      self.nCalculated += 1
      q1 = 0.0
      q2 = 0.0
      if i != j:
        a = self.setSize(i)
        b = self.setSize(j)
        r = None
        if a >= 1 and b >= 2:
          r = self.pairSize(i,j)
          q1 = (r - a) / (a * (b - 1))
        if a >= 2 and b >= 1:
          if r is None:
            r = self.pairSize(i,j)
          q2 = (r - b) / (b * (a - 1))
      self.qValues[i, j] = q1
      self.qValues[j, i] = q2
    return self.qValues[i, j]

  def setSize(self, n):
    self.nSetRequested += 1
    if n < 0 or n >= self.nFeatures:
      return 0

    if self.setSizes[n] < 0:
      self.nSetCalculated += 1
      self.setSizes[n] = len(set(self.data[:,n]))

    return self.setSizes[n]

  def pairSize(self, i, j):
    self.nPairRequested += 1
    if min(i, j) < 0 or max(i,j) >= self.nFeatures:
      return 0

    self.nPairCalculated += 1
    return len(set(zip(self.data[:, i], self.data[:, j])))

  def statistics(self):
    print(f"nRequested = {self.nRequested}")
    print(f"nSetRequested = {self.nSetRequested}")
    print(f"nPairRequested = {self.nPairRequested}")
    print(f"nCalculated = {self.nCalculated}")
    print(f"nSetCalculated = {self.nSetCalculated}")
    print(f"nPairCalculated = {self.nPairCalculated}")

    print(f"Optimized Q-matrix would be:")
    print(f"nCalculated = {(self.nFeatures * (self.nFeatures - 1)) / 2}")
    print(f"nSetCalculated = {self.nFeatures}")
    print(f"nPairCalculated = {(self.nFeatures * (self.nFeatures - 1)) / 2}")

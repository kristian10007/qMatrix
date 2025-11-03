from qFunction.qFunction import *

def predict2ndLayer(q):
  assert( isinstance(q, Q) or isinstance(q, Qfast) or isinstance(q, QFunction) )

  nFeatures = q.nFeatures 
  nRows = q.nRows
  nPairs = int((nFeatures * (nFeatures - 1)) / 2 )

  assert(nFeatures > 0)
  assert(nRows > 0)

  def tau(t, i, j, k):
    # If destination is one of the sources then we have a function.
    if i == k or j == k:
      return 0.0

    # If one part of the pair is a function than the pair is a function.
    if q.calc(i, k) == 0.0 or q.calc(j, k) == 0.0:
      return 0.0

    # If the destination has no or only one value then there is no second
    # value to show that we have no function therefore it is a function.
    b = q.setSize(k)
    if b < 2:
      return 0.0

    # If the pair has the same amount of rows as the table then the pair
    # is a indexing set and therefore we have a function.
    a = q.pairSize(i, j)
    if a == nRows:
      return 0.0

    # Jut to be complete: if there is no pair then there is nothing to show
    # that this is not a function. Therefore it is a function.
    # But: how is this possible in a non empty table?
    if a < 1:
      return 0.0

    # When there was no shortcut then we have to calculate.
    return ((t / a) - 1) / (b - 1)


  matrix = np.zeros((nPairs, nFeatures))
  p = -1
  for i in range(nFeatures):
    for j in range(i):
      p += 1
      for k in range(nFeatures):
        matrix[p, k] = tau(nRows, i, j, k)

  return matrix


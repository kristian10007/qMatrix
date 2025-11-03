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

    # Just to be complete: if there is no pair then there is nothing to show
    # that this is not a function. Therefore it is a function.
    # But: how is this possible in a non empty table?
    if a < 1:
      return 0.0

    # If there are more rows in the table than possible entries for the
    # relation then all connections are possible.
    if t > a * b:
      return 1.0

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


def predictFunctions(qf, layer2Matrix, columns, logicDepLimit=0.2):
  layer2Names = []
  nFeatures = qf.nFeatures
  assert(nFeatures == len(columns))

  for i in range(nFeatures):
    for j in range(i):
      layer2Names.append('{ "' + columns[j] + '", "' + columns[i] + '" }')

  functions = []
  for i in range(nFeatures):
    for j in range(nFeatures):
      if i == j:
        continue

      elif qf.qValues[i,j] < logicDepLimit:
        functions.append( (qf.qValues[i, j], '{ "' + columns[i] + '" }', columns[j]) )

  p = -1
  for i in range(nFeatures):
    for j in range(i):
      p += 1
      for k in range(nFeatures):
        if k == i or k == j:
          continue

        if qf.qValues[i, k] == 0 or qf.qValues[j, k] == 0:
          continue

        if layer2Matrix[p, k] < logicDepLimit:
          functions.append( (layer2Matrix[p, k], layer2Names[p], columns[k]) )

  return functions

def showFunctions(functions, file):
  for (q, s, d) in functions:
    if q == 0.0:
      file.write(s + ' -> { "' + d + '" }\n')
    else:
      file.write(s + ' ~> { "' + d + '" }  (' + str(q) + ')\n')


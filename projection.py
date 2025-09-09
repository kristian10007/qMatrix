def qMatrixToForceScalers(qMatrix):
    scalers = qMatrix - 0.5
    for n in range(scalers.shape[0]):
        scalers[n,n] = 0.0
    return scalers

def createPoints(nPoints):
    points = [[100*np.sin(float(n)), 100*np.cos(float(n))] for n in range(nPoints)]
    points = np.array(points)
    return points

def step(pts, scalers, f = 1.0, fieldSize=100.0):
    # Calculate the sum of forces pulling and pushing to each point.
    # Think of the forces that would pull to planets.
    d = (f * scalers) @ pts

    # Apply the forces to the points.
    pts = pts - d

    # Move the points to the center of the coordinate system
    c = np.sum(pts,axis=0) / pts.shape[0]
    pts = pts - c

    # Rescale so our points will fill the expected square
    m = np.max(np.abs(pts))
    pts = ((fieldSize / m) * pts)
    return pts



def projectFeatures(qMatrix, coolDown=0.4, fieldSize=100.0, epochs=20, drawSteps=False):
    # Check input values
    assert( 0 <= coolDown and coolDown < 1 )
    assert( isinstance(qMatrix, np.ndarray) )
    assert( len(qMatrix.shape) == 2 )
    assert( qMatrix.shape[0] == qMatrix.shape[1] )
    nPoints = qMatrix.shape[0]
    
    # Generate matrix to compute the forces
    scalers = qMatrixToForceScalers(qMatrix)
    assert( isinstance(scalers, np.ndarray) )
    assert( len(scalers.shape) == 2 )
    assert( scalers.shape[0] == nPoints )
    assert( scalers.shape[1] == nPoints )
    assert( np.min(scalers) >= -0.5 )
    assert( np.max(scalers) <= 0.5 )
    assert( np.max(np.abs(scalers)) > 0.0 )

    # Initialize the points
    points = createPoints(nPoints)
    assert( isinstance(points, np.ndarray) )
    assert( len(points.shape) == 2 )
    assert( points.shape[0] == nPoints )
    assert( points.shape[1] == 2 )

    # move the points around
    f = 1.0
    for e in range(epochs):
        if drawSteps and e % 2 == 0:
            plt.scatter(points[:,0], points[:,1])
        points = step(points, scalers=scalers, f=f, fieldSize=fieldSize)
        f = f * (1.0 - coolDown)

    if drawSteps:
        plt.show()

    return points

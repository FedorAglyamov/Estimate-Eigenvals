# Program for iterative estimates of eignvalues and eigenvectors of matrices
# using the Power and Inverse Power Methods


# Imports
import numpy as np
import sys


# Main function for running program
def main():

    # Set numpy options
    floatFormat = "{:.4f}".format
    np.set_printoptions(formatter = {"float_kind":floatFormat})
    
    # Matrix to be operated on
    a = np.array([[ 7,  2,  1, -3],
                  [-4,  1,  6,  3],
                  [ 7,  2, -5,  7],
                  [-5, -1,  3, -4]])
    # Initial vector
    x = np.array([[ 1],
                  [ 0],
                  [ 0],
                  [ 0]])
    # Step count (number of iterations)
    steps = 7

    # Call estimating methods
    power_method(a, x, steps)
    inverse_power_method(a, x, 0, steps)
    inverse_power_method(a, x, 8, steps)


# Find dominant eigenval of passed matrix using Power Method
def power_method(a, x, steps):

    # Check that number of steps is valid
    if steps < 1:
        error()

    # Format strings
    format0 = "Estimating dominant eigenval using Power Method with {} iterations".format
    format1 = ">>>>>>>>>>>>>>>>>>>>>>>> STEP {} <<<<<<<<<<<<<<<<<<<<<<<<\n".format
    format2 = "Estimated Dominant Eigenval = {:.4f}".format
    format3 = "Actual Dominant Eigenval = {:.4f}".format
    format4 = "x_{step} = {x}".format
    formatBars = "----------------------------------------------------------------------"
    formatStars = "**********************************************************************"

    # Print header
    print(formatStars)
    print(format0(steps))
    print(formatStars + "\n")

    # Perform step 0
    print(format1(0))
    print(format4(step = 0, x = x[:, 0]) + "\n")
    result = a @ x
    u = getMu(result)
    print_pow_step(a, x, u, result, 0)
    pastX = result
    x = result

    # Iterate passed number of times
    for i in range(1, steps):

        print(format1(i))
        
        # Find new x vector
        x = (1 / u) * x
        print_finding_x(x, pastX, u, i)

        # Perform calculations
        result = a @ x
        u = getMu(result)
        print_pow_step(a, x, u, result, i)
        pastX = result
        x = result

    # Print estimate for dominant eigenval
    print(formatBars)
    print(format2(u))
    print(formatBars)
    print(format3(getMu(np.linalg.eigvals(a))))
    print(formatBars + "\n\n")
    
    
# Print info for current Power Method step
def print_pow_step(a, x, u, result, step):

    # Format strings
    format1 = "Ax_{} = ".format
    format2 = "   {aRow} {xRow}   =   {resultRow}".format
    format3 = "\nnu_{} = {:.4f}\n".format

    rows = a.shape[0]

    # Print first row
    print(format1(step))

    # Iterate through rows and print results
    for r in range(0, rows):

        aRow = a[r]
        xRow = x[r]
        resultRow = result[r]
        
        print(format2(aRow = aRow, xRow = xRow, resultRow = resultRow))

    # Print final row
    print(format3(step, u))
    

# Find eigenval of passed matrix using Inverse Power Method
def inverse_power_method(a, x, alpha, steps):

    # Check that number of steps is valid
    if steps < 1:
        error()

    # Format strings
    format0 = "Estimating using Inverse Power Method with " \
                "alpha = {alpha} and {steps} iterations".format
    format1 = ">>>>>>>>>>>>>>>>>>>>>>>> STEP {} <<<<<<<<<<<<<<<<<<<<<<<<\n".format
    format2 = "Estimated Eigenval = {:.4f}".format
    format3 = "Actual Eigenvals = {}".format
    format4 = "x_{step} = {x}".format
    formatBars = "----------------------------------------------------------------------"
    formatStars = "**********************************************************************"

    iMatrix = np.identity(a.shape[0])

    # Print header
    print(formatStars)
    print(format0(steps = steps, alpha = alpha))
    print(formatStars + "\n")

    # Print step 0
    print(format1(0))
    print(format4(step = 0, x = x[:, 0]) + "\n")
    y = np.linalg.solve((a - alpha * iMatrix), x)
    mu = getMu(y)
    v = alpha + (1 / mu)
    print_inv_pow_step(a, x, alpha, iMatrix, y, v, mu, 0)
    pastX = x
    x = (1 / mu) * y

    # Iterate passed number of times
    for i in range(1, steps):

        print(format1(i))
        print_finding_x(x, pastX, mu, i)
        y = np.linalg.solve((a - alpha * iMatrix), x)
        mu = getMu(y)
        v = alpha + (1 / mu)
        print_inv_pow_step(a, x, alpha, iMatrix, y, v, mu, i)

        # Update x vector
        pastX = x
        x = (1 / mu) * y

    # Print estimate for eigenval
    print(formatBars)
    print(format2(v))
    print(formatBars)
    print(format3(np.linalg.eigvals(a)))
    print(formatBars + "\n\n")


# Print info for current Inverse Power Method step
def print_inv_pow_step(a, x, alpha, iMatrix, y, v, mu, step):

    # Format strings
    format1 = "Solving (A - aI)y_{step} = x_{step} for y_{step} :".format
    format2 = "   ({aRow}   -   {alpha}  {iMatrixRow}) y_{step}   =   {xRow}".format
    format3 = "y_{step} = {y}T".format
    format4 = "mu_{step} = {mu}".format
    format5 = "v_{step} = {v}".format

    rows = a.shape[0]

    print(format1(step = step))
    
    # Iterate through rows and print results
    for r in range(0, rows):

        print(format2(step = step, aRow = a[r], alpha = alpha, iMatrixRow = iMatrix[r], xRow = x[r]))
    
    # Print results
    print("\n" + format3(step = step, y = y[:, 0]))
    print(format4(step = step, mu = mu))
    print(format5(step = step, v = v) + "\n")
    

# Print computation for finding current x vector
def print_finding_x(x, pastX, mu, step):

    # Format strings
    muString =  "{:.4f}".format(mu)
    format2 = "x_{step} = (1 / nu_{pastStep}) Ax_{pastStep} = (1 / {mu})  {pastX}T =".format
    format3 = "   {x}T\n".format

    # Print info
    print(format2(step = step, pastStep = step - 1, mu = muString, pastX = pastX[:, 0]))
    print(format3(x = x[:, 0]))


# Get value mu
def getMu(x):

    # Account for matrices of different dimensions
    maxVal = x[0]
    if (x.ndim == 2):
        maxVal = x[0, 0]
        
    maxAbsVal = abs(maxVal)

    # Iterate through vector and find elem with greatest absolute val
    for elem in x.flat:

        curAbsVal = abs(elem)
        if curAbsVal > maxAbsVal:
            maxVal = elem
            maxAbsVal = curAbsVal

    return maxVal


# Print error message
def error():
    print("\n>>>>>>>>>>  Error <<<<<<<<<<\n")
    sys.exit()

    
# Run main function
main()

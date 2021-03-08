import matplotlib.pyplot as plt
import numpy as np


strike = 1
p_up = 10/21 # risk neutral probability
p_down = 11/21

# there are 52 weeks. All matrices will have shape (52, 52)
underlying = np.zeros((52, 52))

for m in range(52):
    for n in range(52):
        if m==n:
            if n==0:
                underlying[m,n]=1
            else:
                underlying[m,n]=underlying[m-1, n-1]/1.1
        else:
            underlying[m, n] = underlying[m , n - 1]*1.1





def swing_option(underlying, type="call", swing_previous=None):

    def exercise(type, spot_price, strike_price=1):
        if type == "call":
            return (spot_price - strike_price) * 50
        elif type == "put":
            return (strike_price - spot_price) * 50000
        else:
            raise ValueError('option type has to be either call or put')

    swing = np.zeros_like(underlying)
    exercised = np.zeros_like(underlying)

    if swing_previous is None: # this is the swing1up
        swing_previous = np.zeros_like(underlying)

    for n in range(51, -1, -1):
        for m in range(n+1):

            if n == 51:
                value_not_exercise = 0
                value_if_exercise = exercise(type, underlying[m, n])
            else:
                value_not_exercise = p_up * swing[m, n + 1] + p_down * swing[m + 1, n + 1]
                value_if_exercise = exercise(type, underlying[m, n]) + p_up * swing_previous[m, n + 1] + p_down * swing_previous[m + 1, n + 1]

            if value_if_exercise > value_not_exercise + 1: # plus 1 because two numbers could be equal, but only different due to rounding
                exercised[m, n] = 1
                swing[m, n] = value_if_exercise
            elif value_not_exercise > value_if_exercise + 1:
                exercised[m, n] = 0
                swing[m, n] = value_not_exercise
            else:
                exercised[m, n] = 0.5
                swing[m, n] = value_not_exercise


    return swing, exercised


swing1up, exercised = swing_option(underlying, type="call", swing_previous=None)
swing2up, exercised = swing_option(underlying, type="call", swing_previous=swing1up)
swing3up, exercised = swing_option(underlying, type="call", swing_previous=swing2up)
swing4up, exercised = swing_option(underlying, type="call", swing_previous=swing3up)
plt.imshow(swing4up)
plt.imshow(exercised)

swing1down, exercised = swing_option(underlying, type="put", swing_previous=None)
swing2down, exercised = swing_option(underlying, type="put", swing_previous=swing1down)
swing3down, exercised = swing_option(underlying, type="put", swing_previous=swing2down)
swing4down, exercised = swing_option(underlying, type="put", swing_previous=swing3down)
plt.imshow(swing4down)
plt.imshow(exercised)
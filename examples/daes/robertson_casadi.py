from casadi import *
import pylab as plt

y1 = SX.sym("y1")
y2 = SX.sym("y2")
y3 = SX.sym("y3")

yp1 = SX.sym("yp1")
yp2 = SX.sym("yp2")
yp3 = SX.sym("yp3")

# state
x_impl  = vertcat(
  y1,
  y2,
)

# state derivative
dx_impl = vertcat(
  yp1,
  yp2,
)

z = y3

# implicit equation
alg = vertcat(
  yp1 - (-0.04 * y1 + 1e4 * y2 * y3),
  yp2 - (0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2**2),
  y1 + y2 + y3 - 1,
)

dae = {"x_impl": x_impl, "dx_impl": dx_impl, "z": z, "alg": alg}

# Perform index reduction
dae_reduced, stats = dae_reduce_index(dae)
print("New DAE index:")
print(stats)
print("Here's the reduced DAE:")
print(dae_reduced)

# The DAE is not yet in a form that CasADi integrator can handle (semi-explicit).
# Let's convert it, ad obtain some adaptors
dae_se, state_to_orig, phi = dae_map_semi_expl(dae, dae_reduced)

t0 = 0
t1 = 1e7
num = 100
grid = list(np.logspace(-6, 7, num=num))
tf = DM(grid).T
intg = integrator("intg", "idas", dae_se, t0, grid)

# Before we can integrate, we need a consistant initial guess;
# consistency equations should automatically identify everything else

# Encode the desire to only enforce y and dx
init_strength = {}
init_strength["x_impl"] = vertcat(1, 0)
init_strength["dx_impl"] = vertcat(-0.04, 0.04)
init_strength["z"] = vertcat(0)

# Obtain a generating Function for initial guesses
init_gen = dae_init_gen(dae, dae_reduced, "ipopt", init_strength)

init = {}
# suggest to initialize with the left-hanging solution by having x=-1 as initial guess 
init["x_impl"]  = vertcat(1, 0)  
init["dx_impl"] = vertcat(-0.04, 0.04)
init["z"]  = 0

xz0 = init_gen(**init)
print("We have a consistent initial guess now to feed to the integrator:")
print(xz0)

print("Look, we found that force in the pendulum at t=0 equals:")
print(state_to_orig(xf=xz0['x0'], zf=xz0['z0'])["z"])

print("A consistent initial guess should make the invariants zero (up to solver precision):")
print(phi(x=xz0['x0'], z=xz0['z0']))

# Integrate and get resultant xf,zf on grid
sol = intg(**xz0)

print(sol['xf'].shape)

# Solution projected into original DAE space
sol_orig = state_to_orig(xf=sol["xf"], zf=sol["zf"])

# We can see the large-angle pendulum motion play out well in the u state
# y1, y2 = vertsplit(sol_orig["x_impl"])
# y1 = sol_orig["x_impl"][0, :]
# y2 = sol_orig["x_impl"][1, :]
y1, y2 = np.array(sol_orig["x_impl"])
t = np.array(tf)[0]
# plt.plot(tf.T, sol_orig["x_impl"].T)
plt.plot(t, y1)
plt.plot(t, y2 * 1e4)
plt.plot(tf.T, sol_orig["z"].T)
plt.grid(True)
plt.legend([e.name() for e in vertsplit(x_impl)] + [e.name() for e in vertsplit(z)])
plt.xscale("log")
plt.xlabel("Time [s]")
plt.title("Boundary value problem solution trajectory")

plt.show()

exit()

# A perfect integrator will perfectly preserve the values of the invariants over time
# Integrator errors make the invariants drift in practice
# This is not a detail; the pendulum length is in fact growing!
error = phi(x=sol["xf"],z=sol["zf"])["I"]

plt.figure()
plt.plot(tf.T,error.T)
plt.grid(True)
plt.xlabel("Time [s]")
plt.title("Evolution trajectory of invariants")

# There are some techniques to avoid the drifting of invariants associated with index reduction
#  - Method of Dummy Derviatives (not implemented)
#    hard implementation details: may need to switch between different choices online
#     -> integration in parts triggered by zero-crossing events
#  - Stop the integration every once in a while and project back (not implemented)
#  - Baumgarte stabilization (implemented): build into the equations a form of continuous feedback that brings back deviations in invariants back into the origin


# Demonstrate Baumgarte stabilization with a pole of -1.
# Drift is absent now.
(dae_reduced,stats) = dae_reduce_index(dae, {"baumgarte_pole": -1})
[dae_se, state_to_orig, phi] = dae_map_semi_expl(dae, dae_reduced)
intg = integrator("intg","idas",dae_se,0,grid)
init_gen = dae_init_gen(dae, dae_reduced, "ipopt", init_strength)
xz0 = init_gen(**init)
sol = intg(**xz0)
error = phi(x=sol["xf"],z=sol["zf"])["I"]
plt.figure()
plt.plot(tf.T,error.T)
plt.grid(True)
plt.xlabel("Time [s]")
plt.title("Boundary value problem solution trajectory with Baumgarte pole=-1")



# Sweep across different integrator precisions and Baumgarte poles.
# The choice of pole is not really straightforward
plt.figure()

poles = [-0.1,-1,-10]
precisions = [0.1,1,10]

abstol_default = 1e-8
reltol_default = 1e-6

plt.subplot(len(precisions),len(poles),1)

for i,pole in enumerate(poles):
  for j,precision in enumerate(precisions):
    (dae_reduced,stats) = dae_reduce_index(dae, {"baumgarte_pole": pole})
    [dae_se, state_to_orig, phi] = dae_map_semi_expl(dae, dae_reduced)
    intg = integrator("intg","idas",dae_se,0,grid,{"abstol":abstol_default*precision,"reltol":reltol_default*precision})
    init_gen = dae_init_gen(dae, dae_reduced, "ipopt", init_strength, {"ipopt.print_level":0,"print_time": False})
    xz0 = init_gen(**init)
    sol = intg(**xz0)
    nfevals = intg.stats()["nfevals"]
    error = phi(x=sol["xf"],z=sol["zf"])["I"]
    plt.subplot(len(precisions),len(poles),j*len(poles)+i+1)
    plt.plot(tf.T,error.T)
    plt.grid(True)
    plt.xlabel("Time [s]")
    plt.title("pole=%0.1f, abstol=%0.0e => nfevals=%d" % (pole,abstol_default*precision,nfevals))

plt.show()

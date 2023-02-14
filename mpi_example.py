import numpy as np
import pickle
import os

from mpi4py import MPI

from worker import TaskMaster, Worker

def simulate(sim):

    return "I've simulated {}".format(sim["name"])

if __name__ == "__main__":
    comm = MPI.COMM_WORLD   # get MPI communicator object
    size = comm.size        # total number of processes
    rank = comm.rank        # rank of this process

    if rank == 0:
        # Master process executes code below

        SolveController = TaskMaster([])

        for timestep in range(10):
            print("time = {}".format(timestep))
            simulations = []

            simulations.append({"name":"first calc"})
            simulations.append({"name":"second calc"})
            simulations.append({"name":"third calc"})

            SolveController.task_list = simulations
            SolveController.do_work()


        SolveController.close_workers()

    else:
        Solver = Worker(simulate)
        Solver.work_loop()

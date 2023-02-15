import enum

from mpi4py import MPI

class Status(enum.Enum):
    READY = 0
    DONE = 1
    START = 2
    EXIT = 3

class Worker:

    def __init__(self, work_function):
        self.comm = MPI.COMM_WORLD   # get MPI communicator object
        self.status = MPI.Status()   # get MPI status object
        self.name = MPI.Get_processor_name()
        self.rank = self.comm.rank
        self.function = work_function
        print("I am a worker with rank %d on %s." % (self.rank, self.name))


    def work_loop(self):
        while True:
            self.comm.send(None, dest=0, tag=Status.READY.value)
            task = self.comm.recv(source=0, tag=MPI.ANY_TAG, status=self.status)
            tag = self.status.Get_tag()

            if tag == Status.START.value:

                result = self.function(task)
                self.comm.send(result, dest=0, tag=Status.DONE.value)
            elif tag == Status.EXIT.value:
                break

        self.comm.send(None, dest=0, tag=Status.EXIT.value)


class TaskMaster:
    def __init__(self, task_list):
        self.comm = MPI.COMM_WORLD   # get MPI communicator object
        self.status = MPI.Status()   # get MPI status object
        self.rank = self.comm.rank
        self.size = self.comm.size
        self.task_list = task_list

        print("Master starting with {} workers".format(self.size - 1))


    def do_work(self):
        print("Work Starting")
        task_index = 0
        while task_index < len(self.task_list):
            data = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=self.status)
            source = self.status.Get_source()
            tag = self.status.Get_tag()
            if tag == Status.READY.value:
                self.comm.send(self.task_list[task_index], dest=source, tag=Status.START.value)
                print("Sending task %d of %d to worker %d" % (task_index, len(self.task_list), source))
                task_index += 1
            elif tag == Status.DONE.value:
                print("Got data {} from worker {}".format(data, source))
            elif tag == Status.EXIT.value:
                print("Worker %d exited." % source)

        print("Work Done")

    def close_workers(self):
        num_workers = self.size - 1
        closed_workers = 0
        print("Master stopping {} workers".format(num_workers))
        while closed_workers < num_workers:
            data = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=self.status)
            source = self.status.Get_source()
            tag = self.status.Get_tag()
            if tag == Status.READY.value:
                    self.comm.send(None, dest=source, tag=Status.EXIT.value)
            elif tag == Status.EXIT.value:
                print("Worker %d exited." % source)
                closed_workers += 1
from multiprocessing import Process, Pipe, Value


class PipeClass(object):
    """ Simple thread class that can close safely and share data asynchronously through a pipe. """

    __slots__ = ["exiting", "parent_connection", "child_connection", "child_process"]

    def __init__(self):
        """ Initialize all variables. """
        self.exiting = Value('b', 0)
        self.parent_connection, self.child_connection = Pipe(duplex=True)
        self.child_process = Process(target=PipeClass._child_main_loop, args=(self,))

    def _child_main_loop(self):
        """ This is the loop the child process runs in. child_send_func may be used as the main
        code, as it does not need to return anything. """
        self.child_start_func()
        while self.exiting.value != 1:
            if self.child_connection.poll() is True:
                in_val = self.child_connection.recv()
                self.child_receive_func(in_val)
            out_val = self.child_send_func()
            if out_val is not None:
                self.child_connection.send(out_val)
        self.child_exit_func()
        self.child_connection.close()

    def start_child(self):
        """ This makes starting an explicit class method. """
        self.child_process.start()

    def main_process_receive(self):
        """ Receive data from the child program.

        @return (any)
        any object sent by child process that can be pickled.

        todo: @return (bytearray)
        todo: any object sent by child process that can be packed into a flatbuffer.
        [MASSIVE SPEED UP!]
        """
        if self.parent_connection.poll() is True:
            in_val = self.parent_connection.recv()
            return in_val
        return None

    def main_process_send(self, data):
        """ Send data to the child program.

        @data (any)
        any object that can be pickled before sending to the child process.
        """
        if data is not None:
            self.parent_connection.send(data)

    def soft_exit(self):
        """ Correctly ends child program. """
        self.exiting.value = 1
        self.child_process.join()

    def child_start_func(self):
        """ Virtual function run when child program is first created.
        Should be implemented in derived class. """
        pass

    def child_receive_func(self, data):
        """ Virtual function run when child program receives a message from the parent program.
        Should be implemented in derived class. """
        pass

    def child_send_func(self):
        """ Virtual function run repeatedly until end. Return data is sent to parent program.
        Should be implemented in derived class. """
        pass

    def child_exit_func(self):
        """ Virtual function run right before child program is destroyed.
        Should be implemented in derived class. """
        pass


class ExamplePipeClass(PipeClass):
    def __init__(self):
        super().__init__()
        self.i = 1
        self.j = 0
        self.j_t = 0

    def inf_loop(self):
        self.start_child()
        while self.j < 10:
            self.main_process_send(self.j + 1)
            self.j_t = self.main_process_receive()
            if self.j_t is not None:
                self.j = self.j_t
            print("parent:\t", self.j)
        print('exited')
        self.soft_exit()

    def child_receive_func(self, data):
        print("child:\t", self.j)

    def child_send_func(self):
        self.i += 1
        return self.i

if __name__ == '__main__':
    c = ExamplePipeClass()
    c.inf_loop()

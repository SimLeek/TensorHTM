from multiprocessing import Process, Pipe, Value

class PipeClass(object):

    def __init__(self):
        self.exiting = Value('b', 0)

    def _other_process(self):
        in_val = None
        while self.exiting.value != 1:
            if self.child_conn.poll() is True:
                in_val = self.child_conn.recv()
                self.child_receive_func(in_val)
            out_val = self.child_send_func()
            if out_val is not None:
                self.child_conn.send(out_val)
        print('closed')
        self.child_conn.close()

    def main_process_start(self):
        self.parent_conn, self.child_conn = Pipe(duplex=True)
        self.p = Process(target=PipeClass._other_process, args=(self,))
        self.p.start()

    def main_process_receive(self):
        if self.parent_conn.poll() is True:
            in_val = self.parent_conn.recv()
            return in_val
        return None

    def main_process_send(self, data):
        if data is not None:
            self.parent_conn.send(data)

    def soft_exit(self):
        self.exiting.value = 1
        self.p.join()

    def child_receive_func(self, data):
        pass

    def child_send_func(self):
        pass

    def child_exit_loop(self):
        pass

class TensorClass(PipeClass):
    def __init__(self):
        super().__init__()
        self.i=1
        self.j=0

    def inf_loop(self):
        self.main_process_start()
        while self.j<10:
            self.main_process_send(self.j+1)
            self.j_t = self.main_process_receive()
            if self.j_t is not None:
                self.j = self.j_t
            print(self.j)
        print('exited')
        self.soft_exit()

    def child_receive_func(self, data):
        print(data)

    def child_send_func(self):
        self.i +=1
        return self.i


if __name__ == '__main__':
    c = TensorClass()
    c.inf_loop()
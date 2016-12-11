from multiprocessing import Process, Pipe

def tensor_loop(conn):
    i=0
    while i<100000:
        print('d')
        if conn.poll() is True:
            #todo: python uses pickle to send through pipe, which is slow. Use Google FlatBuffers and send_bytes.
            i+= conn.recv()
        conn.send(i)
        i+=1
    while conn.recv() != 'end':
        pass
    conn.close()


class PipeClass(object):

    def __init__(self, pipe_connection):
        self.exiting = False
        self.conn = pipe_connection

    def main_process(self):
        while not self.exiting:
            if self.conn.poll() is True:
                in_val = self.conn.recv()
                self.receive_func(in_val)
            out_val = self.send_func()
            if out_val is not None:
                self.conn.send(out_val)
        while self.conn.recv() != 'end':
            self.exit_loop()
        self.conn.close()

    def soft_exit(self):
        self.exiting = True

    def receive_func(self, data):
        pass

    def send_func(self):
        pass

    def exit_loop(self):
        pass



if __name__ == '__main__':
    parent_conn, child_conn = Pipe(duplex=True)
    p = Process(target = tensor_loop, args = (child_conn,))
    p.start()
    j=0
    while j<200000:
        if parent_conn.poll() is True:
            print(parent_conn.recv())
        j+=1
        parent_conn.send(j)
    parent_conn.send('end')
    p.join()
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
from socket import socket
from datetime import datetime
from time import sleep
BLOCK_SIZE = 128
PAD = '#'
GLO_TIMEOUT = None 
SV_TIMEOUT = 40
CL_TIMEOUT = 10
PORT = 61666
ENCODING = 'utf-8'
MAX_ATTEMPTS = 64

def format_time(time : datetime):
    return str(time.strftime("%Y-%m-%d_%H-%M-%S"))


def padded_send(connection : socket, msg ,buffer : int = BLOCK_SIZE ,padchar : str = PAD) -> None:
    assert isinstance(msg,(str,bytes,bytearray)) 
    
    while msg:
        block = msg[:buffer]
        msg = msg[buffer:]
        
        if not block:
            break

        if isinstance(block,str):
            out_msg = block.ljust(buffer,padchar).encode(ENCODING)

        elif isinstance(block,(bytearray,bytes)):
            out_msg = msg

        connection.sendall(out_msg)
        print('Sent: ',out_msg)


def receive_normal(connection: socket,expected : int):
    print('Receiving...')
    connection.settimeout(GLO_TIMEOUT)
    msg = connection.recv(BLOCK_SIZE)
    while len(msg) < expected:
        append = connection.recv(BLOCK_SIZE)
        msg += append
        if not append:
            # raise TimeoutError
                #let's try not to resort to violence here ^^
            break
        
    return msg 
    

def receive_strip(connection: socket,expected : int = BLOCK_SIZE,padchar : str = PAD) -> str:
    print('Receiving...')
    msg = ''.encode(ENCODING)
    msg_str = ''
    i = 0
    while len(msg) < expected:
        append = connection.recv(BLOCK_SIZE)
        msg += append
        append_str = append.decode(ENCODING).strip(padchar)
        msg_str += append_str
        print(f'APPENDED:{append_str}')
        if not append:
            raise OSError('Connection might be closed')


    print('RECEIVED',msg_str)
    return msg_str


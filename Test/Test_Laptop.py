import socket                

s = socket.socket()          
print("Socket successfully created")

port = 12345                

s.bind(('192.168.137.1', port))         
print("socket binded to %s" %(port))

s.listen(5)      
print("socket is listening")           

while True: 

   c, addr = s.accept()      
   print('Got connection from', addr )

   send_string = "Hello"
   c.send(send_string.encode()) 

   c.close() 
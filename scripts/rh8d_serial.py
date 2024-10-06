#!/usr/bin/env python

from typing import Optional, Union
from dynamixel_sdk import PortHandler, PacketHandler

class RH8DSerial():

    PROTOCOL_VERSION  = 2.0 
    COMM_SUCCESS      = 0                            
    COMM_TX_FAIL      = -1001                        
    SERVO_MODEL_MIN   = 409
    SERVO_MODEL       = 410
    MODEL             = 0
    MIN_POS           = 0
    MAX_POS           = 4095
    POS_GOAL_REG      = 30
    PRES_POS_REG      = 36
    IDS = {'palm_flex': 32, 'palm_abd': 33, 'thumb_abd': 34, 'thumb_flex': 35, 'index_flex': 36, 'middle_flex': 37, 'ring_flex': 38}

    def __init__(self, 
                 port: Optional[str]="/dev/ttyUSB1",
                 baud: Optional[int]=1000000,
                 ) -> None:
        
        self.port_handler = PortHandler(port)
        self.pckt_handler = PacketHandler(self.PROTOCOL_VERSION)
        if self.port_handler.openPort():
            print("Opened the Dynamixel port")
        else:
            print("Failed to open the Dynamixel port", port)

        if self.port_handler.setBaudRate(baud):
            print("Changed the baudrate")
        else:
            print("Failed to change the baudrate", baud)

        self.pres_ids = self.scan()

    def setMinPos(self, id: int) -> None:
        self.setpos(id, self.MIN_POS)

    def setMaxPos(self, id: int) -> None:
        self.setpos(id, self.MAX_POS)

    def getpos(self, id: int) -> Union[int, None]:
        if not id in self.pres_ids:
            print("Reading id not present")
            return None
        if id in self.pres_ids:
            position = self.pckt_handler.read2ByteTxRx(self.port_handler, id, self.PRES_POS_REG)
            return position[0]

    def setpos(self, id: int, val: int) -> None:
        if not id in self.pres_ids:
            print("Writing id not present")
            return
        cmd = max(min(self.MAX_POS, val), self.MIN_POS)
        if cmd != val:
            print("Restricted position", val, "to", cmd)
        self.pckt_handler.write2ByteTxRx(self.port_handler, id, self.POS_GOAL_REG, cmd)

    def scan(self) -> list:
        ids = []
        for id in self.IDS.values():
            dxl_model_number, dxl_comm_result, dxl_error = self.pckt_handler.read2ByteTxRx(self.port_handler, id, self.MODEL)
            if dxl_comm_result==0:
                print(f"Found {id}")
            else:
                print(f"id: {id}, model: {dxl_model_number}, result: {dxl_comm_result}, err: {dxl_error}")
            if dxl_model_number >= self.SERVO_MODEL_MIN:
                ids.append(id)
        return ids
    
if __name__ == "__main__":
    import time
    s = RH8DSerial()
    id = s.IDS['index_flex']

    while 1:
        s.setMinPos(id)
        while not s.getpos(id) <= 100:
            print(s.getpos(id))
            pass

        time.sleep(1)
        for pos in range(100, s.MAX_POS, 100):
            s.setpos(id, pos)
            time.sleep(0.1)

        time.sleep(2)
        s.setMinPos(id)




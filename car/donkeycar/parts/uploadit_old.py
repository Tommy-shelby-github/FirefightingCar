import paramiko
from PIL import Image
import time
from numpy import double, float64
from donkeycar.utils import arr_to_img    #放在model里面
#上传加文件读取
class SFTP():
    """
    实现ssh远程登陆，并且上传下载文件
    """

    def __init__(self):
        self.angle_x=0.0
        self.angle_y=1.0
        try:
            conn = paramiko.Transport(("59.110.163.177", 22))
        except Exception as e:
            print(e)
        else:
            # 用户名，用户密码
            self.name = "root"
            passwd = "cmj123Aa@"
            try:
                # 尝试与远程服务器连接
                conn.connect(username = self.name, password = passwd)
                self.sftp_ob = paramiko.SFTPClient.from_transport(conn)
            except Exception as e:
                # 失败则打印原因
                print(e)
                return
            else:
                print("连接成功")
    def run(self):
        pass
    def update(self):
        while(1):
            self.run()
    def run_threaded(self,imgarr):
        if imgarr is None:
            pass
        else:
            #time.sleep(0.2)
            image = arr_to_img(imgarr)
            image.save("image_20.jpg")
        time.sleep(0.5)
        self.sftp_ob.put("image_20.jpg", "/root/modelpp/img/image_20.jpg")
        self.sftp_ob.get("/root/modelpp/angle.txt","angle.txt")
        fp = open('angle.txt')        
        line=fp.readline()
        if line:
            item = [i for i in line.split()]
            self.angle_x=float64(item[0])
            self.angle_y=float64(item[1])
        fp.close()
        print(self.angle_x,self.angle_y)
        return self.angle_x,self.angle_y
    
    def shutdown(self):
        pass
        
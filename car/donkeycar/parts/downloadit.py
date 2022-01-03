import paramiko
from numpy import double, float64
import time
#上传加文件读取
class SFTP2():
    """
    实现ssh远程登陆，并且上传下载文件
    """

    def __init__(self):
        self.angle_x=0.0
        self.angle_y=1.0
        try:
            conn = paramiko.Transport(("59.110.160.248", 22))#59.110.163.177
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
        time.sleep(0.1)
        self.sftp_ob.get("/root/modelpp/angle.txt","angle.txt")
        fp = open('angle.txt')
        line=fp.readline()
        if line:
            item = [i for i in line.split()]
            self.angle_x=float64(item[0])
            self.angle_y=float64(item[1])
        fp.close()
        print(self.angle_x,self.angle_y)
    def update(self):
        while(1):
            self.run()
    def run_threaded(self):
        return self.angle_x,self.angle_y

    def shutdown(self):
        pass
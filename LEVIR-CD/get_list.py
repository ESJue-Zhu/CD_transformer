import os
import shutil

# def copy_file(srcfile, dstpath):  # 复制函数
#     if not os.path.isfile(srcfile):
#         print("%s not exist!" % (srcfile))
#     else:
#         fpath, fname = os.path.split(srcfile)  # 分离文件名和路径
#         if not os.path.exists(dstpath):
#             os.makedirs(dstpath)  # 创建路径
#         shutil.copy(srcfile, dstpath + fname)  # 复制文件
#         print("copy %s -> %s" % (srcfile, dstpath + fname))

def write_file(type,path):
    '''
    往list中写入txt文件
    :param type: train , val , test
    :param path: 数据集路径
    :return:
    '''
    filepath = path + '/' + type + '/label'  # 数据集路径
    filenames = os.listdir(filepath)  # 数据集文件名
    txtname = './list/'+ type + '.txt'  # 要写入的txt路径+文件名
    with open(txtname, 'w') as f:
        f.close()
    with open(txtname, 'a') as f:
        for i in filenames:
            f.write(i + '\n')
        f.close()
    print(type + '.txt写入list完成')

path = "D:\Auto_Change_Detection\LEVIR-CD"
# 训练数据
write_file('train',path)

# 验证数据
write_file('val',path)

# 测试数据
write_file('test',path)
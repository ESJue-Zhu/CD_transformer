import os
import gdal
import numpy as np
import torch

from misc.imutils import save_image
from models.networks import *


class CDEvaluator():

    def __init__(self, args):

        self.n_class = args.n_class
        # define G
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)

        self.device = torch.device("cuda:%s" % args.gpu_ids[0]
                                   if torch.cuda.is_available() and len(args.gpu_ids) > 0
                                   else "cpu")

        print(self.device)

        self.checkpoint_dir = args.checkpoint_dir

        self.pred_dir = args.output_folder
        os.makedirs(self.pred_dir, exist_ok=True)

    def load_checkpoint(self, checkpoint_name='best_ckpt.pt'):

        if os.path.exists(os.path.join(self.checkpoint_dir, checkpoint_name)):
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, checkpoint_name),
                                    map_location=self.device)

            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])
            self.net_G.to(self.device)
            # update some other states
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

        else:
            raise FileNotFoundError('no such checkpoint %s' % checkpoint_name)
        return self.net_G

    def _visualize_pred(self):
        pred = torch.argmax(self.G_pred, dim=1, keepdim=True)
        pred_vis = pred * 255
        return pred_vis

    def _forward_pass(self, batch):
        self.batch = batch
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)
        self.shape_h = img_in1.shape[-2]
        self.shape_w = img_in1.shape[-1]
        self.G_pred = self.net_G(img_in1, img_in2)
        return self._visualize_pred()

    def eval(self):
        self.net_G.eval()

    def _save_predictions(self):
        """
        保存模型输出结果，二分类图像
        """

        preds = self._visualize_pred()
        name = self.batch['name']
        for i, pred in enumerate(preds):
            file_name = os.path.join(
                self.pred_dir, name[i].replace('.jpg', '.png'))
            pred = pred[0].cpu().numpy()
            save_image(pred, file_name)

            # TODO 筛选mask面积大于阈值的图片
            threshold = 1000  # 阈值
            if np.sum(pred/255) > threshold:
                aim_dir = self.pred_dir + '/aim'  # 目标路径
                aim_name = os.path.join(
                    aim_dir, name[i].replace('.jpg', '.png'))  # 目标文件名
                save_image(pred, aim_name)  # 保存mask
                source_dir = 'samples/A'  # 源文件路径
                source_name = os.path.join(
                    source_dir, name[i].replace('.jpg', '.png'))  # 源文件名
                in_ds = gdal.Open(source_name)
                ori_transform = in_ds.GetGeoTransform()
                top_left_x = ori_transform[0]  # 左上角x坐标
                top_left_y = ori_transform[3]  # 左上角y坐标
                with open("samples/predict/aim/aim.txt", "a") as f:  # 写入aim.txt
                    f.write(name[i].replace('.jpg', '.png') + ' x:' + str(top_left_x) + ' y:' + str(top_left_y) + '\n')
                    print('写入文件名和坐标成功')
                    f.close()

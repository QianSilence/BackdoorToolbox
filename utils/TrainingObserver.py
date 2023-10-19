import torch
import os
import time
from .compute import compute_accuracy
from .interact.log import Log
from core.base.Observer import Observer
class SaveModelInTraining(Observer):
    def __init__(self,save_epoch_interval):
        self.save_epoch_interval = save_epoch_interval
    def work(self,context):
        self.save_model_in_train(context)

    def save_model_in_train(self,context):
        assert "model" in context, 'This machine has no model!'
        assert "epoch" in context, 'This machine has no epoch!'
        assert "work_dir" in context, 'This machine has no work_dir!'
        assert "log_path" in context, 'This machine has no log_path!'
        model = context["model"]
        epoch = context["epoch"]
        work_dir = context["work_dir"] 
        log_path = context["log_path"] 
        log = Log(log_path)
        if (epoch + 1) % self.save_epoch_interval == 0:
            model.eval()
            # model.model = model.model.cpu()
            ckpt_model_filename = "ckpt_epoch_" + str(epoch+1) + ".pth"
            ckpt_model_path = os.path.join(work_dir, ckpt_model_filename)
            torch.save(model.model.state_dict(), ckpt_model_path)
            msg = "==========Save model in training process==========\n" + \
                time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                f"save model parameter into {ckpt_model_path}\n"
            log(msg)

class TestInTraining(Observer):
    def __init__(self,test_epoch_interval):
        self.test_epoch_interval = test_epoch_interval
    def work(self,context):
         self.test_in_train(context)

    def test_in_train(self,context):
        assert "model" in context, 'This machine has no model!'
        assert "epoch" in context, 'This machine has no epoch!'
        assert "device" in context, 'This machine has no cuda devices!'
        assert "work_dir" in context, 'This machine has no work_dir!'
        assert "log_path" in context, 'This machine has no log_path!'
        assert "last_time" in context, 'This machine has no last_time!'
       
        model = context["model"]
        epoch = context["epoch"]
        work_dir = context["work_dir"] 
        log_path = context["log_path"] 
        last_time = context["last_time"]
        log = Log(log_path)

        if (epoch + 1) % self.test_epoch_interval == 0:
            # test result on benign test dataset
            predict_digits, labels = model.test_in_training()
            total_num = labels.size(0)
            prec1, prec5 = compute_accuracy(predict_digits, labels, topk=(1, 5))
            top1_correct = int(round(prec1.item() / 100.0 * total_num))
            top5_correct = int(round(prec5.item() / 100.0 * total_num))
            msg = "==========Test in training process==========\n" + \
                time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num},\
                Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, \
                time: {time.time()-last_time}\n"
            log(msg)

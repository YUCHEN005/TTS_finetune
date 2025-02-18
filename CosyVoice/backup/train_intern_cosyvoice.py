import sys
import os
import torch
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import logging
import time
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_scheduler
from cosyvoice.cli.cosyvoice import CosyVoice2
os.system('mkdir -p results/')

# 将 `Matcha-TTS` 路径添加到 Python 路径
os.system('export PYTHONPATH=/fs-computility/INTERN6/shared/yuchen/CosyVoice_qwen0.5b/third_party/Matcha-TTS')

# 初始化日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 自定义数据集
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            'text_token': sample['text_token'].squeeze(0),
            'text_token_len': sample['text_token_len'],
            'speech_token': sample['speech_token'].squeeze(0),
            'speech_token_len': sample['speech_token_len'],
        }

def forward_prop(batch_data, model, device):
    output = model(batch_data, device)
    return output['loss'], output['acc']

def eval(model, eval_loader, device):
    eval_loss, eval_acc = 0.0, 0.0
    with torch.no_grad():
        for batch_data in eval_loader:
            for key in batch_data:
                batch_data[key] = batch_data[key].to(device)
            loss, acc = forward_prop(batch_data, model, device)
            eval_loss += loss.item()
            eval_acc += acc.item()
    return eval_loss / len(eval_loader), eval_acc / len(eval_loader)

# 训练函数
def train(world_size):
    dist.init_process_group("nccl", world_size=world_size)
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)

    device = torch.device(f"cuda:{rank}")
    logger.info(f"Rank {rank}: Using device {device}")

    # 设置超参数
    batch_size = 8
    num_epochs = 20
    learning_rate = 4e-5
    weight_decay = 1e-2
    
    # 加载模型
    logger.info(f"Rank {rank}: Loading model...")
    model = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False, device=device).model.llm.to(device)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    # 加载训练数据
    logger.info(f"Rank {rank}: Loading training data...")
    data = torch.load('../CosyVoice_qwen0.5b/Libriheavy_cosy/proc_libriheavy_001_398.0h.pt', map_location=device)
    train_data = data[:-1000]
    eval_data = data[-1000:]

    train_dataset = CustomDataset(train_data)
    eval_dataset = CustomDataset(eval_data)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    eval_sampler = DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=collate_fn)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, sampler=eval_sampler, collate_fn=collate_fn)

    # 设置优化器
    steps_per_epoch = len(train_data) // (batch_size*world_size)
    num_total_steps = num_epochs * steps_per_epoch
    epoch_gamma = 0.7
    step_gamma = epoch_gamma ** (1 / steps_per_epoch)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    # scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=num_total_steps)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=step_gamma)
    scheduler = get_scheduler(
        "linear",  # 线性 warmup + 线性退火
        optimizer=optimizer,
        num_warmup_steps=num_total_steps * 0.1,
        num_training_steps=num_total_steps,
    )
    
    # 训练循环
    logger.info(f"Rank {rank}: Starting training...")
    start_time = time.time()
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)

        train_loss, train_acc = 0.0, 0.0
        for i, batch_data in enumerate(train_loader):
            for key in batch_data:
                batch_data[key] = batch_data[key].to(device)

            # 前向传播
            optimizer.zero_grad()
            model.train()
            loss, acc = forward_prop(batch_data, model, device)
            if i % 100 == 0:
                model.eval()
                loss_e, acc_e = eval(model, eval_loader, device)
                if rank == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    logger.info(f"Epoch {epoch+1}, step {i}/{len(train_data)//(batch_size*world_size)}: lr = {current_lr:.8f}, train_loss = {loss.item():.3f}, train_acc = {acc.item():.3f}, eval_loss = {loss_e:.3f}, eval_acc = {acc_e:.3f}")

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            train_acc += acc.item()

        model.eval()
        eval_loss, eval_acc = eval(model, eval_loader, device)
        
        if rank == 0:  # 仅主进程打印日志和保存模型
            logger.info('='*25 + f'  Epoch {epoch + 1}/{num_epochs}, LR: {current_lr:.8f}, Time: {(time.time() - start_time)/3600:.3f} h  ' + '='*25)
            logger.info(f"train_loss = {train_loss/len(train_loader):.3f}, train_acc = {train_acc/len(train_loader):.3f}")
            logger.info(f"eval_loss =  {eval_loss:.3f}, eval_acc =  {eval_acc:.3f}")
            logger.info('='*50)

            torch.save(model.state_dict(), f'results/epoch_{epoch + 1:03d}.pth')
        
    # 清理分布式环境
    dist.destroy_process_group()

def collate_fn(batch):
    text_tokens = [sample['text_token'] for sample in batch]
    text_lengths = [sample['text_token_len'] for sample in batch]
    speech_tokens = [sample['speech_token'] for sample in batch]
    speech_lengths = [sample['speech_token_len'] for sample in batch]

    return {
        'text_token': pad_sequence(text_tokens, batch_first=True, padding_value=0),
        'text_token_len': torch.stack(text_lengths),
        'speech_token': pad_sequence(speech_tokens, batch_first=True, padding_value=0),
        'speech_token_len': torch.stack(speech_lengths),
    }

# 主函数
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    train(world_size)

